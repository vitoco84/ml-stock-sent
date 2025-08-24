from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware

from app.api.classes import NewsHistoryResponse, PredictionRequest, PredictionResponse, PriceHistoryResponse
from app.api.settings import get_settings
from app.api.utils import _ollama_alive, LimitUploadSizeMiddleware, to_dict
from config.config import Config
from src.data import get_news_history, get_price_history
from src.features import generate_full_feature_row
from src.llm import enrich_news_with_generated
from src.logger import get_logger
from src.sentiment import FinBERT
from src.train import ModelTrainer


logger = get_logger(__name__)

config = Config(Path("config/config.yaml"))
settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context.
    - Loads configuration
    - Initializes FinBERT sentiment model
    - Loads trained prediction model and preprocessor
    - Loads NEWS_API_KEY from .env
    - Stores initialized objects in `app.state` for later access
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Initializing FinBERT on {device}")

    sentiment_model = FinBERT(config, device=device)

    model_path = Path(config.data.models_dir) / "linreg.pkl"
    if not model_path.exists():
        raise RuntimeError(f"Model file not found at {model_path}")
    model, preprocessor, y_scaler, y_scale = ModelTrainer.load(str(model_path))

    app.state.news_api_key = settings.news_api_key
    app.state.sentiment_model = sentiment_model
    app.state.model = model
    app.state.preprocessor = preprocessor
    app.state.y_scaler = y_scaler
    app.state.y_scale = bool(y_scale)

    yield

app = FastAPI(
    root_path=settings.api_root_path,
    title="Stock Prediction API",
    description="Predict stock prices using historical prices, news sentiment, and FinBERT.",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[str(o) for o in settings.cors_origins],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)
app.add_middleware(LimitUploadSizeMiddleware)

@app.get("/healthz")
def healthz(): return {"ok": True}

@app.get("/price-history", response_model=PriceHistoryResponse)
def fetch_price_history(
        symbol: str = Query(..., description="Ticker symbol, e.g., AAPL, ^DJI"),
        end_date: str = Query(..., description="End date in YYYY-MM-DD format"),
        days: int = Query(90, description="Number of calendar days to look back")
):
    """
    Fetch historical stock price data for a given symbol.

    - **symbol**: Ticker symbol (AAPL, TSLA, ^DJI, etc.)
    - **end_date**: End date in YYYY-MM-DD format
    - **days**: Number of calendar days to fetch (default: 90)

    Returns JSON with price rows or an error message.
    """
    try:
        df = get_price_history(symbol, end_date, days)
        if df.empty:
            raise HTTPException(404, "No price data returned. Check the symbol or date range.")

        df["date"] = df["date"].dt.strftime("%Y-%m-%d")
        records = df.to_dict(orient="records")
        logger.info(f"Head of dataframe:\n{df.head()}")
        logger.debug(f"Sample record: {records[0]}")
        return {"price": records}
    except Exception:
        logger.exception("fetch_price_history failed")
        raise HTTPException(500, "Internal server error")

@app.get("/news-history", response_model=NewsHistoryResponse)
def fetch_news_history(
        request: Request,
        query: str = Query(..., description="Search keyword, e.g., Apple, Tesla"),
        end_date: str = Query(..., description="End date in YYYY-MM-DD format"),
        days: int = Query(7, description="Number of calendar days to look back")
):
    """
    Fetch recent news headlines using the NewsAPI.

    - **query**: Keyword to search (e.g., 'Apple')
    - **end_date**: End date in YYYY-MM-DD format
    - **days**: Number of days to look back

    Returns JSON with news rows or an error message.
    """
    try:
        api_key = request.app.state.news_api_key
        if not api_key:
            raise HTTPException(500, "Missing NEWS_API_KEY environment variable")

        df = get_news_history(query, end_date, days, api_key, settings.news_api_base)
        if df.empty:
            return {"news": [], "message": "No news found."}
        return {"news": df.to_dict(orient="records")}
    except Exception:
        logger.exception("fetch_news_history failed")
        raise HTTPException(500, "Internal server error")

@app.post("/predict-raw", response_model=PredictionResponse, response_model_exclude_none=True)
def post_predict_from_raw(
        request_body: PredictionRequest,
        request: Request,
        enrich: bool = Query(False, description="Generate missing headlines using local LLM"),
        horizon: int = Query(30, ge=1, description="How many horizons to return"),
        return_path: bool = Query(True, description="Whether to return the full H-step path"),
        symbol: str = Query(..., description="Ticker symbol for context (e.g., AAPL)")
):
    """
    Predict the next stock price using:
    - Historical price data
    - News headlines (optionally enriched with an LLM)
    - FinBERT sentiment analysis
    - A pre-trained regression model

    Returns
    - **log_return**: Predicted log return
    - **current_price**: Latest price
    - **predicted_price**: Predicted next price
    """
    horizon = min(horizon, 30)
    logger.info("Received prediction request")

    sentiment_model = request.app.state.sentiment_model
    model = request.app.state.model
    preprocessor = request.app.state.preprocessor

    # --- PRICE ---
    try:
        if not getattr(request_body, "price", None):
            raise HTTPException(422, "`price` is required and must be a non-empty list.")
        price_rows = [to_dict(row) for row in request_body.price]
    except HTTPException:
        raise
    except Exception:
        logger.exception("Invalid `price` payload")
        raise HTTPException(
            422,
            "`price` payload malformed. Expect a list of rows with "
            "{date, open, high, low, close, adj_close, volume}."
        )

    price_df = pd.DataFrame(price_rows)

    required_cols = {"date", "open", "high", "low", "close", "adj_close", "volume"}
    missing = required_cols - set(price_df.columns)
    if missing:
        raise HTTPException(422, f"Missing required price columns: {sorted(missing)}")

    price_df["date"] = pd.to_datetime(price_df["date"]).dt.normalize()
    price_df = price_df.sort_values("date").drop_duplicates(subset=["date"], keep="last")

    if price_df.empty:
        raise HTTPException(422, "Price data is empty or invalid.")
    if len(price_df) > 2000:
        raise HTTPException(400, "Price data exceeds 2000-row limit.")
    if (price_df["date"].max() - price_df["date"].min()).days > 365 * 5:
        raise HTTPException(400, "Price data spans more than 5 years.")

    price_dates = price_df["date"].dt.strftime("%Y-%m-%d").tolist()
    logger.info(f"Price DF:\n{price_df.tail()}")

    # --- NEWS ---
    news_payload = getattr(request_body, "news", None) or []
    if len(news_payload) > 0:
        try:
            news_df = pd.DataFrame([to_dict(row) for row in request_body.news])

            if not news_df.empty and len(news_df) > 2000:
                raise HTTPException(400, detail="News data exceeds 2000-row limit.")

            news_df["date"] = pd.to_datetime(news_df["date"]).dt.normalize()
            logger.info(f"News DF:\n{news_df.tail()}")
        except Exception:
            logger.exception("Invalid `news` payload")
            raise HTTPException(422, "`news` payload malformed. Expect list of {date, headline}.")
    else:
        news_df = pd.DataFrame(columns=["date", "headline"])
        logger.info("No initial news provided.")

    # Trim to keep latency bounded
    if not news_df.empty:
        number_of_row, amount_per_day = 1000, 20
        news_df = (
            news_df
            .sort_values(["date"])
            .groupby("date", as_index=False)
            .head(amount_per_day)
            .tail(number_of_row))

    ollama_base = settings.ollama_base
    ollama_ok = _ollama_alive(ollama_base)

    # --- ENRICH NEWS (if requested) ---
    if enrich and ollama_ok:
        logger.info("Enrich flag is ON — generating missing headlines via LLM")
        real_news = news_df.to_dict(orient="records")
        enriched_news = enrich_news_with_generated(
            price_dates=price_dates,
            real_news=real_news,
            symbol=symbol,
            url_llm=f"{ollama_base.rstrip('/')}/api/generate",
            model_llm=settings.ollama_model
        )
        news_df = pd.DataFrame(enriched_news)
        if not news_df.empty:
            news_df["date"] = pd.to_datetime(news_df["date"]).dt.normalize()
            logger.info(f"Enriched News DF:\n{news_df.tail()}")
        else:
            logger.info("Enrichment produced no headlines (continuing with empty news).")
    else:
        logger.info("Enrich flag is OFF or LLM unavailable — skipping headline generation")

    # --- FEATURE GENERATION ---
    try:
        if news_df.empty:
            logger.info("Skipping sentiment: no news to process.")
            feature_row = generate_full_feature_row(price_df, pd.DataFrame(), None, horizon)
        else:
            feature_row = generate_full_feature_row(price_df, news_df, sentiment_model, horizon)
    except Exception:
        logger.exception("Feature generation failed")
        raise HTTPException(500, "Failed to generate features from price/news data.")

    logger.debug(f"Feature row:\n{feature_row}")

    # --- PREDICTION ---
    try:
        X = preprocessor.transform(feature_row)
        yhat = model.predict(X)  # shape (1, H) or (H,) or sometimes (N,H)
        yhat = np.asarray(yhat, dtype=float)
        if yhat.ndim == 1:
            yhat = yhat.reshape(1, -1)
    except Exception:
        logger.exception("Model prediction failed")
        raise HTTPException(500, "Prediction failed.")

    yhat = np.asarray(yhat).reshape(1, -1)

    # Inverse-transform if y was scaled during training
    if getattr(request.app.state, "y_scale", False) and getattr(request.app.state, "y_scaler", None) is not None:
        yhat = request.app.state.y_scaler.inverse_transform(yhat)

    # Clamp horizon to available outputs
    H = int(min(horizon, yhat.shape[1]))
    current_price = float(price_df["adj_close"].iloc[-1])

    # Headline (next day) stays for compatibility
    log_return = float(yhat[0, 0])
    predicted_price = current_price * float(np.exp(log_return))

    response_kwargs = {
        "horizon": H,
        "log_return": log_return,
        "current_price": current_price,
        "predicted_price": predicted_price
    }

    if return_path:
        lr_path = yhat[0, :H]
        price_path = current_price * np.exp(np.cumsum(lr_path))
        from pandas.tseries.offsets import BDay
        future_dates = pd.bdate_range(price_df["date"].iloc[-1] + BDay(1), periods=H)

        response_kwargs.update({
            "log_return_path": lr_path.tolist(),
            "predicted_price_path": [float(x) for x in price_path],
            "predicted_dates": future_dates.strftime("%Y-%m-%d").tolist(),
            "last_date": pd.to_datetime(price_df["date"].iloc[-1]).date(),
        })

    return PredictionResponse(**response_kwargs)
