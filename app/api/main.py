import os
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request

from app.api.classes import NewsHistoryResponse, PredictionRequest, PriceHistoryResponse
from src.config import Config
from src.data import get_news_history, get_price_history
from src.features import generate_full_feature_row
from src.llm import enrich_news_with_generated
from src.logger import get_logger
from src.sentiment import FinBERT
from src.train import ModelTrainer
from src.utils import is_cuda_available


logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context.
    - Loads configuration
    - Initializes FinBERT sentiment model
    - Loads trained prediction model and preprocessor
    - Loads NEWS_API_KEY from .env
    - Stores initialized objects in `app.state` for later access
    """
    config = Config(Path("config/config.yaml"))

    device = "cuda" if is_cuda_available() else "cpu"
    logger.info(f"Initializing FinBERT on {device}")
    sentiment_model = FinBERT(config, device=device)
    model_path = Path(config.model.path)
    if not model_path.exists():
        raise RuntimeError(f"Model file not found at {model_path}")
    model, preprocessor, y_scaler, y_scale = ModelTrainer.load(str(model_path))

    load_dotenv()
    api_key = os.getenv("NEWS_API_KEY")

    app.state.news_api_key = api_key
    app.state.sentiment_model = sentiment_model
    app.state.model = model
    app.state.preprocessor = preprocessor
    app.state.y_scaler = y_scaler
    app.state.y_scale = y_scale or {}

    yield

app = FastAPI(
    title="Stock Prediction API",
    description="Predict stock prices using historical prices, news sentiment, and FinBERT.",
    version="1.0.0",
    lifespan=lifespan
)

from fastapi.middleware.cors import CORSMiddleware


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],  # Change domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    """Return a simple message to confirm the API is running."""
    return {"message": "API is up and running!"}

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
        df = get_price_history(symbol=symbol, end_date=end_date, days=days)

        if df.empty:
            raise HTTPException(status_code=404, detail="No price data returned. Check the symbol or date range.")

        df["date"] = df["date"].dt.strftime("%Y-%m-%d")
        records = df.to_dict(orient="records")

        logger.info(f"Fetched data types:\n{df.dtypes}")
        logger.info(f"Head of dataframe:\n{df.head()}")
        logger.debug(f"Sample record: {df.to_dict(orient='records')[0]}")

        return {"price": records}


    except Exception as e:
        logger.exception("fetch_price_history failed")
        raise HTTPException(status_code=500, detail="Internal server error")

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
            raise HTTPException(status_code=500, detail="Missing NEWS_API_KEY environment variable")

        df = get_news_history(query=query, end_date=end_date, days=days, api_key=api_key)

        if df.empty:
            return {"news": [], "message": "No news found."}

        return {"news": df.to_dict(orient="records")}

    except Exception as e:
        logger.exception("fetch_news_history failed")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/predict-raw")
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
    logger.info("Received prediction request")

    sentiment_model = request.app.state.sentiment_model
    model = request.app.state.model
    preprocessor = request.app.state.preprocessor

    # --- PRICE ---
    try:
        price_rows = [row.model_dump() for row in request_body.price]
    except Exception:
        raise HTTPException(status_code=422, detail="`price` is required and must be a non-empty list.")

    price_df = pd.DataFrame(price_rows)
    price_df["date"] = pd.to_datetime(price_df["date"]).dt.normalize()
    price_df = price_df.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    price_dates = price_df["date"].dt.strftime("%Y-%m-%d").tolist()
    logger.info(f"Price DF:\n{price_df.tail()}")

    # --- NEWS ---
    news_payload = getattr(request_body, "news", None) or []
    if len(news_payload) > 0:
        news_df = pd.DataFrame([row.model_dump() for row in request_body.news])
        news_df["date"] = pd.to_datetime(news_df["date"]).dt.normalize()
        logger.info(f"News DF:\n{news_df.tail()}")
    else:
        news_df = pd.DataFrame(columns=["date", "rank", "headline"])
        logger.info("No initial news provided.")

    # --- ENRICH NEWS (if requested) ---
    if enrich:
        logger.info("Enrich flag is ON — generating missing headlines via LLM")
        real_news = news_df.to_dict(orient="records")
        enriched_news = enrich_news_with_generated(price_dates, real_news, symbol=symbol)
        news_df = pd.DataFrame(enriched_news)
        if not news_df.empty:
            news_df["date"] = pd.to_datetime(news_df["date"]).dt.normalize()
            logger.info(f"Enriched News DF:\n{news_df.tail()}")
        else:
            logger.info("Enrichment produced no headlines (continuing with empty news).")
    else:
        logger.info("Enrich flag is OFF — skipping headline generation")

    # --- FEATURE GENERATION ---
    try:
        if news_df.empty:
            logger.info("Skipping sentiment: no news to process.")
            feature_row = generate_full_feature_row(price_df, pd.DataFrame(), None)
        else:
            feature_row = generate_full_feature_row(price_df, news_df, sentiment_model)
    except Exception:
        logger.exception("Feature generation failed")
        raise HTTPException(status_code=500, detail="Failed to generate features from price/news data.")

    logger.debug(f"Feature row:\n{feature_row}")

    # --- PREDICTION ---
    try:
        X = preprocessor.transform(feature_row)
        yhat = model.predict(X)  # shape (1, H) or (1,)
    except Exception:
        logger.exception("Model prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed.")

    yhat = np.asarray(yhat).reshape(1, -1)

    # inverse-transform if y was scaled during training
    if getattr(request.app.state, "y_scaler", None) is not None:
        yhat = request.app.state.y_scaler.inverse_transform(yhat)

    # clamp horizon to available outputs
    H = int(min(horizon, yhat.shape[1]))
    current_price = float(price_df["adj_close"].iloc[-1])

    # headline (next day) stays for compatibility
    log_return = float(yhat[0, 0])
    predicted_price = current_price * float(np.exp(log_return))

    resp = {
        "horizon": H,
        "log_return": log_return,
        "current_price": current_price,
        "predicted_price": predicted_price
    }

    if return_path:
        lr_path = yhat[0, :H]  # length H
        price_path = current_price * np.exp(np.cumsum(lr_path))
        from pandas.tseries.offsets import BDay
        future_dates = pd.bdate_range(price_df["date"].iloc[-1] + BDay(1), periods=H)

        resp.update({
            "log_return_path": lr_path.tolist(),
            "predicted_price_path": [float(x) for x in price_path],
            "predicted_dates": future_dates.strftime("%Y-%m-%d").tolist(),
            "last_date": pd.to_datetime(price_df["date"].iloc[-1]).strftime("%Y-%m-%d"),
        })

    return resp
