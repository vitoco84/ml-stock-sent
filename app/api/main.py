from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pandas.tseries.offsets import BDay

from app.api.classes import (
    NewsHistoryResponse,
    PredictionRequest,
    PredictionResponse,
    PriceHistoryResponse,
)
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
    """Initialize and tear down global resources (FinBERT, model, preprocessor)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Initializing FinBERT on {device}")

    sentiment_model = FinBERT(config, device=device, max_embedding_dims=17)

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
    description="Predict stock price deltas (AdjClose_{t+1} − AdjClose_t) using prices, news, and FinBERT sentiment.",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[str(o) for o in settings.cors_origins],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"]
)
app.add_middleware(LimitUploadSizeMiddleware)

def _process_price_df(request_body: PredictionRequest) -> pd.DataFrame:
    """Validate and normalize price data from request."""
    if not getattr(request_body, "price", None):
        raise HTTPException(422, "`price` is required and must be a non-empty list.")

    try:
        price_rows = [to_dict(row) for row in request_body.price]
        df = pd.DataFrame(price_rows)
    except Exception:
        logger.exception("Invalid `price` payload")
        raise HTTPException(
            422,
            "`price` payload malformed. Expect rows with {date, open, high, low, close, adj_close, volume}."
        )

    required = {"date", "open", "high", "low", "close", "adj_close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise HTTPException(422, f"Missing required price columns: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last")

    if df.empty:
        raise HTTPException(422, "Price data is empty or invalid.")
    if len(df) > 2000:
        raise HTTPException(400, "Price data exceeds 2000-row limit.")
    if (df["date"].max() - df["date"].min()).days > 365 * 5:
        raise HTTPException(400, "Price data spans more than 5 years.")

    return df

def _process_news_df(
        request_body: PredictionRequest,
        price_dates: list[str],
        *,
        enrich: bool,
        pad_neutral: bool,
        ignore_news: bool,
        symbol: str
) -> pd.DataFrame:
    """Validate and normalize news data, optionally enrich or pad."""
    news_payload = getattr(request_body, "news", None) or []
    if len(news_payload) > 2000:
        raise HTTPException(400, "News data exceeds 2000-row limit.")

    if news_payload:
        try:
            df = pd.DataFrame([to_dict(row) for row in news_payload])
            df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        except Exception:
            logger.exception("Invalid `news` payload")
            raise HTTPException(422, "`news` payload malformed. Expect list of {date, headline}.")
    else:
        df = pd.DataFrame(columns=["date", "headline"])

    if ignore_news:
        return pd.DataFrame(columns=["date", "headline"])

    if not df.empty:
        df = (
            df.sort_values(["date"])
            .groupby("date", as_index=False)
            .head(20)
            .tail(1000)
        )

    ollama_base = settings.ollama_base
    ollama_ok = _ollama_alive(ollama_base)

    if enrich and ollama_ok:
        if df.empty:
            raise HTTPException(422, "Enrich requires ≥1 seed headline.")

        real_news: list[dict[str, str]] = [
            {str(k): str(v) for k, v in row.items()} for row in df.to_dict(orient="records")
        ]

        enriched = enrich_news_with_generated(
            price_dates=price_dates,
            real_news=real_news,
            symbol=symbol,
            url_llm=f"{ollama_base.rstrip('/')}/api/generate",
            model_llm=settings.ollama_model
        )
        df = pd.DataFrame(enriched)
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    if pad_neutral and (df.empty or len(df) < 2):
        raise HTTPException(422, "Pad-neutral requires ≥2 headlines and does not generate news.")

    return df

def _generate_features(
        price_df: pd.DataFrame,
        news_df: pd.DataFrame,
        sentiment_model: Any,
        horizon: int,
        pad_neutral: bool
) -> pd.DataFrame:
    """Generate model-ready feature row."""
    try:
        if news_df.empty:
            return generate_full_feature_row(price_df, pd.DataFrame(), None, forecast_horizon=horizon)

        return generate_full_feature_row(
            price_df,
            news_df,
            sentiment_model,
            forecast_horizon=horizon,
            fill_missing_neutral=pad_neutral,
            max_embedding_dims=17
        )
    except Exception:
        logger.exception("Feature generation failed")
        raise HTTPException(500, "Failed to generate features from price/news data.")

def _make_prediction(
        feature_row: pd.DataFrame,
        model: Any,
        preprocessor: Any,
        y_scaler: Any,
        y_scale: bool,
        price_df: pd.DataFrame,
        horizon: int,
        return_path: bool
) -> PredictionResponse:
    """Run model prediction and build response (aligned with direct-delta training)."""
    try:
        X = preprocessor.transform(feature_row)
        yhat = np.asarray(model.predict(X), dtype=float)
        if yhat.ndim == 1:
            yhat = yhat.reshape(1, -1)
    except Exception:
        logger.exception("Model prediction failed")
        raise HTTPException(500, "Prediction failed.")

    H = min(horizon, yhat.shape[1])
    current_price = float(price_df["adj_close"].iloc[-1])

    # Horizon=1 is just the first direct delta
    delta_1 = float(yhat[0, 0])
    predicted_price = current_price + delta_1

    response_kwargs: dict[str, Any] = {
        "horizon": H,
        "current_price": current_price,
        "delta_price": delta_1,
        "predicted_price": predicted_price,
    }

    if return_path:
        delta_path = yhat[0, :H]
        price_path = current_price + delta_path
        future_dates = pd.bdate_range(price_df["date"].iloc[-1] + BDay(1), periods=H)

        response_kwargs.update(
            delta_price_path=delta_path.tolist(),
            predicted_price_path=price_path.tolist(),
            predicted_dates=future_dates.strftime("%Y-%m-%d").tolist(),
            last_date=pd.to_datetime(price_df["date"].iloc[-1]).date(),
        )

    return PredictionResponse(**response_kwargs)

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.get("/price-history", response_model=PriceHistoryResponse)
def fetch_price_history(
        symbol: str = Query("^DJI", description="Ticker symbol, e.g., AAPL, ^DJI"),
        end_date: str = Query(..., description="End date in YYYY-MM-DD format"),
        days: int = Query(90, ge=1, le=365, description="Number of business days to look back")
):
    """Fetch historical stock price data for a given symbol."""
    try:
        if days > 365:
            raise HTTPException(400, "Max look-back is 365 business days.")

        df = get_price_history(symbol, end_date, days)
        if df.empty:
            raise HTTPException(404, "No price data returned. Check the symbol or date range.")

        df["date"] = df["date"].dt.strftime("%Y-%m-%d")
        return {"price": df.to_dict(orient="records")}
    except Exception:
        logger.exception("fetch_price_history failed")
        raise HTTPException(500, "Internal server error")

@app.get("/news-history", response_model=NewsHistoryResponse)
def fetch_news_history(
        request: Request,
        query: str = Query(..., description="Search keyword, e.g., Apple, Tesla"),
        end_date: str = Query(..., description="End date in YYYY-MM-DD format"),
        days: int = Query(7, ge=1, le=29, description="Number of calendar days to look back")
):
    """Fetch recent news headlines using the NewsAPI."""
    try:
        if days > 29:
            raise HTTPException(400, "Max look-back is 29 days.")

        api_key = request.app.state.news_api_key
        if not api_key:
            raise HTTPException(500, "Missing NEWS_API_KEY environment variable")

        df = get_news_history(query, end_date, days, api_key, settings.news_api_base)
        if df.empty:
            return {"news": [], "message": "No articles found."}
        return {"news": df.to_dict(orient="records")}
    except Exception:
        logger.exception("fetch_news_history failed")
        raise HTTPException(500, "Internal server error")

@app.post("/predict-raw", response_model=PredictionResponse, response_model_exclude_none=True)
def post_predict_from_raw(
        request_body: PredictionRequest,
        request: Request,
        enrich: bool = Query(False, description="Generate missing headlines using local LLM"),
        pad_neutral: bool = Query(False, description="Use provided news and neutral-fill gaps (needs ≥2)"),
        ignore_news: bool = Query(False, description="Ignore all news (neutral every day)"),
        horizon: int = Query(30, ge=1, le=30, description="How many horizons to return"),
        return_path: bool = Query(True, description="Whether to return the full H-step path"),
        symbol: str = Query("^DJI", description="Ticker symbol for context (e.g., AAPL)")
) -> PredictionResponse:
    """Predict next price deltas from prices + optional news."""
    if ignore_news and (enrich or pad_neutral):
        raise HTTPException(400, "Invalid strategy: 'ignore_news' cannot be combined with 'enrich' or 'pad_neutral'.")
    if enrich and pad_neutral:
        raise HTTPException(400, "Choose exactly one strategy: enrich OR pad_neutral OR ignore_news.")

    price_df = _process_price_df(request_body)
    price_dates = price_df["date"].dt.strftime("%Y-%m-%d").tolist()

    news_df = _process_news_df(
        request_body, price_dates, enrich=enrich, pad_neutral=pad_neutral, ignore_news=ignore_news, symbol=symbol
    )

    feature_row = _generate_features(
        price_df, news_df, request.app.state.sentiment_model, horizon, pad_neutral
    )

    return _make_prediction(
        feature_row,
        request.app.state.model,
        request.app.state.preprocessor,
        request.app.state.y_scaler,
        request.app.state.y_scale,
        price_df,
        horizon,
        return_path
    )
