from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware

from app.api.classes import (
    NewsHistoryResponse,
    PredictionRequest,
    PredictionResponse,
    PriceHistoryResponse,
)
from app.api.service import _generate_features, _make_prediction, _process_news_df, _process_price_df
from app.api.settings import get_settings
from app.api.utils import LimitUploadSizeMiddleware
from config.config import Config
from src.data import get_news_history, get_price_history
from src.logger import get_logger
from src.sentiment import FinBERT
from src.train import ModelTrainer


logger = get_logger(__name__)
cfg = Config.load()
settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and tear down global resources (FinBERT, model, preprocessor)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Initializing FinBERT on {device}")

    sentiment_model = FinBERT(device=device, max_embedding_dims=cfg.runtime.max_sentiment_embeddings)

    model_path = Path(cfg.data.models_dir) / "linreg.pkl"
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
    description="Predict stock price log-returns (log(AdjClose_{t+1}/AdjClose_t)) using prices, news, and FinBERT sentiment.",
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
        ignore_news: bool = Query(False, description="Ignore all news (neutral every day)"),
        return_path: bool = Query(True, description="Whether to return the full H-step path"),
        symbol: str = Query("^DJI", description="Ticker symbol (e.g., AAPL)")
) -> PredictionResponse:
    """Predict next price log-returns from prices and optional news."""
    price_df = _process_price_df(request_body)
    price_dates = price_df["date"].dt.strftime("%Y-%m-%d").tolist()

    news_df = _process_news_df(
        request_body,
        price_dates,
        ignore_news=ignore_news,
        symbol=symbol
    )

    feature_row = _generate_features(
        price_df,
        news_df,
        request.app.state.sentiment_model,
        cfg.runtime.horizon
    )

    horizon = cfg.runtime.horizon if cfg.runtime.target_mode == "step" else cfg.runtime.horizon_list

    return _make_prediction(
        feature_row,
        request.app.state.model,
        request.app.state.preprocessor,
        request.app.state.y_scaler,
        request.app.state.y_scale,
        price_df,
        horizon,
        return_path,
        cfg.runtime.target_mode
    )
