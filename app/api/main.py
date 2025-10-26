import copy
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware

from app.api.classes import (
    FineTuneResponse,
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
from src.features import create_features_and_target
from src.logger import get_logger
from src.preprocessing import get_preprocessor
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

    models_dir = Path(cfg.data.models_dir)
    loaded_models = {}

    for model_file in settings.available_models:
        model_path = models_dir / model_file
        if not model_path.exists():
            logger.warning(f"Skipping missing model: {model_file}")
            continue
        try:
            model, preprocessor, y_scaler, y_scale = ModelTrainer.load(str(model_path))
            loaded_models[model_file] = {
                "model": model,
                "preprocessor": preprocessor,
                "y_scaler": y_scaler,
                "y_scale": bool(y_scale),
            }
            logger.info(f"Loaded model: {model_file}")
        except Exception as e:
            logger.error(f"Failed to load {model_file}: {e}")

    if not loaded_models:
        raise RuntimeError("No valid models could be loaded. Check settings.available_models.")

    app.state.news_api_key = settings.news_api_key
    app.state.sentiment_model = sentiment_model
    app.state.models = loaded_models

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
    except Exception as e:
        logger.exception("fetch_price_history failed")
        raise HTTPException(500, f"Failed to fetch price data: {e}")

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
    except Exception as e:
        logger.exception("fetch_news_history failed")
        raise HTTPException(500, f"Failed to fetch news data: {e}")

@app.post("/predict-raw", response_model=PredictionResponse, response_model_exclude_none=True)
def post_predict_from_raw(
        request_body: PredictionRequest,
        request: Request,
        ignore_news: bool = Query(False, description="Ignore all news (neutral every day)"),
        return_path: bool = Query(True, description="Whether to return the full H-step path"),
        symbol: str = Query("^DJI", description="Ticker symbol (e.g.: AAPL)"),
        model_name: str = Query(None, description="Model name (e.g.: linreg.pkl)")
) -> PredictionResponse:
    """Predict next price log-returns from prices and optional news."""
    model_to_use = model_name or settings.model
    if model_to_use not in request.app.state.models:
        raise HTTPException(
            400,
            f"Invalid model '{model_to_use}'. Allowed models: {list(request.app.state.models.keys())}"
        )

    selected = request.app.state.models[model_to_use]

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
        selected["model"],
        selected["preprocessor"],
        selected["y_scaler"],
        selected["y_scale"],
        price_df,
        horizon,
        return_path,
        cfg.runtime.target_mode
    )

@app.get("/models")
def list_models(request: Request):
    return {"models": list(request.app.state.models.keys())}

@app.post("/fine-tune", response_model=FineTuneResponse)
def fine_tune_linreg(
        request: Request,
        symbol: str = Query(..., description="Ticker symbol to fine-tune on"),
        end_date: str = Query(..., description="End date (YYYY-MM-DD)"),
        days: int = Query(180, ge=80, le=365, description="Lookback window in trading days")
):
    """
    Fine-tune the default linear regression model on new stock data.
    The base model remains intact, the fine-tuned one is cached in memory.
    """

    base_name = settings.model
    tuned_name = f"finetuned_{base_name}"

    if base_name not in request.app.state.models:
        raise HTTPException(400, f"Base model '{base_name}' not loaded in memory.")

    entry = request.app.state.models[base_name]
    model, _, _, y_scale = (
        entry["model"],
        entry["preprocessor"],
        entry["y_scaler"],
        entry["y_scale"]
    )

    try:
        df = get_price_history(symbol, end_date, days)
        feat_df = create_features_and_target(
            df,
            forecast_horizon=cfg.runtime.horizon,
            back_horizon=cfg.runtime.lag_horizon,
            training=True,
            target_mode=cfg.runtime.target_mode,
            custom_horizons=cfg.runtime.horizon_list
        )

        target_cols = [c for c in feat_df.columns if c.startswith("target")]

        X = feat_df.drop(columns=["date"] + target_cols, errors="ignore")
        y = feat_df[target_cols]

        fine_model = copy.deepcopy(model)
        fine_preprocessor, _ = get_preprocessor(X, "linreg")

        trainer = ModelTrainer(
            model=fine_model,
            name=tuned_name,
            config={"optimization_metric": "mae"},
            preprocessor=fine_preprocessor,
            y_scale=y_scale
        )

        trainer.fit(X, y)

        request.app.state.models[tuned_name] = {
            "model": trainer.model,
            "preprocessor": trainer.preprocessor,
            "y_scaler": trainer.y_scaler,
            "y_scale": trainer.y_scale
        }

        logger.info(f"Fine-tuned '{base_name}' on {symbol}, cached as '{tuned_name}'")

        return FineTuneResponse(
            status="ok",
            symbol=symbol,
            cached_as=tuned_name,
            samples=len(X),
            start_date=df["date"].min().date(),
            end_date=df["date"].max().date(),
            message=f"Fine-tuned model cached as '{tuned_name}'."
        )
    except Exception as e:
        logger.exception("Fine-tune failed")
        raise HTTPException(500, f"Fine-tune failed: {e}")
