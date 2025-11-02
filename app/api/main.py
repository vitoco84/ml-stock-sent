import copy
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
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
    """Initialize global settings (lazy FinBERT and models)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"App starting on device: {device}")

    app.state.device = device
    app.state.news_api_key = settings.news_api_key
    app.state.sentiment_model = None
    app.state.models = {}
    yield
    logger.info("App shutdown complete.")

def get_model(app: FastAPI, model_name: str):
    if model_name not in app.state.models:
        models_dir = Path(cfg.data.models_dir)
        model_path = models_dir / model_name
        if not model_path.exists():
            raise HTTPException(404, f"Model file not found: {model_name}")
        logger.info(f"Loading model on demand: {model_name}")
        model, preprocessor, y_scaler, y_scale = ModelTrainer.load(str(model_path))
        if hasattr(model, "to"):
            model.to(app.state.device)
        app.state.models[model_name] = {
            "model": model,
            "preprocessor": preprocessor,
            "y_scaler": y_scaler,
            "y_scale": bool(y_scale),
        }
    return app.state.models[model_name]

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

def _validate_symbol(symbol: str):
    if not re.fullmatch(r"^[A-Za-z0-9_.^-]{1,10}$", symbol.strip()):
        raise HTTPException(
            400,
            f"Invalid ticker symbol '{symbol}'. Only letters, numbers, '.', '_', '^', and '-' are allowed."
        )
    return symbol.upper()

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

        symbol = _validate_symbol(symbol)
        df = get_price_history(symbol, end_date, days)
        if df.empty:
            raise HTTPException(404, "No price data returned. Check the symbol or date range.")

        df["date"] = df["date"].dt.strftime("%Y-%m-%d")
        return {"price": df.to_dict(orient="records")}
    except Exception as e:
        logger.warning(f"Failed to fetch price history for {symbol}: {e}")
        raise HTTPException(
            400,
            f"Could not fetch price data for symbol '{symbol}'. Please check the ticker."
        )

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
async def post_predict_from_raw(
        request_body: PredictionRequest,
        request: Request,
        ignore_news: bool = Query(False, description="Ignore all news (neutral every day)"),
        return_path: bool = Query(True, description="Whether to return the full H-step path"),
        symbol: str = Query("^DJI", description="Ticker symbol (e.g.: AAPL)"),
        model_name: str = Query(None, description="Model name (e.g.: linreg.pkl)")
) -> PredictionResponse:
    """Predict next price log-returns from prices and optional news."""
    model_to_use = model_name or settings.model
    selected = get_model(request.app, model_to_use)

    price_df = _process_price_df(request_body)
    price_dates = price_df["date"].dt.strftime("%Y-%m-%d").tolist()

    if request.app.state.sentiment_model is None and not ignore_news:
        logger.info(f"Initializing FinBERT on {request.app.state.device}")
        request.app.state.sentiment_model = FinBERT(
            device=request.app.state.device,
            max_embedding_dims=cfg.runtime.max_sentiment_embeddings
        )

    symbol = _validate_symbol(symbol)
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

    response = await run_in_threadpool(
        _make_prediction,
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

    response.model_name = model_to_use
    return response

@app.get("/models")
def list_models(request: Request):
    models_dir = Path(cfg.data.models_dir)

    available_files = [
        p.name
        for p in models_dir.glob("*.pkl")
        if "wo_sent" not in p.name.lower()
    ]

    loaded = [
        name
        for name in request.app.state.models.keys()
        if "wo_sent" not in name.lower()
    ]
    models = sorted(set(available_files + loaded))
    return {"models": models}

@app.post("/fine-tune", response_model=FineTuneResponse)
async def fine_tune_linreg(
        request: Request,
        background_tasks: BackgroundTasks,
        symbol: str = Query(..., description="Ticker symbol to fine-tune on"),
        end_date: str = Query(..., description="End date (YYYY-MM-DD)"),
        days: int = Query(180, ge=80, le=1500, description="Lookback window in days")
):
    """
    Fine-tune the default Elastic Net regression model on new stock data.
    Automatically runs synchronously if quick (<10s), otherwise starts in background.
    The base model remains intact, the fine-tuned one is cached in memory.
    """
    base_name = settings.model
    symbol = _validate_symbol(symbol)
    symbol_clean = symbol.upper().replace("^", "").replace("/", "_")
    tuned_name = f"finetuned_{symbol_clean}_{base_name}"

    if base_name not in request.app.state.models:
        logger.info(f"[Fine-tune] Loading base model {base_name}")
        _ = get_model(request.app, base_name)

    try:
        df = get_price_history(symbol, end_date, days)
    except Exception as e:
        logger.warning(f"[Fine-tune] Failed to fetch data for {symbol}: {e}")
        raise HTTPException(400, f"Failed to fetch data for symbol '{symbol}'. Please check if the ticker is valid.")

    if df.empty:
        raise HTTPException(
            404,
            f"No historical price data found for '{symbol}'. Please verify the ticker symbol or try a different one."
        )

    samples = len(df) if not df.empty else 0
    if samples == 0:
        raise HTTPException(404, "No price data returned for fine-tuning.")

    start_time = time.perf_counter()
    try:
        await run_in_threadpool(
            run_fine_tune_task,
            request.app,
            symbol,
            end_date,
            days,
            base_name,
            tuned_name,
            dry_run=True
        )
    except Exception as e:
        logger.warning(f"[Fine-tune] Dry-run failed: {e}")
    duration = time.perf_counter() - start_time

    SYNC_THRESHOLD = 10.0
    if duration < SYNC_THRESHOLD:
        logger.info(f"[Fine-tune] Running synchronously for {symbol} (~{duration:.2f}s)")
        await run_in_threadpool(
            run_fine_tune_task,
            request.app,
            symbol,
            end_date,
            days,
            base_name,
            tuned_name
        )
        return FineTuneResponse(
            status="ok",
            symbol=symbol,
            cached_as=tuned_name,
            samples=samples,
            message=f"Fine-tuning complete for '{symbol}'.",
            base_model=base_name
        )

    logger.info(f"[Fine-tune] Running in background for {symbol} (est. {duration:.2f}s)")
    background_tasks.add_task(
        run_fine_tune_task,
        request.app,
        symbol,
        end_date,
        days,
        base_name,
        tuned_name
    )

    return FineTuneResponse(
        status="training",
        symbol=symbol,
        cached_as=tuned_name,
        samples=samples,
        message=f"Fine-tuning started in background for '{symbol}'.",
        base_model=base_name
    )

def run_fine_tune_task(
        app,
        symbol: str,
        end_date: str,
        days: int,
        base_name: str,
        tuned_name: str,
        dry_run: bool = False
):
    """Fine-tuning task. When dry_run=True, only prepares features to estimate runtime."""
    try:
        logger.info(f"[Fine-tune] Starting {'dry-run' if dry_run else 'training'} for {symbol}")

        df = get_price_history(symbol, end_date, days)
        feat_df = create_features_and_target(
            df,
            forecast_horizon=cfg.runtime.horizon,
            back_horizon=cfg.runtime.lag_horizon,
            training=True,
            target_mode=cfg.runtime.target_mode,
            custom_horizons=cfg.runtime.horizon_list,
        )

        if dry_run:
            # Only measure preprocessing speed, no training
            return

        target_cols = [c for c in feat_df.columns if c.startswith("target")]
        X = feat_df.drop(columns=["date"] + target_cols, errors="ignore")
        y = feat_df[target_cols]

        base_entry = app.state.models[base_name]
        base_model = base_entry["model"]
        y_scale = base_entry["y_scale"]

        fine_model = copy.deepcopy(base_model)
        fine_preprocessor, _ = get_preprocessor(X, "linreg")

        trainer = ModelTrainer(
            model=fine_model,
            name=tuned_name,
            config={"optimization_metric": "mae"},
            preprocessor=fine_preprocessor,
            y_scale=y_scale
        )

        trainer.fit(X, y)

        app.state.models[tuned_name] = {
            "model": trainer.model,
            "preprocessor": trainer.preprocessor,
            "y_scaler": trainer.y_scaler,
            "y_scale": trainer.y_scale
        }

        logger.info(f"[Fine-tune] Completed for {symbol} → {tuned_name}")
    except Exception as e:
        logger.exception(f"[Fine-tune] Failed for {symbol}: {e}")
