from typing import Any

import numpy as np
import pandas as pd
from fastapi import HTTPException
from pandas.tseries.offsets import BDay

from app.api.classes import (
    PredictionRequest,
    PredictionResponse,
)
from app.api.settings import get_settings
from app.api.utils import _ollama_alive, to_dict
from config.config import Config
from src.features import generate_full_feature_row
from src.llm import enrich_news_with_generated
from src.logger import get_logger


logger = get_logger(__name__)
settings = get_settings()

cfg = Config.load()

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
        ignore_news: bool,
        symbol: str
) -> pd.DataFrame:
    """Validate and normalize news data. Two modes only:
       - ignore_news=True → return empty df
       - ignore_news=False → require ≥1 headline, auto-enrich missing dates with LLM
    """
    if ignore_news:
        return pd.DataFrame(columns=["date", "headline"])

    news_payload = getattr(request_body, "news", None) or []
    if len(news_payload) > 2000:
        raise HTTPException(400, "News data exceeds 2000-row limit.")
    if not news_payload:
        raise HTTPException(422, "At least one headline is required when using news.")

    try:
        df = pd.DataFrame([to_dict(row) for row in news_payload])
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df = df.sort_values("date").groupby("date", as_index=False).head(20).tail(1000)
    except Exception:
        logger.exception("Invalid `news` payload")
        raise HTTPException(422, "`news` payload malformed. Expect list of {date, headline}.")

    real_news: list[dict[str, str]] = [
        {"date": str(pd.to_datetime(row["date"]).strftime("%Y-%m-%d")),
         "headline": str(row["headline"])}
        for row in df.to_dict(orient="records")
    ]

    ollama_base = settings.ollama_base
    if not _ollama_alive(ollama_base):
        raise HTTPException(500, "Local LLM backend not available for news enrichment.")

    enriched = enrich_news_with_generated(
        price_dates=price_dates,
        real_news=real_news,
        symbol=symbol,
        url_llm=f"{ollama_base.rstrip('/')}/api/generate",
        model_llm=settings.ollama_model,
    )
    df = pd.DataFrame(enriched)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df

def _generate_features(
        price_df: pd.DataFrame,
        news_df: pd.DataFrame,
        sentiment_model: Any,
        horizon: int
) -> pd.DataFrame:
    """Generate model-ready feature row."""
    try:
        if news_df.empty:
            return generate_full_feature_row(
                price_df,
                pd.DataFrame(),
                None,
                forecast_horizon=horizon,
                back_horizon=cfg.runtime.lag_horizon,
                fill_missing_neutral=True,
                max_embedding_dims=cfg.runtime.max_sentiment_embeddings,
                target_mode=cfg.runtime.target_mode
            )

        return generate_full_feature_row(
            price_df,
            news_df,
            sentiment_model,
            forecast_horizon=horizon,
            back_horizon=cfg.runtime.lag_horizon,
            fill_missing_neutral=False,
            max_embedding_dims=cfg.runtime.max_sentiment_embeddings,
            target_mode=cfg.runtime.target_mode
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
        return_path: bool,
        target_mode: str
) -> PredictionResponse:
    """Run model prediction and build response (direct or rolling log-returns)."""
    try:
        X = preprocessor.transform(feature_row)
        yhat = np.asarray(model.predict(X), dtype=float)
        if yhat.ndim == 1:
            yhat = yhat.reshape(1, -1)
        # Inverse-transform if y was scaled during training
        if y_scale and y_scaler is not None:
            yhat = y_scaler.inverse_transform(yhat)
    except Exception:
        logger.exception("Model prediction failed")
        raise HTTPException(500, "Prediction failed.")

    current_price = float(price_df["adj_close"].iloc[-1])
    last_date = pd.to_datetime(price_df["date"]).iloc[-1]

    if target_mode == "step":
        H = min(horizon, yhat.shape[1])
        # Horizon=1: first-step log return
        log_return = float(yhat[0, 0])
        predicted_price = current_price * float(np.exp(log_return))

        response_kwargs: dict[str, Any] = {
            "horizon": H,
            "current_price": current_price,
            "log_return": log_return,
            "predicted_price": predicted_price,
        }

        if return_path:
            logret_path = yhat[0, :H]
            predicted_price_path = current_price * np.exp(np.cumsum(logret_path))
            future_dates = pd.bdate_range(last_date + BDay(1), periods=H)
            response_kwargs.update(
                log_return_path=logret_path.tolist(),
                predicted_price_path=predicted_price_path.tolist(),
                predicted_dates=future_dates.strftime("%Y-%m-%d").tolist(),
                last_date=last_date.date(),
            )

        return PredictionResponse(**response_kwargs)

    elif target_mode == "rolling":
        # yhat shape (1,X)
        horizons = cfg.runtime.horizon_list
        horizon_preds = {h: float(yhat[0, i]) for i, h in enumerate(horizons) if i < yhat.shape[1]}

        # Default to horizon=20 for path
        H = 20
        total_return = horizon_preds.get(20, 0.0)
        # Approximate daily path as equal-split of 20-day return
        logret_path = np.full(H, total_return / H)
        predicted_price_path = current_price * np.exp(np.cumsum(logret_path))

        response_kwargs: dict[str, Any] = {
            "horizon": H,
            "current_price": current_price,
            "log_return": horizon_preds.get(1, float("nan")),
            "predicted_price": float(predicted_price_path[-1]),
            "log_return_1": horizon_preds.get(1),
            "log_return_5": horizon_preds.get(5),
            "log_return_20": horizon_preds.get(20)
        }

        if return_path:
            future_dates = pd.bdate_range(last_date + BDay(1), periods=H)
            response_kwargs.update(
                log_return_path=logret_path.tolist(),
                predicted_price_path=predicted_price_path.tolist(),
                predicted_dates=future_dates.strftime("%Y-%m-%d").tolist(),
                last_date=last_date.date(),
            )

        return PredictionResponse(**response_kwargs)

    else:
        raise HTTPException(400, f"Unknown target_mode={target_mode}")
