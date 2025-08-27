from typing import Optional

import numpy as np
import pandas as pd

from src.data import merge_price_news
from src.sentiment import FinBERT


def create_features_and_target(
        df: pd.DataFrame,
        forecast_horizon: int = 1,
        back_horizon: int = 7,
) -> pd.DataFrame:
    """
    Features:
      - Sliding lagged log returns: lag_1 ... lag_{n_lags}
      - Calendar: day-of-week
    Targets:
      - Multi-step log return targets (target_1 ... target_H), or 'target' for 1-step
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    if "adj_close" not in df.columns:
        raise ValueError("Expected 'adj_close' column in df.")

    price = df["adj_close"].astype(float)
    df["log_return"] = np.log(price / price.shift(1))

    # Targets
    if forecast_horizon > 1:
        for h in range(1, forecast_horizon + 1):
            df[f"target_{h}"] = df["log_return"].shift(-h)
    else:
        df["target"] = df["log_return"].shift(-1)

    # Lags of log-returns
    for k in range(1, back_horizon + 1):
        df[f"lag_{k}"] = df["log_return"].shift(k)

    # Optional (Volatility, Moving Averages and Distance, Momentum Returns, Month)
    # df["vol_5"]  = df["log_return"].rolling(5).std().shift(1)
    # df["sma_5"] = df["adj_close"].rolling(5).mean().shift(1)
    # df["dist_sma_5"] = (df["adj_close"] / df["sma_5"]).shift(1)
    # df["ret_5"] = df["adj_close"].pct_change(5).shift(1)
    # df["month"] = df["date"].dt.month.astype(int)

    df["dow"] = df["date"].dt.dayofweek.astype(int)

    return df.iloc[back_horizon:].copy()

def _neutral_sentiment(max_embedding_dims, price_dates_norm):
    daily_sentiment = pd.DataFrame({"date": price_dates_norm})
    daily_sentiment["pos"] = 0.0
    daily_sentiment["neg"] = 0.0
    daily_sentiment["neu"] = 0.0
    daily_sentiment["pos_minus_neg"] = 0.0
    for i in range(max_embedding_dims):
        daily_sentiment[f"emb_{i}"] = 0.0
    return daily_sentiment

def _ensure_embeddings(daily_sentiment, max_embedding_dims):
    for i in range(max_embedding_dims):
        col = f"emb_{i}"
        if col not in daily_sentiment.columns:
            daily_sentiment[col] = 0.0

def _fill_missing_neutral(daily_sentiment, fill_missing_neutral, max_embedding_dims, price_dates_norm):
    if fill_missing_neutral:
        ds = daily_sentiment.copy()
        ds["date"] = pd.to_datetime(ds["date"]).dt.normalize()

        emb_cols = [c for c in ds.columns if c.startswith("emb_")]
        if not emb_cols and max_embedding_dims:
            emb_cols = [f"emb_{i}" for i in range(max_embedding_dims)]
            for c in emb_cols:
                ds[c] = 0.0

        for c in ["pos", "neg", "neu", "pos_minus_neg", *emb_cols]:
            if c not in ds.columns:
                ds[c] = 0.0

        ds = (
            ds.set_index("date")
            .reindex(price_dates_norm.unique())
            .reset_index()
            .fillna(0.0)
        )
        daily_sentiment = ds
    return daily_sentiment

def _drop_target_columns(features_df, forecast_horizon):
    # Drop Targets, keep only last feature row for inference
    target_cols = [f"target_{i}" for i in range(1, forecast_horizon + 1)]
    if "target" in features_df.columns:
        target_cols.append("target")
    features_df = features_df.drop(columns=[c for c in target_cols if c in features_df.columns], errors="ignore")
    return features_df

def generate_full_feature_row(
        price_df: pd.DataFrame,
        news_df: Optional[pd.DataFrame],
        sentiment_model: Optional[FinBERT],
        *,
        forecast_horizon: int = 30,
        back_horizon: int = 7,
        max_embedding_dims: int = 17,
        fill_missing_neutral: bool = True
) -> pd.DataFrame:
    """Generate a full feature row."""
    price_dates_norm = pd.to_datetime(price_df["date"]).dt.normalize()

    if sentiment_model is None or news_df is None or news_df.empty:
        daily_sentiment = _neutral_sentiment(max_embedding_dims, price_dates_norm)
    else:
        enriched_news = sentiment_model.transform(news_df)
        daily_sentiment = sentiment_model.aggregate_daily(enriched_news)
        daily_sentiment.drop(columns=["headline_count"], inplace=True, errors="ignore")

        daily_sentiment = _fill_missing_neutral(
            daily_sentiment,
            fill_missing_neutral,
            max_embedding_dims,
            price_dates_norm
        )

        _ensure_embeddings(daily_sentiment, max_embedding_dims)

    merged = merge_price_news(price_df, daily_sentiment)
    features_df = create_features_and_target(merged, forecast_horizon, back_horizon)

    if features_df.empty:
        raise ValueError("Feature DataFrame is empty. Likely due to insufficient price history.")

    features_df = _drop_target_columns(features_df, forecast_horizon)
    return features_df.tail(1).copy()
