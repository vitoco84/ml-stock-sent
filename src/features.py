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
      - Calendar: quarter, day-of-week
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

    if forecast_horizon > 1:
        for h in range(1, forecast_horizon + 1):
            df[f"target_{h}"] = df["log_return"].shift(-h)
    else:
        df["target"] = df["log_return"].shift(-1)

    for k in range(1, back_horizon + 1):
        df[f"lag_{k}"] = df["log_return"].shift(k)

    df["quarter"] = df["date"].dt.quarter.astype(int)
    df["dow"] = df["date"].dt.dayofweek.astype(int)

    return df.iloc[back_horizon:].copy()

def generate_full_feature_row(
        price_df: pd.DataFrame,
        news_df: Optional[pd.DataFrame],
        sentiment_model: Optional[FinBERT],
        forecast_horizon: int = 30,
        back_horizon: int = 7,
        max_embedding_dims: int = 17
) -> pd.DataFrame:
    """Generate a full feature row."""
    if sentiment_model is None or news_df is None or news_df.empty:
        daily_sentiment = pd.DataFrame({"date": price_df["date"].copy()})
        daily_sentiment["pos"] = 0.0
        daily_sentiment["neg"] = 0.0
        daily_sentiment["neu"] = 0.0
        daily_sentiment["pos_minus_neg"] = 0.0
        for i in range(max_embedding_dims):
            daily_sentiment[f"emb_{i}"] = 0.0
    else:
        enriched_news = sentiment_model.transform(news_df)
        daily_sentiment = sentiment_model.aggregate_daily(enriched_news)
        daily_sentiment.drop(columns=["headline_count"], inplace=True)

    merged = merge_price_news(price_df, daily_sentiment)
    features_df = create_features_and_target(merged, forecast_horizon, back_horizon)

    if features_df.empty:
        raise ValueError("Feature DataFrame is empty. Likely due to insufficient price history.")

    # Drop Targets, keep only last feature row for inference
    target_cols = [f"target_{i}" for i in range(1, forecast_horizon + 1)]
    if "target" in features_df.columns:
        target_cols.append("target")
    features_df = features_df.drop(columns=[c for c in target_cols if c in features_df.columns])

    return features_df.iloc[[-1]].copy()
