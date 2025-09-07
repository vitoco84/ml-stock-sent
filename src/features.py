from typing import Optional

import numpy as np
import pandas as pd

from src.data import merge_price_news
from src.sentiment import FinBERT


def create_features_and_target(
        df: pd.DataFrame,
        forecast_horizon: int = 1,
        back_horizon: int = 7,
        training: bool = False
) -> pd.DataFrame:
    """
    Create features and targets from price + sentiment data.

    Features:
      - Price deltas (1, 5, 10 days)
      - OHLC shifted features
      - Calendar (day of week)
      - Lagged log returns
      - Rolling momentum

    Targets:
      - If horizon=1: 'target'
      - If horizon>1: 'target_1' ... 'target_H'
        defined as ΔAdjClose(t+h) = AdjClose_{t+h} − AdjClose_t
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    if "adj_close" not in df.columns:
        raise ValueError("Expected 'adj_close' column in df.")

    # Targets
    if forecast_horizon > 1:
        for h in range(1, forecast_horizon + 1):
            df[f"target_{h}"] = df["adj_close"].shift(-h) - df["adj_close"]
    else:
        df["target"] = df["adj_close"].shift(-1) - df["adj_close"]

    # Lags of log-returns
    df["log_return"] = np.log(df["adj_close"] / df["adj_close"].shift(1))
    for k in range(1, back_horizon + 1):
        df[f"lag_{k}"] = df["log_return"].shift(k)

    # Delta Price features
    df["delta_1"] = df["adj_close"].diff(1)
    df["delta_5"] = df["adj_close"].diff(5)
    df["delta_10"] = df["adj_close"].diff(10)

    # Rolling and Momentum
    df["mom_10"] = np.log(df["adj_close"] / df["adj_close"].shift(10))

    # OHLC Shifted
    for c in ["open", "high", "low", "close", "adj_close", "volume"]:
        if c in df.columns:
            val = np.log1p(df[c]) if c == "volume" else df[c]
            df[f"{c}_l"] = val.shift(1)

    # Calendar
    df["dow"] = df["date"].dt.dayofweek.astype(int)

    #  Drop incomplete rows (training), keep last for Inference
    if training:
        return df.iloc[max(back_horizon, 10): len(df) - forecast_horizon].copy()
    return df.iloc[back_horizon:].copy()

def _neutral_sentiment(max_embedding_dims: int, price_dates_norm: pd.Series) -> pd.DataFrame:
    """Return neutral (zeroed) sentiment embeddings aligned with dates."""
    daily_sentiment = pd.DataFrame({"date": price_dates_norm})
    base_cols = {"pos": 0.0, "neg": 0.0, "neu": 0.0, "pos_minus_neg": 0.0}
    for k, v in base_cols.items():
        daily_sentiment[k] = v
    for i in range(max_embedding_dims):
        daily_sentiment[f"emb_{i}"] = 0.0
    return daily_sentiment

def _ensure_embeddings(daily_sentiment: pd.DataFrame, max_embedding_dims: int) -> None:
    """Ensure sentiment DataFrame has embedding columns up to max_embedding_dims."""
    for i in range(max_embedding_dims):
        col = f"emb_{i}"
        if col not in daily_sentiment.columns:
            daily_sentiment[col] = 0.0

def _fill_missing_neutral(
        daily_sentiment: pd.DataFrame,
        fill_missing_neutral: bool,
        max_embedding_dims: int,
        price_dates_norm: pd.Series,
) -> pd.DataFrame:
    """Fill missing dates and sentiment values with neutral placeholders."""
    if not fill_missing_neutral:
        return daily_sentiment

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
    return ds

def _drop_target_columns(features_df: pd.DataFrame, forecast_horizon: int) -> pd.DataFrame:
    """Remove target columns from features (for inference)."""
    target_cols = [f"target_{i}" for i in range(1, forecast_horizon + 1)]
    if "target" in features_df.columns:
        target_cols.append("target")
    return features_df.drop(columns=[c for c in target_cols if c in features_df.columns], errors="ignore")

def _generate_feat_target(fill_missing_neutral, max_embedding_dims, news_df, price_df, sentiment_model):
    price_dates_norm = pd.to_datetime(price_df["date"]).dt.normalize()
    if sentiment_model is None or news_df is None or news_df.empty:
        daily_sentiment = _neutral_sentiment(max_embedding_dims, price_dates_norm)
    else:
        enriched_news = sentiment_model.transform(news_df)
        daily_sentiment = sentiment_model.aggregate_daily(enriched_news)
        daily_sentiment.drop(columns=["headline_count"], inplace=True, errors="ignore")

        daily_sentiment = _fill_missing_neutral(
            daily_sentiment, fill_missing_neutral, max_embedding_dims, price_dates_norm
        )
        _ensure_embeddings(daily_sentiment, max_embedding_dims)
    merged = merge_price_news(price_df, daily_sentiment)
    return merged

def generate_full_feature_row(
        price_df: pd.DataFrame,
        news_df: Optional[pd.DataFrame],
        sentiment_model: Optional[FinBERT],
        *,
        forecast_horizon: int = 30,
        back_horizon: int = 7,
        max_embedding_dims: int = 17,
        fill_missing_neutral: bool = True,
) -> pd.DataFrame:
    """
    Generate the latest full feature row for inference.

    Combines price history, engineered features, and optional FinBERT sentiment.
    """
    merged = _generate_feat_target(fill_missing_neutral, max_embedding_dims, news_df, price_df, sentiment_model)
    features_df = create_features_and_target(merged, forecast_horizon, back_horizon)

    if features_df.empty:
        raise ValueError("Feature DataFrame is empty. Likely due to insufficient price history.")

    features_df = _drop_target_columns(features_df, forecast_horizon)
    return features_df.tail(1).copy()

def generate_training_data(
        price_df: pd.DataFrame,
        news_df: Optional[pd.DataFrame],
        sentiment_model: Optional[FinBERT],
        *,
        forecast_horizon: int = 30,
        back_horizon: int = 7,
        max_embedding_dims: int = 17,
        fill_missing_neutral: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate features + targets for model fine-tuning on a new stock.
    """
    merged = _generate_feat_target(fill_missing_neutral, max_embedding_dims, news_df, price_df, sentiment_model)
    features_df = create_features_and_target(
        merged, forecast_horizon=forecast_horizon, back_horizon=back_horizon, training=True
    )

    target_cols = [c for c in features_df.columns if c.startswith("target")]
    if not target_cols:
        raise ValueError("No target columns found for fine-tuning.")

    X = features_df.drop(columns=target_cols + ["date"], errors="ignore")
    y = features_df[target_cols]

    return X, y
