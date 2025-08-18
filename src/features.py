import numpy as np
import pandas as pd
import ta
from ta.momentum import rsi
from ta.trend import macd_diff
from ta.volatility import bollinger_hband, bollinger_lband

from src.data import merge_price_news
from src.sentiment import FinBERT


def create_features_and_target(df: pd.DataFrame, forecast_horizon: int = 1) -> pd.DataFrame:
    """
    Create features and target.

    Technical Indicators:
    * Relative Strength Index (RSI)
    * Moving Average Convergence Divergence (MACD)
    * Bollinger Bands

    Lag Features:
    Past values used as predictors

    Moving Averages:
    These smooth out short-term fluctuations
    * Simple Moving Average (SMA)
    * Exponential Moving Average (EMA)

    Aggregations:
    Using mean, standard deviation in different time periods help the model
    to have additional information
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    if 'adj_close' not in df.columns:
        raise ValueError("Expected 'adj_close' column in df.")
    price = df['adj_close']

    df["log_return"] = np.log(price / price.shift(1))

    if forecast_horizon > 1:
        for h in range(1, forecast_horizon + 1):
            df[f'target_{h}'] = df['log_return'].shift(-h)
    else:
        df['target'] = df['log_return'].shift(-1)

    df['rsi'] = rsi(price)
    df['macd'] = macd_diff(df['log_return'])
    df['h_bollinger'] = bollinger_hband(df['log_return'])
    df['l_bollinger'] = bollinger_lband(df['log_return'])

    # Optional -> Shorter lags -> [1, 2, 3, 5, 10, 20, 25]:
    for lag in [5, 10, 25]:
        df[f'lag_{lag}'] = df['log_return'].shift(lag)
        df[f'sma_{lag}'] = ta.trend.sma_indicator(df['log_return'], lag)
        df[f'ema_{lag}'] = ta.trend.ema_indicator(df['log_return'], lag)

    df['quarter'] = df['date'].dt.quarter.astype(int)
    df["dow"] = df["date"].dt.dayofweek.astype(int)

    grp = df.groupby('quarter')['log_return']
    df['q_mean'] = grp.expanding().mean().reset_index(level=0, drop=True).shift(1)
    df['q_std'] = grp.expanding().std().reset_index(level=0, drop=True).shift(1)
    df['q_skew'] = grp.expanding().skew().reset_index(level=0, drop=True).shift(1)

    max_lag = 26
    return df.iloc[max_lag:].copy()

def generate_full_feature_row(price_df: pd.DataFrame, news_df: pd.DataFrame, sentiment_model: FinBERT | None) -> pd.DataFrame:
    if sentiment_model is None or news_df is None or news_df.empty:
        daily_sentiment = pd.DataFrame({
            "date": price_df["date"].copy(),
            "sentiment": 0.0,
            "sentiment_score": 0.0
        })
    else:
        enriched_news = sentiment_model.transform(news_df)
        daily_sentiment = sentiment_model.aggregate_daily(enriched_news)
        daily_sentiment.drop(columns=["headline_count"], inplace=True)

    merged = merge_price_news(price_df, daily_sentiment)
    features_df = create_features_and_target(merged)

    if features_df.empty:
        raise ValueError("Feature DataFrame is empty. Likely due to insufficient price history.")

    return features_df.iloc[[-1]].copy()

def convert_log_return(current_price: float, predicted_log_return: float) -> float:
    """Converts predicted log return to actual price."""
    return current_price * np.exp(predicted_log_return)
