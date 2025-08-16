import numpy as np
import pandas as pd
import ta
from ta.momentum import rsi
from ta.trend import macd_diff
from ta.volatility import bollinger_hband, bollinger_lband


def create_features_and_target(df: pd.DataFrame, forecast_horizon: int = 30) -> pd.DataFrame:
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

    df["log_return"] = np.log(df["adj_close"] / df["adj_close"].shift(1))
    df['target'] = df['log_return'].shift(-forecast_horizon)

    df['rsi'] = rsi(df['log_return'])
    df['macd'] = macd_diff(df['log_return'])
    df['h_bollinger'] = bollinger_hband(df['log_return'])
    df['l_bollinger'] = bollinger_lband(df['log_return'])

    for lag in [5, 10, 25, 50]:
        df[f'lag_{lag}'] = df['log_return'].shift(lag)
        df[f'sma_{lag}'] = ta.trend.sma_indicator(df['log_return'], lag)
        df[f'ema_{lag}'] = ta.trend.ema_indicator(df['log_return'], lag)

    df['quarter'] = df['date'].dt.quarter.astype(int)
    df["dow"] = df["date"].dt.dayofweek.astype(int)

    df['q_mean'] = df['quarter'].map(df.groupby('quarter')['log_return'].mean())
    df['q_std'] = df['quarter'].map(df.groupby('quarter')['log_return'].std())
    df['q_skew'] = df['quarter'].map(df.groupby('quarter')['log_return'].skew())

    max_lag = 51
    return df.iloc[max_lag:].copy()
