from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np
import pandas as pd

from src.train import ModelTrainer


def recursive_forecast(
        trainer: ModelTrainer,
        X_last: pd.DataFrame,
        forecast_horizon: int = 20,
        p0: Optional[float] = None,
        past_prices: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Recursive H-step forecast in log-return units.

    Args:
        trainer: Trained ModelTrainer with a predict() method.
        X_last: Last feature row (must be a single row DataFrame).
        forecast_horizon: Number of steps ahead to forecast.
        p0: Optional initial price override (default: adj_close_l).
        past_prices: Optional array of historical prices for initializing momentum.

    Returns:
        np.ndarray of shape (H,).
    """
    if len(X_last) != 1:
        raise ValueError("X_last must be a single row.")

    X = X_last.copy(deep=True)
    idx = X.index[0]

    # Initialize current price
    price = float(p0 if p0 is not None else X.at[idx, "adj_close_l"])

    # Rolling buffer for momentum (need 11 for mom_10)
    if past_prices is not None and len(past_prices) >= 11:
        price_buf = deque(past_prices[-11:], maxlen=11)
        price = float(past_prices[-1])
    else:
        price_buf = deque([price] * 11, maxlen=11)

    preds: list[float] = []

    for _ in range(forecast_horizon):
        log_r = float(np.asarray(trainer.predict(X)).ravel()[0])
        preds.append(log_r)
        next_price = price * np.exp(log_r)

        # Update features
        _update_log_return_features(X, idx, log_r)
        _update_prices(X, idx, next_price)
        _update_momentum(X, idx, price_buf, next_price)
        _update_day_of_week(X, idx)

        price = next_price

    return np.asarray(preds, dtype=float)

def _update_log_return_features(X: pd.DataFrame, idx, log_r: float) -> None:
    """Shift lagged log return features and insert the new prediction."""
    lag_cols = [c for c in X.columns if c.startswith("lag_")]
    if not lag_cols:
        return
    for k in range(len(lag_cols), 1, -1):
        prev_col = f"lag_{k - 1}"
        if prev_col in X.columns:
            X.at[idx, f"lag_{k}"] = X.at[idx, prev_col]
    X.at[idx, "lag_1"] = log_r

def _update_prices(X: pd.DataFrame, idx, price: float) -> None:
    """Update shifted OHLC features with new price."""
    for col in ("open_l", "high_l", "low_l", "close_l", "adj_close_l"):
        if col in X.columns:
            X.at[idx, col] = price

def _update_momentum(X: pd.DataFrame, idx, price_buf: deque, new_price: float) -> None:
    """Update momentum feature using oldest price in buffer (10 steps back)."""
    if "mom_10" in X.columns and len(price_buf) == 11:
        oldest_price = price_buf[-1]
        if oldest_price > 0:
            X.at[idx, "mom_10"] = float(np.log(new_price / oldest_price))
    price_buf.appendleft(new_price)

def _update_day_of_week(X: pd.DataFrame, idx) -> None:
    """Naive day-of-week increment (mod 7). Does not skip weekends/holidays."""
    if "dow" in X.columns:
        X.at[idx, "dow"] = (int(X.at[idx, "dow"]) + 1) % 7
