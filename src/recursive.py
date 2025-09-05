from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np
import pandas as pd

from src.train import ModelTrainer


def recursive_forecast(
        trainer: ModelTrainer,
        X_last: pd.DataFrame,
        forecast_horizon: int = 30,
        p0: Optional[float] = None,
        past_prices: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Recursive H-step forecast in ΔPrice units.

    Args:
        trainer: Trained ModelTrainer with a predict() method.
        X_last: Last feature row (must be a single row DataFrame).
        forecast_horizon: Number of steps ahead to forecast.
        p0: Optional initial price override (default: adj_close_l).
        past_prices: Optional array of historical prices for initializing momentum.

    Returns:
        np.ndarray of shape (H,) with forecasted deltas.
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
        delta = float(np.asarray(trainer.predict(X)).ravel()[0])
        preds.append(delta)
        next_price = price + delta

        # Update features
        _update_delta_features(X, idx, price, next_price, price_buf)
        _update_prices(X, idx, next_price)
        _update_momentum(X, idx, price_buf, next_price)
        _update_day_of_week(X, idx)

        price = next_price

    return np.asarray(preds, dtype=float)

def _update_delta_features(
        X: pd.DataFrame, idx, prev_price: float, new_price: float, price_buf: deque
) -> None:
    """Update delta features based on new price."""
    X.at[idx, "delta_1"] = new_price - prev_price
    if "delta_5" in X.columns and len(price_buf) >= 6:
        X.at[idx, "delta_5"] = new_price - price_buf[-5]
    if "delta_10" in X.columns and len(price_buf) >= 11:
        X.at[idx, "delta_10"] = new_price - price_buf[-10]

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
