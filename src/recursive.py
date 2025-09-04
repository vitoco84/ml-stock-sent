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
        past_prices: Optional[np.ndarray] = None
) -> np.ndarray:
    """Recursive H-step forecast in ΔPrice (price change) units."""
    if len(X_last) != 1:
        raise ValueError("X_last must be a single row.")

    X = X_last.copy(deep=True)
    idx = X.index[0]

    # Initialize price
    price = float(p0 if p0 is not None else X.at[idx, "adj_close_l"])

    # Buffer for momentum feature
    if past_prices is not None and len(past_prices) >= 11:
        price_buf = deque(list(past_prices[-11:]), maxlen=11)
        price = float(past_prices[-1])
    else:
        price_buf = deque([price] * 11, maxlen=11)

    preds = []

    for _ in range(forecast_horizon):
        delta = float(np.asarray(trainer.predict(X)).ravel()[0])
        preds.append(delta)
        next_price = price + delta

        # Update features
        _update_delta_features(X, idx, price, next_price)
        _update_prices(X, idx, next_price)
        _update_momentum(X, idx, price_buf, next_price)
        _update_day_of_week(X, idx)

        price = next_price

    return np.asarray(preds, dtype=float)

def _update_delta_features(X: pd.DataFrame, idx, prev_price: float, new_price: float):
    # Recompute delta features
    X.at[idx, "delta_1"] = new_price - prev_price
    # Optionally update delta_5, delta_10 if you're tracking them
    # e.g., using a rolling buffer or saving previous predictions (not shown here)

def _update_prices(X: pd.DataFrame, idx, price: float):
    for col in ("open_l", "high_l", "low_l", "close_l", "adj_close_l"):
        if col in X.columns:
            X.at[idx, col] = price

def _update_momentum(X: pd.DataFrame, idx, price_buf: deque, new_price: float):
    if "mom_10" in X.columns and len(price_buf) == 11:
        X.at[idx, "mom_10"] = float(np.log(new_price / price_buf[-1]))
    price_buf.appendleft(new_price)

def _update_day_of_week(X: pd.DataFrame, idx):
    if "dow" in X.columns:
        X.at[idx, "dow"] = (int(X.at[idx, "dow"]) + 1) % 7
