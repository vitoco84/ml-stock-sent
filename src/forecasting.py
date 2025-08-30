from collections import deque

import numpy as np
import pandas as pd

from src.train import ModelTrainer


def recursive_forecast(
        trainer: ModelTrainer,
        X_last: pd.DataFrame,
        forecast_horizon: int = 30,
        p0: float | None = None
):
    """Recursive H-step forecast in log-returns."""
    if len(X_last) != 1:
        raise ValueError("X_last must be one row.")

    X = X_last.copy(deep=True)
    idx = X.index[0]
    price = float(p0 if p0 is not None else X.at[idx, "adj_close_l"])
    lag_cols = sorted([c for c in X.columns if c.startswith("lag_")], key=lambda s: int(s.split("_")[1]))
    price_buf = deque([price], maxlen=11)
    preds = []

    for _ in range(forecast_horizon):
        lr = float(np.asarray(trainer.predict(X)).ravel()[0])
        preds.append(lr)
        next_price = price * np.exp(lr)

        _update_lags(X, idx, lag_cols, lr)
        _update_returns(X, idx, lag_cols)
        _update_prices(X, idx, next_price)
        _update_momentum(X, idx, price_buf, next_price)
        _update_day_of_week(X, idx)

        price = next_price

    return np.asarray(preds, dtype=float)

def _update_lags(X: pd.DataFrame, idx, lag_cols: list[str], lr: float):
    if len(lag_cols) > 1:
        X.loc[:, lag_cols[1:]] = X.loc[:, lag_cols[:-1]].to_numpy()
    if "lag_1" in X.columns:
        X.at[idx, "lag_1"] = lr

def _update_returns(X: pd.DataFrame, idx, lag_cols: list[str]):
    first5 = lag_cols[:5]
    if len(first5) == 5 and "ret_mean_5" in X.columns:
        vals = X.loc[idx, first5].astype(float).to_numpy()
        X.at[idx, "ret_mean_5"] = vals.mean()
        if "ret_std_5" in X.columns:
            X.at[idx, "ret_std_5"] = vals.std(ddof=1)

def _update_prices(X: pd.DataFrame, idx, price: float):
    for col in ("open_l", "high_l", "low_l", "close_l", "adj_close_l"):
        if col in X.columns:
            X.at[idx, col] = price

def _update_momentum(X: pd.DataFrame, idx, price_buf: deque, new_price: float):
    if "mom_10" in X.columns and len(price_buf) == 11:
        X.at[idx, "mom_10"] = float(np.log(price_buf[0] / price_buf[-1]))
    price_buf.appendleft(new_price)

def _update_day_of_week(X: pd.DataFrame, idx):
    if "dow" in X.columns:
        X.at[idx, "dow"] = (int(X.at[idx, "dow"]) + 1) % 7
