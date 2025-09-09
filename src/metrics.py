from typing import Any

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Directional accuracy: fraction of times predicted and actual returns have the same sign.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)))

def _mase(y_true: np.ndarray, y_pred: np.ndarray, y_insample: np.ndarray) -> float:
    """
    Mean Absolute Scaled Error (MASE).
    y_insample: Historical target values (before forecast window).
    """
    y_insample = np.asarray(y_insample)
    if y_insample.size < 2:
        return float("nan")
    naive = np.mean(np.abs(np.diff(y_insample)))
    naive = np.maximum(naive, 1e-8)
    return float(np.mean(np.abs(y_true - y_pred)) / naive)

def metrics(y_true: np.ndarray, y_pred: np.ndarray, y_insample: np.ndarray = None) -> dict[str, Any]:
    """
    Aggregate regression/forecasting metrics.
    Flattens arrays if multi-output (n, H).

    Returns:
        Dict of metrics: MAE, MSE, RMSE, SMAPE, R², [MASE if y_insample].
    """
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    if yt.shape != yp.shape:
        raise ValueError(f"Shape mismatch: y_true {yt.shape} vs y_pred {yp.shape}")
    if yt.ndim == 2:
        yt = yt.ravel()
        yp = yp.ravel()

    mse = mean_squared_error(yt, yp)
    out = {
        "mae": float(mean_absolute_error(yt, yp)),
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(yt, yp)),
        "directional_accuracy": _directional_accuracy(yt, yp)
    }
    if y_insample is not None:
        out["mase"] = _mase(yt, yp, y_insample)
    return out
