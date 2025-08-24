from typing import Any, Dict

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric mean absolute percentage error."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred)) + 1e-12
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom))

def _mase(y_true: np.ndarray, y_pred: np.ndarray, y_insample: np.ndarray) -> float:
    """
    Mean Absolute Scaled Error.
    y_insample: Historical target values (before forecast window) used to compute MASE.
    """
    y_insample = np.asarray(y_insample)
    if y_insample.size < 2:
        return float("nan")
    naive = np.mean(np.abs(np.diff(y_insample)))
    naive = np.maximum(naive, 1e-8)
    return float(np.mean(np.abs(y_true - y_pred)) / naive)

def metrics(y_true: np.ndarray, y_pred: np.ndarray, y_insample: np.ndarray = None) -> Dict[str, Any]:
    """Aggregate metrics. Accepts shapes (n,) or (n, H); flattens if needed."""
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
        "smape": _smape(yt, yp),
        "r2": float(r2_score(yt, yp)),
    }
    if y_insample is not None:
        out["mase"] = _mase(yt, yp, y_insample)
    return out
