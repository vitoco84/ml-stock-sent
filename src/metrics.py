from typing import Any, Dict

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric mean absolute error."""
    denom = (np.abs(y_true) + np.abs(y_pred)) + 1e-12
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom))

def _mase(y_true: np.ndarray, y_pred: np.ndarray, y_insample: np.ndarray) -> float:
    """Mean Absolute Scaled Error.
    y_insample: Historical target values (before forecast window) used to compute MASE.
    """
    naive = np.mean(np.abs(np.diff(y_insample)))
    naive = np.maximum(naive, 1e-8)
    return float(np.mean(np.abs(y_true - y_pred)) / naive)

def metrics(y_true: np.ndarray, y_pred: np.ndarray, y_insample: np.ndarray = None) -> Dict[str, Any]:
    """Aggregate Metrics."""
    mse = mean_squared_error(y_true, y_pred)
    result = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "smape": _smape(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred))
    }
    if y_insample is not None:
        result["mase"] = _mase(y_true, y_pred, y_insample)
    return result
