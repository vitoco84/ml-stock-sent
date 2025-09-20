from typing import Any

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Directional accuracy, Ignores zero values in either series.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = (y_true != 0) & (y_pred != 0)
    if np.count_nonzero(mask) == 0:
        return float("nan")

    correct = np.sign(y_true[mask]) == np.sign(y_pred[mask])
    return float(np.mean(correct))

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
    Aggregate and per-horizon metrics for multi-step forecasting.

    - If y_true/y_pred are 1D: compute standard metrics.
    - If 2D (n_samples, horizon): compute both
        (a) aggregate metrics (flattened),
        (b) per-horizon metrics.
    """
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)

    if yt.shape != yp.shape:
        raise ValueError(f"Shape mismatch: y_true {yt.shape} vs y_pred {yp.shape}")

    out: dict[str, Any] = {}

    yt_flat = yt.ravel()
    yp_flat = yp.ravel()
    mse = mean_squared_error(yt_flat, yp_flat)
    out["aggregate"] = {
        "mae": float(mean_absolute_error(yt_flat, yp_flat)),
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(yt_flat, yp_flat)),
        "directional_accuracy": _directional_accuracy(yt_flat, yp_flat)
    }
    if y_insample is not None:
        out["aggregate"]["mase"] = _mase(yt_flat, yp_flat, y_insample)

    if yt.ndim == 2 and yt.shape[1] > 1:
        per_h = {}
        for h in range(yt.shape[1]):
            mse_h = mean_squared_error(yt[:, h], yp[:, h])
            per_h[h + 1] = {
                "mae": float(mean_absolute_error(yt[:, h], yp[:, h])),
                "mse": float(mse_h),
                "rmse": float(np.sqrt(mse_h)),
                "r2": float(r2_score(yt[:, h], yp[:, h])),
                "directional_accuracy": _directional_accuracy(yt[:, h], yp[:, h]),
            }
            if y_insample is not None:
                per_h[h + 1]["mase"] = _mase(yt[:, h], yp[:, h], y_insample)
        out["per_horizon"] = per_h

    return out
