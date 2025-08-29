from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf


def _ensure_path(path: Path | str) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

def _dates_np(s: pd.Series) -> np.ndarray:
    dt = pd.to_datetime(s, utc=True, errors="coerce").dt.tz_convert(None)
    return dt.to_numpy(dtype="datetime64[ns]")

def _extract_lr(y_pred) -> np.ndarray:
    if y_pred is None:
        return np.array([], dtype=float)
    arr = np.asarray(y_pred)
    if arr.size == 0:
        return np.array([], dtype=float)
    if arr.ndim == 2:
        arr = arr[:, 0]
    else:
        arr = arr.ravel()
    arr = pd.to_numeric(pd.Series(arr), errors="coerce").to_numpy()
    arr = arr[np.isfinite(arr)]
    return arr

def _align_price_path(base_today: np.ndarray, logret_t1: np.ndarray) -> np.ndarray:
    n = min(len(base_today), len(logret_t1))
    if n <= 0:
        return np.array([], dtype=float)
    return base_today[:n] * np.exp(logret_t1[:n])

def _next_step_arrays(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    dates = _dates_np(df["date"])
    adj = pd.to_numeric(df["adj_close"], errors="coerce").to_numpy(dtype=float)
    return dates[1:], adj[1:], adj[:-1]

def plot_val_test_overlay(df_val: pd.DataFrame, df_test: pd.DataFrame, results: List[Dict], path: Path):
    if df_val.empty or df_test.empty:
        raise ValueError("val or test set is empty.")
    df_all = pd.concat([df_val, df_test], ignore_index=True)
    dates_all, actual_all, base_today = _next_step_arrays(df_all)
    split_date = _dates_np(df_val["date"])[-1]

    plt.figure(figsize=(12, 5))
    plt.plot(dates_all, actual_all, "--", label="Actual (t+1)", linewidth=2)

    for res in results:
        lr_val = _extract_lr(res.get("y_pred_val"))
        lr_test = _extract_lr(res.get("y_pred_test"))
        if lr_val.size == 0 and lr_test.size == 0:
            continue
        lr_all = np.concatenate([lr_val, lr_test]) if lr_test.size else lr_val
        yhat = _align_price_path(base_today, lr_all)
        n = min(len(dates_all), len(yhat))
        if n > 0:
            plt.plot(dates_all[:n], yhat[:n], label=f"{res.get('kind', 'model')} (val+test)", linewidth=2)

    plt.axvline(split_date, color="k", linestyle=":", alpha=0.7, label="Val/Test split")
    plt.title("Actual vs Predicted Adj Close (Val + Test, H=1)")
    plt.xlabel("Date")
    plt.ylabel("Adj Close")
    plt.grid(True, alpha=0.25)
    plt.legend()
    _ensure_path(path)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.show()
    plt.close()

def plot_val_overlay(df_val: pd.DataFrame, results: List[Dict], path: Path):
    if df_val.empty:
        raise ValueError("df_val is empty.")
    dates_next, actual_next, base_today = _next_step_arrays(df_val)

    plt.figure(figsize=(12, 5))
    plt.plot(dates_next, actual_next, "--", label="Actual (t+1)", linewidth=2)

    for res in results:
        lr1 = _extract_lr(res.get("y_pred_val"))
        yhat_next = _align_price_path(base_today, lr1)
        n = min(len(dates_next), len(yhat_next))
        if n > 0:
            plt.plot(dates_next[:n], yhat_next[:n], label=f"{res.get('kind', 'model')} (t+1)", linewidth=2)

    plt.title("Actual vs Predicted Adj Close (Validation, H=1)")
    plt.xlabel("Date")
    plt.ylabel("Adj Close")
    plt.grid(True, alpha=0.25)
    plt.legend()
    _ensure_path(path)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.show()
    plt.close()

def plot_test_overlay(df_test: pd.DataFrame, results: List[Dict], path: Path):
    if df_test.empty:
        raise ValueError("df_test is empty.")
    dates_next, actual_next, base_today = _next_step_arrays(df_test)

    plt.figure(figsize=(12, 5))
    plt.plot(dates_next, actual_next, "--", label="Actual (t+1)", linewidth=2)

    for res in results:
        lr1 = _extract_lr(res.get("y_pred_test"))
        yhat = _align_price_path(base_today, lr1)
        n = min(len(dates_next), len(yhat))
        if n > 0:
            plt.plot(dates_next[:n], yhat[:n], label=f"{res.get('kind', 'model')} (t+1)", linewidth=2)

    plt.title("Actual vs Predicted Adj Close (Test Set, H=1)")
    plt.xlabel("Date")
    plt.ylabel("Adj Close")
    plt.grid(True, alpha=0.25)
    plt.legend()
    _ensure_path(path)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.show()
    plt.close()

def plot_forecast_overlay(df_test: pd.DataFrame, df_forecast: pd.DataFrame, results: List[Dict], path: Path):
    if df_test.empty or df_forecast.empty:
        raise ValueError("df_test or df_forecast is empty.")
    H = int(results[0]["horizon"])

    hist_dates = _dates_np(df_test["date"])
    hist_prices = pd.to_numeric(df_test["adj_close"], errors="coerce").to_numpy(dtype=float)
    anchor_date = hist_dates[-1]
    p0 = float(hist_prices[-1])

    fut_dates = _dates_np(df_forecast["date"])
    actual_path = pd.to_numeric(df_forecast["adj_close"], errors="coerce").to_numpy(dtype=float)

    plt.figure(figsize=(12, 5))
    plt.plot(hist_dates, hist_prices, label="History (adj_close)", alpha=0.9)
    plt.axvline(anchor_date, linestyle=":", alpha=0.7)

    if len(fut_dates) and len(actual_path):
        plt.plot(fut_dates, actual_path, label=f"Actual next {len(actual_path)}d", linewidth=2)

    for res in results:
        lr_path = _extract_lr(res.get("y_pred_last"))[:H]
        if lr_path.size == 0:
            continue
        forecast_prices = p0 * np.exp(np.cumsum(lr_path))
        n = min(len(fut_dates), len(forecast_prices))
        if n > 0:
            plt.plot(fut_dates[:n], forecast_prices[:n], "--", linewidth=2,
                     label=f"{res.get('kind', 'model')} forecast")

    ad = pd.to_datetime(anchor_date).date()
    plt.title(f"Forecast vs Actuals from {ad} (H={H}d)")
    plt.xlabel("Date")
    plt.ylabel("Adj Close")
    plt.grid(True, alpha=0.25)
    plt.legend()
    _ensure_path(path)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.show()
    plt.close()

def plot_forecast_diagnostics(df_forecast: pd.DataFrame, df_test: pd.DataFrame, results: List[Dict], path: Path):
    if df_test.empty or df_forecast.empty:
        raise ValueError("df_test or df_forecast is empty.")
    H = int(results[0]["horizon"])

    fut_dates = _dates_np(df_forecast["date"])
    actual_path = pd.to_numeric(df_forecast["adj_close"], errors="coerce").to_numpy(dtype=float)
    p0 = pd.to_numeric(df_test["adj_close"], errors="coerce").to_numpy(dtype=float)[-1]

    preds = {res["kind"]: p0 * np.exp(np.cumsum(np.asarray(res["y_pred_last"]).ravel()[:H])) for res in results}

    fig, (ax_res, ax_clr, ax_pr) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    if len(fut_dates) and len(actual_path):
        for kind, yhat in preds.items():
            n = min(len(fut_dates), len(actual_path), len(yhat))
            if n > 0:
                ax_res.plot(fut_dates[:n], actual_path[:n] - yhat[:n], label=kind)
        ax_res.axhline(0, color="k", linewidth=0.8, alpha=0.5)
        ax_res.set_title("Residuals over Horizon (Actual − Forecast)")
        ax_res.set_ylabel("Residual (Price)")
        ax_res.grid(True, alpha=0.25)
        ax_res.legend()
    else:
        ax_res.text(0.02, 0.5, "No actuals in forecast window", transform=ax_res.transAxes)
        ax_res.set_axis_off()

    if len(actual_path):
        act_cum = np.cumsum(np.log(actual_path / actual_path[0]))
        ax_clr.plot(fut_dates[:len(act_cum)], act_cum, label="Actual")
    for kind, yhat in preds.items():
        pred_cum = np.cumsum(np.log(yhat / yhat[0]))
        ax_clr.plot(fut_dates[:len(pred_cum)], pred_cum, "--", label=f"{kind}")
    ax_clr.axhline(0, color="k", linewidth=0.8, alpha=0.5)
    ax_clr.set_title("Cumulative Log Return over Horizon")
    ax_clr.set_ylabel("Cumulative Return")
    ax_clr.grid(True, alpha=0.25)
    ax_clr.legend()

    if len(actual_path):
        ax_pr.plot(fut_dates, actual_path, label="Actual")
    for kind, yhat in preds.items():
        n = min(len(fut_dates), len(yhat))
        if n > 0:
            ax_pr.plot(fut_dates[:n], yhat[:n], "--", label=f"{kind} forecast")
    ax_pr.set_title("Price Overlay over Horizon")
    ax_pr.set_xlabel("Date")
    ax_pr.set_ylabel("Adj Close")
    ax_pr.grid(True, alpha=0.25)
    ax_pr.legend()

    _ensure_path(path)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.show()
    plt.close()

def plot_price_series(df: pd.DataFrame, path: Path | str) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(pd.to_datetime(df["date"]), df["adj_close"])
    ax.set_title("Closing Price Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Adj Close")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(_ensure_path(path), dpi=150)
    plt.show()
    plt.close(fig)

def plot_correlation_heatmap(df: pd.DataFrame, col: List[str], path: Path | str,
                             figsize: Tuple[int, int] = (8, 5)) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    num = df[col].select_dtypes(include=[np.number])
    sns.heatmap(num.corr(), annot=True, cmap="coolwarm", ax=ax)
    ax.set_title("OHLCV Correlation Heatmap")
    fig.tight_layout()
    fig.savefig(_ensure_path(path), dpi=150)
    plt.show()
    plt.close(fig)

def plot_moving_averages(df: pd.DataFrame, path: Path | str) -> None:
    df = df.copy()
    df["sma_10"] = df["adj_close"].rolling(window=10).mean()
    df["ema_10"] = df["adj_close"].ewm(span=10).mean()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(pd.to_datetime(df["date"]), df["adj_close"], label="Adj Close")
    ax.plot(pd.to_datetime(df["date"]), df["sma_10"], label="SMA 10")
    ax.plot(pd.to_datetime(df["date"]), df["ema_10"], label="EMA 10")
    ax.set_title("Moving Averages")
    ax.legend()
    fig.tight_layout()
    fig.savefig(_ensure_path(path), dpi=150)
    plt.show()
    plt.close(fig)

def plot_log_return_distribution(df: pd.DataFrame, path: Path | str, bins: int = 50, log_scale: bool = False) -> None:
    ser = pd.to_numeric(df["log_return"], errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(ser, bins=bins, kde=True, log_scale=log_scale, ax=ax)
    ax.set_title("Log-Returns Distribution")
    fig.tight_layout()
    fig.savefig(_ensure_path(path), dpi=150)
    plt.show()
    plt.close(fig)

def plot_rolling_volatility(df: pd.DataFrame, path: Path | str) -> None:
    df = df.copy()
    ser = pd.to_numeric(df["log_return"], errors="coerce")
    df["volatility_rolling"] = ser.rolling(window=20).std()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(pd.to_datetime(df["date"]), df["volatility_rolling"])
    ax.set_title("Rolling Volatility (20-Day)")
    fig.tight_layout()
    fig.savefig(_ensure_path(path), dpi=150)
    plt.show()
    plt.close(fig)

def plot_autocorrelation(df: pd.DataFrame, path: Path | str, lags: int = 30) -> None:
    ser = pd.to_numeric(df["log_return"], errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(8, 3))
    plot_acf(ser, lags=lags, zero=False, ax=ax)
    ax.set_title("ACF: Log Returns")
    fig.tight_layout()
    fig.savefig(_ensure_path(path), dpi=150)
    plt.show()
    plt.close(fig)

def plot_ohlc_pairplot(df: pd.DataFrame, path: Path | str) -> None:
    g = sns.pairplot(df[["open", "high", "low", "close"]].sample(n=min(len(df), 1000), random_state=42))
    g.fig.suptitle("Pairplot: OHLC", y=1.02)
    g.fig.savefig(_ensure_path(path), dpi=150)
    plt.show()
    plt.close(g.fig)

def plot_sentiment_trend(df: pd.DataFrame, path: Path | str, window: int = 7) -> None:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    df["smoothed"] = df["pos_minus_neg"].rolling(window=window).mean()

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df["date"], df["pos_minus_neg"], label="Daily pos_minus_neg", alpha=0.3, color="green")
    ax.plot(df["date"], df["smoothed"], label=f"{window}-Day Rolling Avg", color="black", linewidth=2)
    ax.axhline(0.0, linestyle="--", color="gray", linewidth=1)
    ax.axhline(0.05, linestyle="--", color="blue", alpha=0.5, linewidth=1)
    ax.axhline(-0.05, linestyle="--", color="red", alpha=0.5, linewidth=1)
    ax.set_title("Sentiment Trend Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("pos_minus_neg")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(_ensure_path(path), dpi=150)
    plt.show()
    plt.close(fig)
