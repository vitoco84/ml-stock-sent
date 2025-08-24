from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.typing import NDArray
from statsmodels.graphics.tsaplots import plot_acf


# --- Utilities ---

def _ensure_path(path: Path | str) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

def _dt_naive(s: pd.Series) -> pd.Series:
    d = pd.to_datetime(s, errors="coerce")
    try:
        return d.dt.tz_localize(None)
    except Exception:
        return d

# --- Data prep ---

def prep_h1_overlay(
        df_full: pd.DataFrame,
        results: list[dict],
) -> tuple[NDArray[np.datetime64], NDArray[np.float64], dict[str, NDArray[np.float64]]]:
    df = df_full.copy()
    df["date"] = _dt_naive(df["date"])

    ti = np.asarray(results[0]["test_index"], dtype=int)
    valid = (ti + 1) < len(df)
    i0, i1 = ti[valid], (ti + 1)[valid]

    dates_next = df.loc[i1, "date"].to_numpy()
    actual_next = df.loc[i1, "adj_close"].to_numpy(float)
    p_t = df.loc[i0, "adj_close"].to_numpy(float)

    pred_next_by_model: Dict[str, NDArray[np.float64]] = {}
    for res in results:
        ypred = np.asarray(res["y_pred_test"])
        lr1 = ypred[valid, 0] if ypred.ndim == 2 else ypred[valid].ravel()
        pred_next_by_model[res["kind"]] = p_t * np.exp(lr1)
    return dates_next, actual_next, pred_next_by_model

def prep_h_overlay(
        df_full: pd.DataFrame,
        results: list[dict],
        H: int,
        hist_window: int = 200,
) -> tuple[
    NDArray[np.datetime64], NDArray[np.float64],
    NDArray[np.datetime64], NDArray[np.float64],
    pd.Timestamp, dict[str, NDArray[np.float64]]
]:
    df = df_full.copy()
    df["date"] = _dt_naive(df["date"])

    ti = np.asarray(results[0]["test_index"], dtype=int)
    anchor_idx = int(ti[-1])

    anchor_date = pd.to_datetime(df.loc[anchor_idx, "date"])
    p0 = float(df.loc[anchor_idx, "adj_close"])

    hist = df.iloc[max(0, anchor_idx - (hist_window - 1)): anchor_idx + 1]
    hist_dates = hist["date"].to_numpy()
    hist_prices = hist["adj_close"].to_numpy(float)

    future = df.iloc[anchor_idx + 1: anchor_idx + 1 + H]
    fut_dates = pd.to_datetime(future["date"]).to_numpy()
    actual_path = future["adj_close"].to_numpy(float)

    forecast_paths_by_model: Dict[str, NDArray[np.float64]] = {}
    for res in results:
        lr_path = np.asarray(res["y_pred_last"]).ravel()[:H]
        forecast_paths_by_model[res["kind"]] = p0 * np.exp(np.cumsum(lr_path))

    return hist_dates, hist_prices, fut_dates, actual_path, anchor_date, forecast_paths_by_model

# --- Plotting ---

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

def plot_test_overlay_h1(
        dates_next: Iterable[pd.Timestamp],
        actual_next: NDArray[np.float64],
        pred_next_by_model: Dict[str, NDArray[np.float64]],
        path: Path | str,
        title: str = "Actual vs Predicted Adj Close (H=1)",
        ylabel: str = "Adj Close",
) -> None:
    dates_next = np.asarray(dates_next)
    y_act = np.asarray(actual_next, float)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(dates_next, y_act, "--", label="Actual (t+1)", linewidth=2)

    for kind, yhat in pred_next_by_model.items():
        yhat = np.asarray(yhat, float)
        n = min(len(dates_next), len(yhat))
        if n == 0:
            continue
        ax.plot(dates_next[:n], yhat[:n], label=f"{kind} (t+1)", linewidth=2)

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(_ensure_path(path), dpi=150)
    plt.show()
    plt.close(fig)

def plot_forecast_next_h(
        hist_dates: Iterable[pd.Timestamp],
        hist_prices: NDArray[np.float64],
        fut_dates: Iterable[pd.Timestamp],
        forecast_paths_by_model: Dict[str, NDArray[np.float64]],
        path: Path | str,
        anchor_date: Optional[pd.Timestamp] = None,
        actual_path: Optional[NDArray[np.float64]] = None,
        H: Optional[int] = None,
        ylabel: str = "Price",
        title_prefix: str = "Forecast vs Actuals",
) -> None:
    hist_dates = np.asarray(hist_dates)
    hist_prices = np.asarray(hist_prices, float)
    fut_dates = np.asarray(fut_dates)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(hist_dates, hist_prices, label="History (adj_close)", alpha=0.85)

    if anchor_date is not None:
        ax.axvline(pd.to_datetime(anchor_date), linestyle=":", alpha=0.7)

    if actual_path is not None and len(actual_path) and len(fut_dates):
        n = min(len(fut_dates), len(actual_path))
        ax.plot(fut_dates[:n], np.asarray(actual_path, float)[:n], label=f"Actual next {n}d", linewidth=2)

    for kind, path_vals in forecast_paths_by_model.items():
        y = np.asarray(path_vals, float)
        n = min(len(fut_dates), len(y))
        if n == 0:
            continue
        ax.plot(fut_dates[:n], y[:n], "--", linewidth=2, label=f"{kind} forecast")

    if H is None:
        H = len(fut_dates)
    suffix = f"(H={H}d)" if H else ""
    date_txt = f" from {pd.to_datetime(anchor_date).date()}" if anchor_date is not None else ""
    ax.set_title(f"{title_prefix}{date_txt} {suffix}".strip())
    ax.set_xlabel("Date")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(_ensure_path(path), dpi=150)
    plt.show()
    plt.close(fig)

def plot_forecast_diagnostics(
        fut_dates: Iterable[pd.Timestamp],
        forecast_paths_by_model: Dict[str, NDArray[np.float64]],
        path: Path | str,
        actual_path: Optional[NDArray[np.float64]] = None,
        ylabel: str = "Price",
) -> None:
    fut_dates = np.asarray(fut_dates)

    fig, (ax_res, ax_clr, ax_pr) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    if actual_path is not None and len(fut_dates) and len(actual_path):
        y_act = np.asarray(actual_path, float)
        for kind, yhat in forecast_paths_by_model.items():
            yhat = np.asarray(yhat, float)
            n = min(len(fut_dates), len(y_act), len(yhat))
            if n == 0:
                continue
            ax_res.plot(fut_dates[:n], y_act[:n] - yhat[:n], label=kind)
        ax_res.axhline(0, color="k", linewidth=0.8, alpha=0.5)
        ax_res.set_title("Residuals over Horizon (Actual − Forecast)")
        ax_res.set_ylabel("Residual (Price)")
        ax_res.grid(True, alpha=0.25)
        ax_res.legend()
    else:
        ax_res.text(0.02, 0.5, "No actuals available beyond anchor", transform=ax_res.transAxes)
        ax_res.set_axis_off()

    if actual_path is not None and len(actual_path):
        y_act = np.asarray(actual_path, float)
        n = min(len(fut_dates), len(y_act))
        act_cum = np.cumsum(np.log(y_act[:n] / float(y_act[0])))
        ax_clr.plot(fut_dates[:n], act_cum, label="Actual Cumulative Return")
    for kind, yhat in forecast_paths_by_model.items():
        yhat = np.asarray(yhat, float)
        m = min(len(fut_dates), len(yhat))
        if m > 0:
            pred_cum = np.cumsum(np.log(yhat[:m] / float(yhat[0])))
            ax_clr.plot(fut_dates[:m], pred_cum, "--", label=f"{kind} pred")
    ax_clr.axhline(0, color="k", linewidth=0.8, alpha=0.5)
    ax_clr.set_title("Cumulative Log Return over Horizon")
    ax_clr.set_ylabel("Cumulative Return")
    ax_clr.grid(True, alpha=0.25)
    ax_clr.legend()

    if actual_path is not None and len(actual_path):
        n = min(len(fut_dates), len(actual_path))
        ax_pr.plot(fut_dates[:n], np.asarray(actual_path, float)[:n], label="Actual")
    for kind, yhat in forecast_paths_by_model.items():
        yhat = np.asarray(yhat, float)
        m = min(len(fut_dates), len(yhat))
        if m == 0:
            continue
        ax_pr.plot(fut_dates[:m], yhat[:m], "--", label=f"{kind} forecast")
    ax_pr.set_title("Price Overlay over Horizon")
    ax_pr.set_xlabel("Date")
    ax_pr.set_ylabel(ylabel)
    ax_pr.grid(True, alpha=0.25)
    ax_pr.legend()

    fig.tight_layout()
    fig.savefig(_ensure_path(path), dpi=150)
    plt.show()
    plt.close(fig)
