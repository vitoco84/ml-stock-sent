from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.typing import NDArray
from statsmodels.graphics.tsaplots import plot_acf


_DPI_DEFAULT: int = 150
_FIGSIZE_WIDE: tuple[int, int] = (12, 5)
_FIGSIZE_STD: tuple[int, int] = (10, 4)
_PHASE = str  # "val" | "test"
_ADJ_CLOSE: str = "Adj Close"

def _ensure_parent(path: Path | str) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

def _finalize_figure(
        fig: plt.Figure,
        path: Path | str,
        *,
        dpi: int = _DPI_DEFAULT,
        tight: bool = True,
        show: bool = True,
) -> None:
    if tight:
        fig.tight_layout()
    outfile = _ensure_parent(path)
    fig.savefig(outfile, dpi=dpi)
    if show:
        plt.show()
    plt.close(fig)

def _dates_np(s: pd.Series) -> NDArray[np.datetime64]:
    dt = pd.to_datetime(s, utc=True, errors="coerce").dt.tz_convert(None)
    return dt.to_numpy(dtype="datetime64[ns]")

def _extract_lr(y_pred: object) -> NDArray[np.float64]:
    if y_pred is None:
        return np.array([], dtype=float)
    arr = np.asarray(y_pred)
    if arr.size == 0:
        return np.array([], dtype=float)
    arr = arr[:, 0] if arr.ndim == 2 else arr.ravel()
    arr = pd.to_numeric(arr, errors="coerce").astype(float)
    return arr[np.isfinite(arr)]

def _predicted_prices(base_today: NDArray[np.float64], logrets: NDArray[np.float64]) -> NDArray[np.float64]:
    n = min(len(base_today), len(logrets))
    if n <= 0:
        return np.array([], dtype=float)
    return base_today[:n] * np.exp(logrets[:n])

def _mean_baseline(prices: NDArray[np.float64], H: int) -> NDArray[np.float64]:
    if len(prices) < 2:
        return np.full(H, prices[-1] if len(prices) else np.nan)
    logrets = np.diff(np.log(prices))
    mean_r = np.nanmean(logrets)
    return prices[-1] * np.exp(np.cumsum(np.full(H, mean_r)))

def _next_step_arrays(
        df: pd.DataFrame,
) -> tuple[NDArray[np.datetime64], NDArray[np.float64], NDArray[np.float64]]:
    dates = _dates_np(df["date"])
    adj = pd.to_numeric(df["adj_close"], errors="coerce").to_numpy(dtype=float)
    return dates[1:], adj[1:], adj[:-1]

def plot_overlay(
        df: pd.DataFrame,
        results: list[Mapping],
        path: Path | str,
        *,
        phase: _PHASE,
        title: Optional[str] = None,
) -> None:
    dates_next, actual_next, base_today = _next_step_arrays(df)

    fig, ax = plt.subplots(figsize=_FIGSIZE_WIDE)
    ax.plot(dates_next, actual_next, "--", label=f"Actual ({phase})", linewidth=2)

    pred_key = f"y_pred_{phase}"
    for res in results:
        lr = _extract_lr(res.get(pred_key))
        yhat = _predicted_prices(base_today, lr)
        n = min(len(dates_next), len(yhat))
        if n > 0:
            ax.plot(dates_next[:n], yhat[:n], label=f"{res.get('kind', 'model')} ({phase})", linewidth=2)

    ax.set_title(
        title
        or (
            "Actual vs Predicted Adj Close (Validation, H=1)"
            if phase == "val"
            else "Actual vs Predicted Adj Close (Test, H=1)"
        )
    )
    ax.set_xlabel("Date")
    ax.set_ylabel(_ADJ_CLOSE)
    ax.grid(True, alpha=0.25)
    ax.legend()
    _finalize_figure(fig, path)

def plot_val_overlay(df_val: pd.DataFrame, results: list[Mapping], path: Path | str) -> None:
    plot_overlay(df_val, results, path, phase="val")

def plot_test_overlay(df_test: pd.DataFrame, results: list[Mapping], path: Path | str) -> None:
    plot_overlay(df_test, results, path, phase="test")

def plot_val_test_overlay(
        df_val: pd.DataFrame,
        df_test: pd.DataFrame,
        results: list[Mapping],
        path: Path | str,
) -> None:
    dates_val, actual_val, base_val = _next_step_arrays(df_val)
    dates_test, actual_test, base_test = _next_step_arrays(df_test)
    split_date = dates_val[-1]

    fig, ax = plt.subplots(figsize=_FIGSIZE_WIDE)

    ax.plot(dates_val, actual_val, "--", label="Actual (Val)", linewidth=2, color="C0")
    ax.plot(dates_test, actual_test, "--", label="Actual (Test)", linewidth=2, color="C1")

    for res in results:
        lr_val = _extract_lr(res.get("y_pred_val"))
        if lr_val.size:
            yhat_val = _predicted_prices(base_val, lr_val)
            ax.plot(dates_val[:len(yhat_val)], yhat_val, label=f"{res.get('kind', 'model')} (Val)", linewidth=2,
                    color="C3")

        lr_test = _extract_lr(res.get("y_pred_test"))
        if lr_test.size:
            yhat_test = _predicted_prices(base_test, lr_test)
            ax.plot(dates_test[:len(yhat_test)], yhat_test, label=f"{res.get('kind', 'model')} (Test)", linewidth=2,
                    color="C4")

    ax.axvline(split_date, color="k", linestyle=":", alpha=0.7, label="Val/Test split")
    ax.set_title("Actual vs Predicted Adj Close (Val+Test, H=1)")
    ax.set_xlabel("Date")
    ax.set_ylabel(_ADJ_CLOSE)
    ax.grid(True, alpha=0.25)
    ax.legend()
    _finalize_figure(fig, path)

def plot_forecast_overlay(
        df_test: pd.DataFrame,
        df_forecast: pd.DataFrame,
        results: list[Mapping],
        path: Path | str,
) -> None:
    H = int(results[0]["horizon"])
    hist_dates = _dates_np(df_test["date"])
    hist_prices = pd.to_numeric(df_test["adj_close"], errors="coerce").to_numpy(dtype=float)
    anchor_date, p0 = hist_dates[-1], float(hist_prices[-1])

    fut_dates = _dates_np(df_forecast["date"])
    actual_path = pd.to_numeric(df_forecast["adj_close"], errors="coerce").to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=_FIGSIZE_WIDE)
    ax.plot(hist_dates, hist_prices, label="History (adj_close)", alpha=0.9)
    ax.axvline(anchor_date, linestyle=":", alpha=0.7)

    if len(fut_dates) and len(actual_path):
        ax.plot(fut_dates, actual_path, label="Actual", linewidth=2)

    if len(fut_dates):
        mean_base = _mean_baseline(hist_prices, H)
        n = min(len(fut_dates), len(mean_base))
        ax.plot(fut_dates[:n], mean_base[:n], "--", label="Mean baseline", alpha=0.7)

    for res in results:
        lr_path = _extract_lr(res.get("y_pred_last"))[:H]
        if lr_path.size == 0:
            continue
        forecast = p0 * np.exp(np.cumsum(lr_path))
        n = min(len(fut_dates), len(forecast))
        if n > 0:
            ax.plot(fut_dates[:n], forecast[:n], "--", linewidth=2, label=f"{res.get('kind', 'model')}")

    ax.set_title(f"Forecast vs Actuals from {pd.to_datetime(anchor_date).date()} (H={H})")
    ax.set_xlabel("Date")
    ax.set_ylabel(_ADJ_CLOSE)
    ax.grid(True, alpha=0.25)
    ax.legend()
    _finalize_figure(fig, path)

def plot_forecast_diagnostics(
        df_forecast: pd.DataFrame,
        df_test: pd.DataFrame,
        results: list[Mapping],
        path: Path | str,
) -> None:
    H = int(results[0]["horizon"])
    fut_dates = _dates_np(df_forecast["date"])
    actual_path = pd.to_numeric(df_forecast["adj_close"], errors="coerce").to_numpy(dtype=float)
    p0 = pd.to_numeric(df_test["adj_close"], errors="coerce").to_numpy(dtype=float)[-1]

    preds: dict[str, NDArray[np.float64]] = {
        str(res.get("kind", "model")): p0 * np.exp(np.cumsum(np.asarray(res.get("y_pred_last")).ravel()[:H]))
        for res in results
    }

    fig, (ax_res, ax_clr, ax_pr) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Residuals
    if len(fut_dates) and len(actual_path):
        for kind, yhat in preds.items():
            n = min(len(fut_dates), len(actual_path), len(yhat))
            if n > 0:
                ax_res.plot(fut_dates[:n], actual_path[:n] - yhat[:n], label=kind)
        ax_res.axhline(0, color="k", linewidth=0.8, alpha=0.5)
        ax_res.set_title("Residuals (Actual − Forecast)")
        ax_res.legend()
    else:
        ax_res.text(0.5, 0.5, "No actuals", ha="center")
        ax_res.set_axis_off()

    # Cumulative log return
    if len(actual_path):
        ax_clr.plot(fut_dates, np.cumsum(np.log(actual_path / actual_path[0])), label="Actual")
    for kind, yhat in preds.items():
        ax_clr.plot(fut_dates, np.cumsum(np.log(yhat / yhat[0])), "--", label=kind)
    ax_clr.axhline(0, color="k", linewidth=0.8, alpha=0.5)
    ax_clr.set_title("Cumulative Log Return")
    ax_clr.legend()

    # Price overlay
    if len(actual_path):
        ax_pr.plot(fut_dates, actual_path, label="Actual")
    for kind, yhat in preds.items():
        ax_pr.plot(fut_dates[: len(yhat)], yhat, "--", label=f"{kind}")
    ax_pr.set_title("Forecast Price Overlay")
    ax_pr.legend()

    _finalize_figure(fig, path)

def plot_price_series(df: pd.DataFrame, path: Path | str) -> None:
    fig, ax = plt.subplots(figsize=_FIGSIZE_STD)
    ax.plot(pd.to_datetime(df["date"]), df["adj_close"])
    ax.set_title("Closing Price Over Time")
    ax.set_ylabel(_ADJ_CLOSE)
    ax.grid(True)
    _finalize_figure(fig, path)

def plot_correlation_heatmap(
        df: pd.DataFrame,
        cols: list[str],
        path: Path | str,
        figsize: tuple[int, int] = (8, 5),
) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    num = df[cols].select_dtypes(include=[np.number])
    sns.heatmap(num.corr(), annot=True, cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Heatmap")
    _finalize_figure(fig, path)

def plot_moving_averages(df: pd.DataFrame, path: Path | str) -> None:
    df_ = df.copy()
    df_["sma_10"] = df_["adj_close"].rolling(10).mean()
    df_["ema_10"] = df_["adj_close"].ewm(span=10).mean()

    fig, ax = plt.subplots(figsize=_FIGSIZE_STD)
    t = pd.to_datetime(df_["date"])
    ax.plot(t, df_["adj_close"], label=_ADJ_CLOSE)
    ax.plot(t, df_["sma_10"], label="SMA 10")
    ax.plot(t, df_["ema_10"], label="EMA 10")
    ax.legend()
    _finalize_figure(fig, path)

def plot_return_distribution(df: pd.DataFrame, path: Path | str, bins: int = 50) -> None:
    ser = pd.to_numeric(df["log_return"], errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(ser, bins=bins, kde=True, ax=ax)
    ax.set_title("Log Return Distribution (1-Day)")
    _finalize_figure(fig, path)

def plot_rolling_volatility(df: pd.DataFrame, path: Path | str, window: int = 20) -> None:
    df_ = df.copy()
    df_["vol"] = pd.to_numeric(df_["log_return"], errors="coerce").rolling(window).std()

    fig, ax = plt.subplots(figsize=_FIGSIZE_STD)
    ax.plot(pd.to_datetime(df_["date"]), df_["vol"])
    ax.set_title(f"Rolling Volatility ({window}-day)")
    _finalize_figure(fig, path)

def plot_return_autocorrelation(df: pd.DataFrame, path: Path | str, lags: int = 20) -> None:
    ser = pd.to_numeric(df["log_return"], errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(8, 3))
    plot_acf(ser, lags=lags, zero=False, ax=ax)
    ax.set_title(f"Log Return ACF (lags={lags})")
    _finalize_figure(fig, path)

def plot_ohlc_pairplot(df: pd.DataFrame, path: Path | str) -> None:
    g = sns.pairplot(df[["open", "high", "low", "close"]].sample(n=min(len(df), 1000), random_state=42))
    g.fig.suptitle("OHLC Pairplot", y=1.02)
    _ensure_parent(path)
    g.fig.savefig(path, dpi=_DPI_DEFAULT)
    plt.close(g.fig)

def plot_seasonality_dow(df: pd.DataFrame, path: Path | str) -> None:
    df_ = df.copy()
    df_["dow"] = pd.to_datetime(df_["date"]).dt.dayofweek
    df_["log_return"] = pd.to_numeric(df_["log_return"], errors="coerce")
    avg = df_.groupby("dow")["log_return"].mean()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Mon", "Tue", "Wed", "Thu", "Fri"], avg)
    ax.set_title("Avg Log Return by Day-of-Week")
    _finalize_figure(fig, path)

def plot_feature_vs_return(df: pd.DataFrame, feature: str, path: Path | str) -> None:
    df_ = df.copy()
    df_["log_return"] = pd.to_numeric(df_["log_return"], errors="coerce")
    df_[feature] = pd.to_numeric(df_[feature], errors="coerce")

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(x=df_[feature], y=df_["log_return"], alpha=0.6, ax=ax)
    ax.set_title(f"{feature} vs Log Return")
    _finalize_figure(fig, path)

def plot_sentiment_trend(df: pd.DataFrame, path: Path | str, window: int = 7) -> None:
    df_ = df.copy()
    df_["date"] = pd.to_datetime(df_["date"])
    df_ = df_.sort_values("date")
    df_["smoothed"] = df_["pos_minus_neg"].rolling(window).mean()

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df_["date"], df_["pos_minus_neg"], alpha=0.3, label="Daily")
    ax.plot(df_["date"], df_["smoothed"], color="black", linewidth=2, label=f"{window}-day Avg")
    ax.axhline(0, linestyle="--", color="gray")
    ax.axhline(0.05, linestyle="--", color="blue", alpha=0.5)
    ax.axhline(-0.05, linestyle="--", color="red", alpha=0.5)
    ax.set_title("Sentiment Trend")
    ax.legend()
    _finalize_figure(fig, path)
