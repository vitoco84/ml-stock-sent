from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Literal, Mapping, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.typing import NDArray
from statsmodels.graphics.tsaplots import plot_acf


_DPI_DEFAULT: int = 150
_FIGSIZE_WIDE: Tuple[int, int] = (12, 5)
_FIGSIZE_STD: Tuple[int, int] = (10, 4)
_PHASE = Literal["val", "test"]
_ADJ_CLOSE = "Adj Close"

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
    """Convert a date-like Series to a timezone-naive numpy datetime64 array."""
    dt = pd.to_datetime(s, utc=True, errors="coerce").dt.tz_convert(None)
    return dt.to_numpy(dtype="datetime64[ns]")

def _extract_delta(y_pred: Iterable[float] | None) -> NDArray[np.float64]:
    """Extract a 1D float array of log-returns."""
    if y_pred is None:
        return np.array([], dtype=float)

    arr = np.asarray(list(y_pred))
    if arr.size == 0:
        return np.array([], dtype=float)

    if arr.ndim == 2:
        arr = arr[:, 0]
    else:
        arr = arr.ravel()

    arr = pd.to_numeric(pd.Series(arr), errors="coerce").to_numpy(dtype=float)
    return arr[np.isfinite(arr)]

def _align_price_path(base_today: NDArray[np.float64], logret_t1: NDArray[np.float64]) -> NDArray[np.float64]:
    """Given today's base prices and (t+1) **deltas**, align lengths and build price path."""
    n = min(len(base_today), len(logret_t1))
    if n <= 0:
        return np.array([], dtype=float)
    return base_today[:n] + logret_t1[:n]

def _next_step_arrays(df: pd.DataFrame) -> Tuple[NDArray[np.datetime64], NDArray[np.float64], NDArray[np.float64]]:
    """
    For a series of dates and adjusted close, return:
      - dates[1:] (timestamps that correspond to t+1 actual)
      - actual_next = adj_close[1:]
      - base_today  = adj_close[:-1]
    """
    dates = _dates_np(df["date"])
    adj = pd.to_numeric(df["adj_close"], errors="coerce").to_numpy(dtype=float)
    return dates[1:], adj[1:], adj[:-1]

def plot_val_test_overlay(
        df_val: pd.DataFrame,
        df_test: pd.DataFrame,
        results: List[Mapping],
        path: Path | str,
) -> None:
    """
    Overlay actual (t+1) adjusted close vs. predicted paths (val+test concatenated).
    Draws a split line at the end of validation segment.
    """
    df_all = pd.concat([df_val, df_test], ignore_index=True)
    dates_all, actual_all, base_today = _next_step_arrays(df_all)
    split_date = _dates_np(df_val["date"])[-1]

    fig, ax = plt.subplots(figsize=_FIGSIZE_WIDE)
    ax.plot(dates_all, actual_all, "--", label="Actual (t+1)", linewidth=2)

    for res in results:
        lr_val = _extract_delta(res.get("y_pred_val"))
        lr_test = _extract_delta(res.get("y_pred_test"))
        if lr_val.size == 0 and lr_test.size == 0:
            continue
        lr_all = np.concatenate([lr_val, lr_test]) if lr_test.size else lr_val
        yhat = _align_price_path(base_today, lr_all)
        n = min(len(dates_all), len(yhat))
        if n > 0:
            ax.plot(dates_all[:n], yhat[:n], label=f"{res.get('kind', 'model')} (val+test)", linewidth=2)

    ax.axvline(split_date, color="k", linestyle=":", alpha=0.7, label="Val/Test split")
    ax.set_title("Actual vs Predicted Adj Close (Val + Test, H=1)")
    ax.set_xlabel("Date")
    ax.set_ylabel(_ADJ_CLOSE)
    ax.grid(True, alpha=0.25)
    ax.legend()

    _finalize_figure(fig, path)

def plot_overlay(
        df: pd.DataFrame,
        results: List[Mapping],
        path: Path | str,
        *,
        phase: _PHASE,
        title: Optional[str] = None,
) -> None:
    """Overlay actual (t+1) adjusted close vs. predicted paths for a given phase."""
    dates_next, actual_next, base_today = _next_step_arrays(df)

    fig, ax = plt.subplots(figsize=_FIGSIZE_WIDE)
    ax.plot(dates_next, actual_next, "--", label="Actual (t+1)", linewidth=2)

    pred_key = f"y_pred_{phase}"
    for res in results:
        lr = _extract_delta(res.get(pred_key))
        yhat = _align_price_path(base_today, lr)
        n = min(len(dates_next), len(yhat))
        if n > 0:
            ax.plot(
                dates_next[:n],
                yhat[:n],
                label=f"{res.get('kind', 'model')} (t+1)",
                linewidth=2,
            )

    if title is None:
        title = (
            "Actual vs Predicted Adj Close (Validation, H=1)"
            if phase == "val"
            else "Actual vs Predicted Adj Close (Test Set, H=1)"
        )

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel(_ADJ_CLOSE)
    ax.grid(True, alpha=0.25)
    ax.legend()

    _finalize_figure(fig, path)

def plot_val_overlay(
        df_val: pd.DataFrame,
        results: List[Mapping],
        path: Path | str,
) -> None:
    plot_overlay(df_val, results, path, phase="val")

def plot_test_overlay(
        df_test: pd.DataFrame,
        results: List[Mapping],
        path: Path | str,
) -> None:
    plot_overlay(df_test, results, path, phase="test")

def plot_forecast_overlay(
        df_test: pd.DataFrame,
        df_forecast: pd.DataFrame,
        results: List[Mapping],
        path: Path | str,
) -> None:
    """
    Plot forecast paths against actuals forward of the last test point,
    including a naive (flat) baseline for the first H days.
    """
    H = int(results[0]["horizon"])

    hist_dates = _dates_np(df_test["date"])
    hist_prices = pd.to_numeric(df_test["adj_close"], errors="coerce").to_numpy(dtype=float)
    anchor_date = hist_dates[-1]
    p0 = float(hist_prices[-1])

    fut_dates = _dates_np(df_forecast["date"])
    actual_path = pd.to_numeric(df_forecast["adj_close"], errors="coerce").to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=_FIGSIZE_WIDE)
    ax.plot(hist_dates, hist_prices, label="History (adj_close)", alpha=0.9)
    ax.axvline(anchor_date, linestyle=":", alpha=0.7)

    if len(fut_dates) and len(actual_path):
        ax.plot(fut_dates, actual_path, label=f"Actual next {len(actual_path)}d", linewidth=2)

    if len(fut_dates):
        ax.plot(
            fut_dates[:H],
            [p0] * min(H, len(fut_dates)),
            label="Naive baseline",
            linestyle="--",
            linewidth=2,
            alpha=0.7,
        )

    for res in results:
        delta_path = _extract_delta(res.get("y_pred_last"))[:H]
        if delta_path.size == 0:
            continue
        forecast_prices = p0 + delta_path
        n = min(len(fut_dates), len(forecast_prices))
        if n > 0:
            ax.plot(
                fut_dates[:n],
                forecast_prices[:n],
                "--",
                linewidth=2,
                label=f"{res.get('kind', 'model')} forecast",
            )

    ad = pd.to_datetime(anchor_date).date()
    ax.set_title(f"Forecast vs Actuals from {ad} (H={H}d)")
    ax.set_xlabel("Date")
    ax.set_ylabel(_ADJ_CLOSE)
    ax.grid(True, alpha=0.25)
    ax.legend()

    _finalize_figure(fig, path)

def plot_forecast_diagnostics(
        df_forecast: pd.DataFrame,
        df_test: pd.DataFrame,
        results: List[Mapping],
        path: Path | str,
) -> None:
    """
    Diagnostic:
      1) Residuals (Actual − Forecast)
      2) Cumulative log return (actual vs forecasts)
      3) Price overlay (actual vs forecasts)
    """
    H = int(results[0]["horizon"])

    fut_dates = _dates_np(df_forecast["date"])
    actual_path = pd.to_numeric(df_forecast["adj_close"], errors="coerce").to_numpy(dtype=float)
    p0 = pd.to_numeric(df_test["adj_close"], errors="coerce").to_numpy(dtype=float)[-1]

    preds: Dict[str, NDArray[np.float64]] = {
        str(res.get("kind", "model")): p0 + np.asarray(res.get("y_pred_last")).ravel()[:H]
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
        ax_res.set_title("Residuals over Horizon (Actual − Forecast)")
        ax_res.set_ylabel("Residual (Price)")
        ax_res.grid(True, alpha=0.25)
        ax_res.legend()
    else:
        ax_res.text(0.02, 0.5, "No actuals in forecast window", transform=ax_res.transAxes)
        ax_res.set_axis_off()

    # Cumulative Log Return
    if len(actual_path):
        act_cum = np.cumsum(np.log(actual_path / actual_path[0]))
        ax_clr.plot(fut_dates[: len(act_cum)], act_cum, label="Actual")
    for kind, yhat in preds.items():
        pred_cum = np.cumsum(np.log(yhat / yhat[0]))
        ax_clr.plot(fut_dates[: len(pred_cum)], pred_cum, "--", label=f"{kind}")
    ax_clr.axhline(0, color="k", linewidth=0.8, alpha=0.5)
    ax_clr.set_title("Cumulative Log Return over Horizon")
    ax_clr.set_ylabel("Cumulative Return")
    ax_clr.grid(True, alpha=0.25)
    ax_clr.legend()

    # Price Overlay
    if len(actual_path):
        ax_pr.plot(fut_dates, actual_path, label="Actual")
    for kind, yhat in preds.items():
        n = min(len(fut_dates), len(yhat))
        if n > 0:
            ax_pr.plot(fut_dates[:n], yhat[:n], "--", label=f"{kind} forecast")
    ax_pr.set_title("Price Overlay over Horizon")
    ax_pr.set_xlabel("Date")
    ax_pr.set_ylabel(_ADJ_CLOSE)
    ax_pr.grid(True, alpha=0.25)
    ax_pr.legend()

    _finalize_figure(fig, path)

def plot_price_series(df: pd.DataFrame, path: Path | str) -> None:
    """Time series plot of adjusted close."""
    fig, ax = plt.subplots(figsize=_FIGSIZE_STD)
    ax.plot(pd.to_datetime(df["date"]), df["adj_close"])
    ax.set_title("Closing Price Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel(_ADJ_CLOSE)
    ax.grid(True)
    _finalize_figure(fig, path)

def plot_correlation_heatmap(
        df: pd.DataFrame,
        col: List[str],
        path: Path | str,
        figsize: Tuple[int, int] = (8, 5),
) -> None:
    """Correlation heatmap over numeric subset of provided columns."""
    fig, ax = plt.subplots(figsize=figsize)
    num = df[col].select_dtypes(include=[np.number])
    sns.heatmap(num.corr(), annot=True, cmap="coolwarm", ax=ax)
    ax.set_title("OHLCV Correlation Heatmap")
    _finalize_figure(fig, path)

def plot_moving_averages(df: pd.DataFrame, path: Path | str) -> None:
    """Plot price with 10-day SMA and EMA."""
    df_ = df.copy()
    df_["sma_10"] = df_["adj_close"].rolling(window=10).mean()
    df_["ema_10"] = df_["adj_close"].ewm(span=10).mean()

    fig, ax = plt.subplots(figsize=_FIGSIZE_STD)
    t = pd.to_datetime(df_["date"])
    ax.plot(t, df_["adj_close"], label=_ADJ_CLOSE)
    ax.plot(t, df_["sma_10"], label="SMA 10")
    ax.plot(t, df_["ema_10"], label="EMA 10")
    ax.set_title("Moving Averages")
    ax.legend()
    _finalize_figure(fig, path)

def plot_delta_distribution(
        df: pd.DataFrame,
        path: Path | str,
        bins: int = 50
) -> None:
    """Histogram + KDE of daily delta-Price (1-day price change)."""
    ser = pd.to_numeric(df["delta_1"], errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(ser, bins=bins, kde=True, ax=ax)
    ax.set_title("delta-Price Distribution (1-Day Change)")
    ax.set_xlabel("delta-Price")
    ax.set_ylabel("Frequency")
    _finalize_figure(fig, path)

def plot_rolling_delta_volatility(
        df: pd.DataFrame,
        path: Path | str,
        window: int = 20
) -> None:
    """Rolling volatility (std) of delta-Price."""
    df_ = df.copy()
    df_["volatility_delta"] = pd.to_numeric(df_["delta_1"], errors="coerce").rolling(window=window).std()

    fig, ax = plt.subplots(figsize=_FIGSIZE_STD)
    ax.plot(pd.to_datetime(df_["date"]), df_["volatility_delta"])
    ax.set_title(f"Rolling Volatility (delta-Price, {window}-Day Window)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Std of delta-Price")
    ax.grid(True)
    _finalize_figure(fig, path)

def plot_delta_autocorrelation(df: pd.DataFrame, path: Path | str, lags: int = 30) -> None:
    """Autocorrelation of delta-Price (ACF)."""
    ser = pd.to_numeric(df["delta_1"], errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(8, 3))
    plot_acf(ser, lags=lags, zero=False, ax=ax)
    ax.set_title(f"ACF: delta-Price (lags={lags})")
    _finalize_figure(fig, path)

def plot_ohlc_pairplot(df: pd.DataFrame, path: Path | str) -> None:
    """Seaborn pairplot over OHLC (sample up to 1000 rows for speed)."""
    g = sns.pairplot(
        df[["open", "high", "low", "close"]].sample(n=min(len(df), 1000), random_state=42)
    )
    g.fig.suptitle("Pairplot: OHLC", y=1.02)
    _ensure_parent(path)
    g.fig.tight_layout()
    g.fig.savefig(path, dpi=_DPI_DEFAULT)
    plt.show()
    plt.close(g.fig)

def plot_seasonality_dow(df: pd.DataFrame, path: Path | str) -> None:
    """Average delta-Price by day-of-week."""
    df_ = df.copy()
    df_["dow"] = pd.to_datetime(df_["date"]).dt.dayofweek
    df_["delta_1"] = pd.to_numeric(df_["delta_1"], errors="coerce")
    avg = df_.groupby("dow")["delta_1"].mean()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Mon", "Tue", "Wed", "Thu", "Fri"], avg)
    ax.set_title("Average delta-Price by Day of Week")
    ax.set_ylabel("Mean delta-Price")
    _finalize_figure(fig, path)

def plot_feature_vs_delta(df: pd.DataFrame, feature: str, path: Path | str) -> None:
    """Scatterplot of a feature vs. delta-Price."""
    df_ = df.copy()
    df_["delta_1"] = pd.to_numeric(df_["delta_1"], errors="coerce")
    df_[feature] = pd.to_numeric(df_[feature], errors="coerce")

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(x=df_[feature], y=df_["delta_1"], alpha=0.6, ax=ax)
    ax.set_title(f"{feature} vs delta-Price (1-Day)")
    ax.set_xlabel(feature)
    ax.set_ylabel("delta-Price")
    _finalize_figure(fig, path)

def plot_sentiment_trend(
        df: pd.DataFrame,
        path: Path | str,
        window: int = 7,
) -> None:
    """Plot daily sentiment (pos_minus_neg) with rolling mean and threshold lines."""
    df_ = df.copy()
    df_["date"] = pd.to_datetime(df_["date"])
    df_ = df_.sort_values("date")
    df_["smoothed"] = df_["pos_minus_neg"].rolling(window=window).mean()

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df_["date"], df_["pos_minus_neg"], label="Daily pos_minus_neg", alpha=0.3, color="green")
    ax.plot(df_["date"], df_["smoothed"], label=f"{window}-Day Rolling Avg", color="black", linewidth=2)
    ax.axhline(0.0, linestyle="--", color="gray", linewidth=1)
    ax.axhline(0.05, linestyle="--", color="blue", alpha=0.5, linewidth=1)
    ax.axhline(-0.05, linestyle="--", color="red", alpha=0.5, linewidth=1)
    ax.set_title("Sentiment Trend Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("pos_minus_neg")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()

    _finalize_figure(fig, path)
