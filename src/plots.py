from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf


ADJ_CLOSE = "Adj Close"

def plot_price_series(df: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(10, 4))
    plt.plot(df["date"], df["adj_close"])
    plt.title("Closing Price Over Time")
    plt.xlabel("Date")
    plt.ylabel(ADJ_CLOSE)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.show()
    plt.close()

def plot_price_overlay(
        df_feat: pd.DataFrame,
        X_test: pd.DataFrame | None = None,
        y_pred: pd.Series | None = None,
        path: Path | None = None,
        *,
        lr1: float | None = None,
        anchor_idx: int | None = None
):
    ADJ_CLOSE = "Adj Close"

    if lr1 is not None and anchor_idx is not None:
        idx0, idx1 = anchor_idx, anchor_idx + 1
        if idx1 >= len(df_feat):
            raise ValueError("Not enough future rows to plot H=1 overlay (idx1 out of range).")

        d1 = pd.to_datetime(df_feat.iloc[idx1]["date"])
        actual = float(df_feat.iloc[idx1]["adj_close"])
        start_price = float(df_feat.iloc[idx0]["adj_close"])
        pred = start_price * float(np.exp(lr1))

        dates = [d1]
        actual_vals = [actual]
        pred_vals = [pred]

        plt.figure(figsize=(12, 5))
        plt.plot(dates, actual_vals, label="Actual (t+1)", linewidth=2, marker="o")
        plt.plot(dates, pred_vals, label="Predicted (t+1)", linestyle="--", linewidth=2, marker="o")

        ymin = min(actual, pred)
        ymax = max(actual, pred)
        pad = max(1e-6, 0.002 * max(abs(ymin), abs(ymax)))
        plt.ylim(ymin - pad, ymax + pad)

    else:
        idx = X_test.index.to_numpy()
        valid = idx + 1 < len(df_feat)
        idx, idx1 = idx[valid], idx[valid] + 1

        dates = pd.to_datetime(df_feat.loc[idx1, "date"])
        actual_vals = df_feat.loc[idx1, "adj_close"].to_numpy(float)
        pred_vals = df_feat.loc[idx, "adj_close"].to_numpy(float) * np.exp(y_pred[valid, 0])

        plt.figure(figsize=(12, 5))
        plt.plot(dates, actual_vals, label="Actual (t+1)", linewidth=2)
        plt.plot(dates, pred_vals, label="Predicted (t+1)", linestyle="--", linewidth=2)

    plt.title("Actual vs Predicted Adj Close (H=1)")
    plt.xlabel("Date")
    plt.ylabel(ADJ_CLOSE)
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    if path is not None:
        plt.savefig(path)
    plt.show()

def plot_price_overlay_next_30(
        df_feat: pd.DataFrame,
        test_df: pd.DataFrame | None = None,
        y_pred: pd.Series | None = None,
        horizon: int = 30,
        hist_window: int = 200,
        path: Path | None = None,
        *,
        lr_path: np.ndarray | None = None,
        anchor_idx: int | None = None
):
    if lr_path is not None and anchor_idx is not None:
        lr_path = np.asarray(lr_path, dtype=float).ravel()
        H = int(len(lr_path))
        if H == 0:
            raise ValueError("lr_path is empty.")
        horizon = H

        if anchor_idx + 1 + horizon > len(df_feat):
            raise ValueError("Not enough future rows to compare against actuals.")

        anchor_date = pd.to_datetime(df_feat.iloc[anchor_idx]["date"])
        start_price = float(df_feat.iloc[anchor_idx]["adj_close"])
        pred_price_path = start_price * np.exp(np.cumsum(lr_path))

        future = df_feat.iloc[anchor_idx + 1: anchor_idx + 1 + horizon]
        future_dates = pd.to_datetime(future["date"].to_numpy())
        actual_price_path = future["adj_close"].to_numpy(float)

    else:
        anchor_idx = int(test_df.index[-1])
        anchor_date = pd.to_datetime(df_feat.loc[anchor_idx, "date"])
        start_price = float(df_feat.loc[anchor_idx, "adj_close"])

        pred_lr = np.asarray(y_pred[-1]).reshape(-1)[:horizon]
        pred_price_path = start_price * np.exp(np.cumsum(pred_lr))

        future = df_feat.iloc[anchor_idx + 1: anchor_idx + 1 + horizon]
        future_dates = pd.to_datetime(future["date"].to_numpy())
        actual_price_path = future["adj_close"].to_numpy()

    hist = df_feat.iloc[max(0, anchor_idx - hist_window + 1): anchor_idx + 1].copy()
    hist["date"] = pd.to_datetime(hist["date"])

    plt.figure(figsize=(12, 5))
    plt.plot(hist["date"], hist["adj_close"], label="History (adj_close)", alpha=0.8)
    plt.plot(future_dates, actual_price_path, label=f"Actual next {horizon}d", linewidth=2)
    plt.plot(future_dates, pred_price_path, label=f"Forecast next {horizon}d", linestyle="--", linewidth=2)
    plt.axvline(anchor_date, linestyle=":", alpha=0.7)
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title(f"{horizon}-Day Forecast vs Actuals from {anchor_date.date()} (H={horizon})")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    if path is not None:
        plt.savefig(path)
    plt.show()

def plot_correlation_heatmap(df: pd.DataFrame, col: List[str], path: Path, figsize: Tuple = (8, 5)):
    plt.figure(figsize=figsize)
    sns.heatmap(df[col].corr(), annot=True, cmap="coolwarm")
    plt.title("OHLCV Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(path)
    plt.show()
    plt.close()

def plot_moving_averages(df: pd.DataFrame, path: Path):
    df = df.copy()
    df["sma_10"] = df["adj_close"].rolling(window=10).mean()
    df["ema_10"] = df["adj_close"].ewm(span=10).mean()
    plt.figure(figsize=(10, 4))
    plt.plot(df["date"], df["adj_close"], label=ADJ_CLOSE)
    plt.plot(df["date"], df["sma_10"], label="SMA 10")
    plt.plot(df["date"], df["ema_10"], label="EMA 10")
    plt.title("Moving Averages")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.show()
    plt.close()

def plot_log_return_distribution(df: pd.DataFrame, path: Path, bins: int = 50,
                                 log_scale: bool = False):
    plt.figure(figsize=(6, 4))
    sns.histplot(df["log_return"], bins=bins, kde=True, log_scale=log_scale)
    plt.title("Log-Returns Distribution")
    plt.tight_layout()
    plt.savefig(path)
    plt.show()
    plt.close()

def plot_rolling_volatility(df: pd.DataFrame, path: Path):
    df = df.copy()
    df["volatility_rolling"] = df["log_return"].rolling(window=20).std()
    plt.figure(figsize=(10, 4))
    plt.plot(df["date"], df["volatility_rolling"])
    plt.title("Rolling Volatility (20-Day)")
    plt.tight_layout()
    plt.savefig(path)
    plt.show()
    plt.close()

def plot_autocorrelation(df: pd.DataFrame, path: Path, lags=30):
    _, ax = plt.subplots(figsize=(8, 3))
    plot_acf(df["log_return"], lags=lags, zero=False, ax=ax)
    plt.title("ACF: Log Returns")
    plt.tight_layout()
    plt.savefig(path)
    plt.show()
    plt.close()

def plot_ohlc_pairplot(df: pd.DataFrame, path: Path):
    sns.pairplot(df[["open", "high", "low", "close"]].sample(n=1000, random_state=42))
    plt.suptitle("Pairplot: OHLC", y=1.02)
    plt.savefig(path)
    plt.show()
    plt.close()

def plot_forecast_diagnostics(
        future_dates: np.ndarray,
        actual_price_path: np.ndarray,
        pred_price_path: np.ndarray,
        path: Path
):
    # --- Residuals ---
    residuals = actual_price_path - pred_price_path

    plt.figure(figsize=(10, 4))
    plt.plot(future_dates, residuals, label="Residuals (Actual - Forecast)")
    plt.axhline(0, linestyle="--", color="gray", linewidth=1)
    plt.title("Next 30-day Residuals Over Time")
    plt.xlabel("Date")
    plt.ylabel("Residual (Price Difference)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.show()
    plt.close()

    # --- Cumulative Returns ---
    actual_return = np.cumsum(np.log(actual_price_path / actual_price_path[0]))
    pred_return = np.cumsum(np.log(pred_price_path / pred_price_path[0]))

    plt.figure(figsize=(10, 4))
    plt.plot(future_dates, actual_return, label="Actual Cumulative Return")
    plt.plot(future_dates, pred_return, label="Predicted Cumulative Return", linestyle="--")
    plt.title("Next 30-da Cumulative Log Return")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.show()
    plt.close()

    # --- Price Comparison ---
    plt.figure(figsize=(10, 4))
    plt.plot(future_dates, actual_price_path, label="Actual Price")
    plt.plot(future_dates, pred_price_path, label="Predicted Price", linestyle="--")
    plt.title("Next 30-day Forecast vs Actual Price")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.show()
    plt.close()

def plot_sentiment_trend(df: pd.DataFrame, path: Path, window: int = 7):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    df["smoothed"] = df["pos_minus_neg"].rolling(window=window).mean()

    plt.figure(figsize=(14, 5))
    plt.plot(df["date"], df["pos_minus_neg"], label="Daily pos_minus_neg", alpha=0.3, color="green")
    plt.plot(df["date"], df["smoothed"], label=f"{window}-Day Rolling Avg", color="black", linewidth=2)

    plt.axhline(0.0, linestyle="--", color="gray", linewidth=1)
    plt.axhline(0.05, linestyle="--", color="blue", alpha=0.5, linewidth=1)
    plt.axhline(-0.05, linestyle="--", color="red", alpha=0.5, linewidth=1)

    plt.title("Sentiment Trend Over Time", fontsize=14)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("pos_minus_neg", fontsize=12)

    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.show()
    plt.close()
