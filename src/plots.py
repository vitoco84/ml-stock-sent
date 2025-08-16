import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf


def plot_price_series(df: pd.DataFrame, output_dir: str, filename: str) -> None:
    plt.figure(figsize=(10, 4))
    plt.plot(df["date"], df["adj_close"])
    plt.title("Closing Price Over Time")
    plt.xlabel("Date")
    plt.ylabel("Adj Close")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{filename}")
    plt.show()
    plt.close()

def plot_correlation_heatmap(df: pd.DataFrame, output_dir: str, filename: str):
    plt.figure(figsize=(8, 5))
    sns.heatmap(df[["open", "high", "low", "close", "volume"]].corr(), annot=True, cmap="coolwarm")
    plt.title("OHLCV Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{filename}")
    plt.show()
    plt.close()

def plot_moving_averages(df: pd.DataFrame, output_dir: str, filename: str):
    df = df.copy()
    df["sma_10"] = df["close"].rolling(window=10).mean()
    df["ema_10"] = df["close"].ewm(span=10).mean()
    plt.figure(figsize=(10, 4))
    plt.plot(df["date"], df["close"], label="Close")
    plt.plot(df["date"], df["sma_10"], label="SMA 10")
    plt.plot(df["date"], df["ema_10"], label="EMA 10")
    plt.title("Moving Averages")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{filename}")
    plt.show()
    plt.close()

def plot_log_return_distribution(df: pd.DataFrame, output_dir: str, filename: str):
    plt.figure(figsize=(6, 4))
    sns.histplot(df["log_return"], bins=50, kde=True)
    plt.title("Log-Returns Distribution")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{filename}")
    plt.show()
    plt.close()

def plot_rolling_volatility(df: pd.DataFrame, output_dir: str, filename: str):
    df = df.copy()
    df["volatility_rolling"] = df["log_return"].rolling(window=20).std()
    plt.figure(figsize=(10, 4))
    plt.plot(df["date"], df["volatility_rolling"])
    plt.title("Rolling Volatility (20-Day)")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{filename}")
    plt.show()
    plt.close()

def plot_autocorrelation(df: pd.DataFrame, output_dir: str, filename: str, lags=30):
    _, ax = plt.subplots(figsize=(8, 3))
    plot_acf(df["log_return"], lags=lags, ax=ax)
    plt.title("ACF: Log Returns")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{filename}")
    plt.show()
    plt.close()

def plot_ohlc_pairplot(df: pd.DataFrame, output_dir: str, filename: str):
    sns.pairplot(df[["open", "high", "low", "close"]])
    plt.suptitle("Pairplot: OHLC", y=1.02)
    plt.savefig(f"{output_dir}/{filename}")
    plt.show()
    plt.close()
