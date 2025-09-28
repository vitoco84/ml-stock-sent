from datetime import timedelta
from pathlib import Path
from typing import Tuple

import pandas as pd
import requests
import yfinance as yf
from joblib import Memory
from pandas.tseries.offsets import BDay
from requests import RequestException

from src.logger import get_logger


logger = get_logger(__name__)
memory = Memory(location=Path(".cache"), verbose=0)

def _rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with normalized (lowercase, snake_case) column names."""
    df = df.copy()
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
    return df

def load_price(path: Path) -> pd.DataFrame:
    """Load price dataset from CSV."""
    df = pd.read_csv(path)
    df = _rename_columns(df)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)

def load_news(path: Path) -> pd.DataFrame:
    """Load news dataset from CSV and unpivot top headlines."""
    df = pd.read_csv(path)
    df = _rename_columns(df)

    top_cols = [c for c in df.columns if c.startswith("top")]
    df = df.melt(
        id_vars="date",
        value_vars=top_cols,
        var_name="rank",
        value_name="headline"
    ).dropna()
    df["headline"] = df["headline"].astype(str).str.strip()
    df["date"] = pd.to_datetime(df["date"])
    return df.drop(columns=["rank"]).sort_values("date").reset_index(drop=True)

def merge_price_news(price: pd.DataFrame, news: pd.DataFrame) -> pd.DataFrame:
    """Merge price and news on normalized date."""
    left = price.copy()
    left["date"] = pd.to_datetime(left["date"]).dt.normalize()

    right = news.copy()
    if "date" in right.columns:
        right["date"] = pd.to_datetime(right["date"]).dt.normalize()

    return (
        pd.merge(left, right, on="date", how="left", validate="one_to_many")
        .sort_values("date")
        .reset_index(drop=True)
    )

def time_series_split(
        df: pd.DataFrame,
        train_ratio: float,
        val_ratio: float,
        horizon: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronologically split into train/val/test sets with forecast holdout."""
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1.0")

    df = df.sort_values("date").reset_index(drop=True)

    # Target columns (single- or multi-output)
    target_cols = [c for c in df.columns if c == "target" or c.startswith("target_")]
    if not target_cols:
        raise ValueError("No target columns found. Create the feature dataset first!")

    # Rows with fully observed targets and exclude overlap
    usable = df[df[target_cols].notna().all(axis=1)].copy()
    forecast = df.tail(horizon).copy()

    usable = usable.loc[usable.index < forecast.index.min()]
    if usable.empty:
        raise ValueError("Not enough data for train/val/test split before forecast horizon.")

    total = len(usable)
    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))

    train = usable.iloc[:train_end].copy()
    val = usable.iloc[train_end:val_end].copy()
    test = usable.iloc[val_end:].copy()

    return train, val, test, forecast

@memory.cache
def get_price_history(symbol: str, end_date: str, days: int) -> pd.DataFrame:
    """Fetch OHLCV and Adj Close from Yahoo Finance."""
    end = pd.to_datetime(end_date).normalize()
    start = end - BDay(days - 1)

    try:
        df = yf.download(
            symbol,
            start=start.strftime("%Y-%m-%d"),
            end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            auto_adjust=False,
            progress=False,
            threads=True
        )
    except Exception as e:
        logger.error(f"yfinance download failed for {symbol}: {e}")
        raise

    if df.empty:
        raise ValueError(f"No data returned for symbol {symbol}")

    df = df.reset_index()
    if "date" not in df.columns and "Date" in df.columns:
        df.rename(columns={"Date": "date"}, inplace=True)

    # Normalize columns: yfinance sometimes returns a MultiIndex
    cols = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in cols]

    # Mapping of possible adjusted-close variants to 'adj_close'
    if "adjclose" in df.columns and "adj_close" not in df.columns:
        df.rename(columns={"adjclose": "adj_close"}, inplace=True)
    if "adjusted_close" in df.columns and "adj_close" not in df.columns:
        df.rename(columns={"adjusted_close": "adj_close"}, inplace=True)

    df = _rename_columns(df)

    expected = ["date", "open", "high", "low", "close", "adj_close", "volume"]
    missing = set(expected) - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df[expected].sort_values("date").reset_index(drop=True)

@memory.cache
def get_news_history(query: str, end_date: str, days: int, api_key: str, url: str) -> pd.DataFrame:
    """Fetch news headlines from NewsAPI (single page, up to 100 results)."""
    to_date = pd.to_datetime(end_date)
    from_date = to_date - timedelta(days=days)

    params = {
        "q": query,
        "from": from_date.strftime("%Y-%m-%d"),
        "to": to_date.strftime("%Y-%m-%d"),
        "sortBy": "relevancy",
        "language": "en",
        "pageSize": 100,
        "apiKey": api_key
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        payload = response.json()
    except RequestException as e:
        logger.error(f"NewsAPI request failed: {e}")
        return pd.DataFrame(columns=["date", "headline"])

    articles = payload.get("articles", []) if isinstance(payload, dict) else []
    if not articles:
        logger.warning("No articles returned from NewsAPI.")

    records = [
        {
            "date": article["publishedAt"][:10],
            "headline": article["title"]
        }
        for article in articles
    ]

    df = pd.DataFrame(records)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df.sort_values("date").reset_index(drop=True)
