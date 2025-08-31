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

def _rename_columns(df: pd.DataFrame) -> None:
    """Rename the columns of the dataframe to lower case."""
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

def load_price(path: Path) -> pd.DataFrame:
    """Loads Price Dataset, only used in Jupyter Notebook."""
    df = pd.read_csv(path)
    _rename_columns(df)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)

def load_news(path: Path) -> pd.DataFrame:
    """Loads News Dataset, only used in Jupyter Notebook."""
    df = pd.read_csv(path)
    _rename_columns(df)
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
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        horizon: int = 30
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronologically split into train/val/test, and keep the last `horizon` rows as forecast holdout."""
    df = df.sort_values("date").reset_index(drop=True)

    # Target columns (single- or multi-output)
    target_cols = [c for c in df.columns if c == "target" or c.startswith("target_")]
    if not target_cols:
        raise ValueError("No target columns found. Create the feature dataset first!")

    # Rows with fully observed targets and exclude overlap
    usable = df[df[target_cols].notna().all(axis=1)].copy()
    forecast = df.tail(horizon).copy()
    usable = usable.loc[usable.index < forecast.index.min()]

    total = len(usable)
    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))

    train = usable.iloc[:train_end].copy()
    val = usable.iloc[train_end:val_end].copy()
    test = usable.iloc[val_end:].copy()

    return train, val, test, forecast

@memory.cache
def get_price_history(symbol: str, end_date: str, days: int = 90) -> pd.DataFrame:
    """Fetch Open, High, Low, Close, Volume and Adj Close Prices from Yahoo Finance."""
    end = pd.to_datetime(end_date).normalize()
    start = end - BDay(days - 1)

    df = yf.download(
        symbol,
        start=start.strftime("%Y-%m-%d"),
        end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        auto_adjust=False,
        progress=False,
        threads=True
    )
    if df.empty:
        raise ValueError(f"No data returned for symbol {symbol}")

    df = df.reset_index()

    # Ensure Date column exists
    if "date" not in df.columns and "Date" in df.columns:
        df.rename(columns={"Date": "date"}, inplace=True)

    # Normalize columns: yfinance sometimes returns a MultiIndex
    cols = df.columns
    cols = [c[0] if isinstance(c, tuple) else c for c in cols]
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in cols]

    # Mapping of possible adjusted-close variants to 'adj_close'
    if "adjclose" in df.columns and "adj_close" not in df.columns:
        df.rename(columns={"adjclose": "adj_close"}, inplace=True)
    if "adjusted_close" in df.columns and "adj_close" not in df.columns:
        df.rename(columns={"adjusted_close": "adj_close"}, inplace=True)

    _rename_columns(df)

    expected = ["date", "open", "high", "low", "close", "adj_close", "volume"]
    missing = set(expected) - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df[expected].sort_values("date").reset_index(drop=True)

@memory.cache
def get_news_history(query: str, end_date: str, days: int, api_key: str, url: str) -> pd.DataFrame:
    """Fetch News from NewsAPI (single page, up to 100 results)."""
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
        payload = {}

    articles = (payload or {}).get("articles", []) if isinstance(payload, dict) else []
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
    return df.sort_values("date").reset_index(drop=True)
