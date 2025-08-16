from typing import Tuple

import pandas as pd

from src.logger import get_logger
from src.utils import load_csv


logger = get_logger(__name__)

def _rename_columns(df: pd.DataFrame) -> None:
    """Rename the columns of the dataframe to lower case."""
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

def load_price(path: str) -> pd.DataFrame:
    """Loads Price Dataset."""
    df = load_csv(path)
    _rename_columns(df)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)

def load_news(path: str) -> pd.DataFrame:
    """Loads News Dataset."""
    df = load_csv(path)
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
    return df.sort_values("date").reset_index(drop=True)

def load_prices_sentiment(path: str) -> pd.DataFrame:
    """Loads Combined Sentiment with Prices Dataset."""
    try:
        df = load_csv(path)
        df["date"] = pd.to_datetime(df["date"])
        df.sort_values("date").reset_index(drop=True)
    except FileNotFoundError:
        logger.warning("Sentiment dataset not found, run sentiment notebook first.")
        df = pd.DataFrame()
    return df

def merge_price_news(price: pd.DataFrame, news: pd.DataFrame) -> pd.DataFrame:
    """Merge price and news data."""
    return (pd.merge(price, news, on="date", how="left", validate="many_to_many")
            .sort_values("date")
            .reset_index(drop=True))

def time_series_split(df: pd.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.1, horizon: int = 30) \
        -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataframe into train, test, validation and forecast sets. """
    df = df.sort_values("date").reset_index(drop=True)
    total_len = len(df)

    forecast = df.tail(horizon).copy()

    train_end = int(train_ratio * total_len)
    val_end = int((train_ratio + val_ratio) * total_len)

    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]

    return train, val, test, forecast
