from typing import List

from pydantic import BaseModel

# ------------------------------------------------------------
# REQUEST SCHEMA
# ------------------------------------------------------------

class PriceRow(BaseModel):
    """Schema for a single row of historical price data."""
    date: str
    open: float
    high: float
    low: float
    close: float
    adj_close: float
    volume: float

class NewsRow(BaseModel):
    """Schema for a single news headline."""
    date: str
    rank: str
    headline: str

class PredictionRequest(BaseModel):
    """
    Schema for prediction request.
    Contains:
    - A list of price rows
    - A list of news headlines
    """
    price: List[PriceRow]
    news: List[NewsRow]