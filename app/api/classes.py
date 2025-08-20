from datetime import date
from typing import List, Optional

from pydantic import BaseModel, Field
from typing_extensions import Annotated


class PriceRow(BaseModel):
    """Schema for a single row of historical price data."""
    date: date
    open: float
    high: float
    low: float
    close: float
    adj_close: float
    volume: float

class NewsRow(BaseModel):
    """Schema for a single news headline."""
    date: date
    rank: int
    headline: str

class PredictionRequest(BaseModel):
    """
    Schema for prediction request.
    Contains:
    - A list of price rows
    - A list of news headlines
    """
    price: Annotated[list[PriceRow], Field(max_length=2000)]
    news: Optional[Annotated[list[NewsRow], Field(max_length=2000)]] = None

class PriceHistoryResponse(BaseModel):
    """Schema for price history response."""
    price: List[PriceRow]

class NewsHistoryResponse(BaseModel):
    """Schema for news history response."""
    news: List[NewsRow]
    message: Optional[str] = None

class PredictionResponse(BaseModel):
    """Full schema for prediction response."""
    horizon: int
    log_return: float
    current_price: float
    predicted_price: float
    log_return_path: Optional[List[float]] = None
    predicted_price_path: Optional[List[float]] = None
    predicted_dates: Optional[List[date]] = None
    last_date: Optional[date] = None
