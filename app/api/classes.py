from datetime import date
from typing import Optional

from pydantic import BaseModel, Field


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
    headline: str

class PredictionRequest(BaseModel):
    """Schema for prediction request with prices and optional news."""
    price: list[PriceRow] = Field(..., max_length=2000)
    news: Optional[list[NewsRow]] = Field(default=None, max_length=2000)

class PriceHistoryResponse(BaseModel):
    """Schema for price history response."""
    price: list[PriceRow]

class NewsHistoryResponse(BaseModel):
    """Schema for news history response."""
    news: list[NewsRow]
    message: Optional[str] = None

class PredictionResponse(BaseModel):
    """Schema for prediction response."""
    horizon: int
    current_price: float
    log_return: float
    predicted_price: float
    log_return_path: Optional[list[float]] = None
    predicted_price_path: Optional[list[float]] = None
    predicted_dates: Optional[list[date]] = None
    last_date: Optional[date] = None
