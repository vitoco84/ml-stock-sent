from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from fastapi import FastAPI, Query, Request
from pydantic import BaseModel

from src.config import Config
from src.data import get_price_history
from src.features import generate_full_feature_row
from src.logger import get_logger
from src.sentiment import FinBERT
from src.train import ModelTrainer
from src.utils import is_cuda_available


logger = get_logger(__name__)

# ------------------------------------------------------------
# INIT
# ------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    config = Config("config/config.yaml")
    sentiment_model = FinBERT(config, device="cuda" if is_cuda_available() else "cpu")
    model_path = Path(__file__).resolve().parents[2] / "data" / "models" / "lin_reg_best.pkl"
    model, preprocessor = ModelTrainer.load(str(model_path))

    app.state.sentiment_model = sentiment_model
    app.state.model = model
    app.state.preprocessor = preprocessor

    yield

app = FastAPI(lifespan=lifespan)

# ------------------------------------------------------------
# REQUEST SCHEMA
# ------------------------------------------------------------

class PriceRow(BaseModel):
    date: str
    open: float
    high: float
    low: float
    close: float
    adj_close: float
    volume: float

class NewsRow(BaseModel):
    date: str
    rank: str
    headline: str

class PredictionRequest(BaseModel):
    price: List[PriceRow]
    news: List[NewsRow]

# ------------------------------------------------------------
# ROUTES
# ------------------------------------------------------------

@app.get("/")
def root():
    return {"message": "API is up and running!"}

@app.get("/price-history")
def fetch_price_history(
        symbol: str = Query(..., description="Ticker symbol, e.g., AAPL, ^DJI"),
        end_date: str = Query(..., description="End date in YYYY-MM-DD format"),
        days: int = Query(90, description="Number of calendar days to look back")
):
    try:
        df = get_price_history(symbol=symbol, end_date=end_date, days=days)

        if df.empty:
            return {"error": "No price data returned. Check the symbol or date range."}

        df["date"] = df["date"].dt.strftime("%Y-%m-%d")
        records = df.to_dict(orient="records")

        logger.info(f"Fetched data types:\n{df.dtypes}")
        logger.info(f"Head of dataframe:\n{df.head()}")
        logger.info(f"Sample record: {df.to_dict(orient='records')[0]}")

        return {"price": records}


    except Exception as e:
        return {"error": str(e)}

@app.post("/predict-raw")
def post_predict_from_raw(request_body: PredictionRequest, request: Request):
    sentiment_model = request.app.state.sentiment_model
    model = request.app.state.model
    preprocessor = request.app.state.preprocessor

    price_df = pd.DataFrame([row.model_dump() for row in request_body.price])
    price_df["date"] = pd.to_datetime(price_df["date"])

    news_df = pd.DataFrame([row.model_dump() for row in request_body.news])
    news_df["date"] = pd.to_datetime(news_df["date"])

    feature_row = generate_full_feature_row(price_df, news_df, sentiment_model)

    X = preprocessor.transform(feature_row)
    log_return = float(model.predict(X)[0])
    current_price = price_df["adj_close"].iloc[-1]
    predicted_price = current_price * float(np.exp(log_return))

    return {
        "log_return": log_return,
        "current_price": current_price,
        "predicted_price": predicted_price
    }
