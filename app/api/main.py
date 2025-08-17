import os
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, Query, Request

from app.api.classes import PredictionRequest
from src.config import Config
from src.data import get_news_history, get_price_history
from src.features import generate_full_feature_row
from src.llm import enrich_news_with_generated
from src.logger import get_logger
from src.sentiment import FinBERT
from src.train import ModelTrainer
from src.utils import is_cuda_available


logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context.
    - Loads configuration
    - Initializes FinBERT sentiment model
    - Loads trained prediction model and preprocessor
    - Loads NEWS_API_KEY from .env
    - Stores initialized objects in `app.state` for later access
    """
    config = Config("config/config.yaml")
    sentiment_model = FinBERT(config, device="cuda" if is_cuda_available() else "cpu")
    model_path = Path(__file__).resolve().parents[2] / "data" / "models" / "lin_reg_best.pkl"
    model, preprocessor = ModelTrainer.load(str(model_path))

    load_dotenv()
    api_key = os.getenv("NEWS_API_KEY")
    app.state.news_api_key = api_key
    app.state.sentiment_model = sentiment_model
    app.state.model = model
    app.state.preprocessor = preprocessor

    yield

app = FastAPI(
    title="Stock Prediction API",
    description="Predict stock prices using historical prices, news sentiment, and FinBERT.",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/")
def root():
    """Return a simple message to confirm the API is running."""
    return {"message": "API is up and running!"}

@app.get("/price-history")
def fetch_price_history(
        symbol: str = Query(..., description="Ticker symbol, e.g., AAPL, ^DJI"),
        end_date: str = Query(..., description="End date in YYYY-MM-DD format"),
        days: int = Query(90, description="Number of calendar days to look back")
):
    """
    Fetch historical stock price data for a given symbol.

    - **symbol**: Ticker symbol (AAPL, TSLA, ^DJI, etc.)
    - **end_date**: End date in YYYY-MM-DD format
    - **days**: Number of calendar days to fetch (default: 90)

    Returns JSON with price rows or an error message.
    """
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

@app.get("/news-history")
def fetch_news_history(
        query: str = Query(..., description="Search keyword, e.g., Apple, Tesla"),
        end_date: str = Query(..., description="End date in YYYY-MM-DD format"),
        days: int = Query(7, description="Number of calendar days to look back"),
        request: Request = None
):
    """
    Fetch recent news headlines using the NewsAPI.

    - **query**: Keyword to search (e.g., 'Apple')
    - **end_date**: End date in YYYY-MM-DD format
    - **days**: Number of days to look back

    Returns JSON with news rows or an error message.
    """
    try:
        api_key = request.app.state.news_api_key
        if not api_key:
            return {"error": "Missing NEWS_API_KEY environment variable"}

        df = get_news_history(query=query, end_date=end_date, days=days, api_key=api_key)

        if df.empty:
            return {"news": [], "message": "No news found."}

        return {"news": df.to_dict(orient="records")}

    except Exception as e:
        return {"error": str(e)}

@app.post("/predict-raw")
def post_predict_from_raw(
        request_body: PredictionRequest,
        request: Request,
        enrich: bool = Query(False, description="Generate missing headlines using local LLM")
):
    """
    Predict the next stock price using:
    - Historical price data
    - News headlines (optionally enriched with an LLM)
    - FinBERT sentiment analysis
    - A pre-trained regression model

    Returns
    - **log_return**: Predicted log return
    - **current_price**: Latest price
    - **predicted_price**: Predicted next price
    """
    logger.info("Received prediction request")
    sentiment_model = request.app.state.sentiment_model
    model = request.app.state.model
    preprocessor = request.app.state.preprocessor

    # --- PRICE ---
    price_df = pd.DataFrame([row.model_dump() for row in request_body.price])
    price_df["date"] = pd.to_datetime(price_df["date"])
    price_dates = price_df["date"].dt.strftime("%Y-%m-%d").tolist()
    logger.info(f"Price DF:\n{price_df.tail()}")

    # --- NEWS ---
    news_df = pd.DataFrame([row.model_dump() for row in request_body.news])
    if not news_df.empty:
        news_df["date"] = pd.to_datetime(news_df["date"]).dt.normalize()
        logger.info(f"News DF:\n{news_df.tail()}")
    else:
        logger.info("No initial news provided.")

    # --- ENRICH NEWS (if requested) ---
    if enrich:
        logger.info("Enrich flag is ON — generating missing headlines via LLM")
        real_news = news_df.to_dict(orient="records")
        enriched_news = enrich_news_with_generated(price_dates, real_news, symbol="AAPL")
        news_df = pd.DataFrame(enriched_news)
        news_df["date"] = pd.to_datetime(news_df["date"]).dt.normalize()
        logger.info(f"Enriched News DF:\n{news_df.tail()}")
    else:
        logger.info("Enrich flag is OFF — skipping headline generation")

    # --- FEATURE GENERATION ---
    feature_row = generate_full_feature_row(price_df, news_df, sentiment_model)
    logger.info(f"Feature row:\n{feature_row}")

    # --- PREDICTION ---
    X = preprocessor.transform(feature_row)
    log_return = float(model.predict(X)[0])
    current_price = price_df["adj_close"].iloc[-1]
    predicted_price = current_price * float(np.exp(log_return))
    logger.info(f"Prediction done — log return: {log_return}, predicted price: {predicted_price}")

    return {
        "log_return": log_return,
        "current_price": current_price,
        "predicted_price": predicted_price
    }
