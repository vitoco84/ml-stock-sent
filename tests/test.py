import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from src.data import _rename_columns, load_prices_sentiment
from src.llm import enrich_news_with_generated, generate_local_headlines


@pytest.fixture(scope="session")
def client():
    return TestClient(app)

def test_rename_columns():
    data = [
        {"Name": "Alice", "Price": 25, "Adj Price": 32564, "Description Is": ""}
    ]
    df = pd.DataFrame(data)
    _rename_columns(df)
    assert df.columns.tolist() == ["name", "price", "adj_price", "description_is"]

def test_exception_and_logger():
    df = load_prices_sentiment("foobar")
    assert df.empty == True

def test_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "API is up and running!"}

@pytest.mark.integration
def test_price_history(client):
    response = client.get("/price-history", params={
        "symbol": "^DJI",
        "end_date": "2025-08-01",
        "days": 10
    })
    assert response.status_code == 200

    json_data = response.json()

    assert "price" in json_data
    assert isinstance(json_data["price"], list)

    if json_data["price"]:
        row = json_data["price"][0]
        expected_keys = {"date", "open", "high", "low", "close", "adj_close", "volume"}
        assert expected_keys.issubset(set(row.keys()))

@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("NEWS_API_KEY"),
    reason="NEWS_API_KEY not set"
)
def test_news_history(client):
    end_date = datetime.today().strftime("%Y-%m-%d")

    response = client.get("/news-history", params={
        "query": "Apple",
        "end_date": end_date,
        "days": 7
    })

    assert response.status_code == 200

    data = response.json()

    assert "news" in data
    assert isinstance(data["news"], list)

def test_predict_raw_from_file():
    payload_path = Path(__file__).parent / "payload_predict.json"
    with open(payload_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    with TestClient(app) as client:
        response = client.post("/predict-raw?enrich=false", json=payload)

        assert response.status_code == 200, response.text
        data = response.json()
        assert {"log_return", "current_price", "predicted_price"} <= data.keys()

@pytest.mark.integration
def test_generate_local_headlines():
    dates = ["2025-07-01", "2025-07-02"]
    result = generate_local_headlines(symbol="AAPL", dates=dates)

    assert len(result) == len(dates)
    assert "headline" in result[0]

@pytest.mark.integration
def test_enrich_news():
    price_dates = ["2025-07-01", "2025-07-02", "2025-07-03"]
    existing_news = [
        {"date": "2025-07-01", "rank": "top1", "headline": "Apple hits new high"}
    ]
    enriched = enrich_news_with_generated(price_dates, existing_news, symbol="AAPL")

    assert len(enriched) == len(price_dates)
    assert all("date" in row and "headline" in row for row in enriched)
