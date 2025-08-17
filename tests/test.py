import json

import pandas as pd
from fastapi.testclient import TestClient

from app.api.main import app
from src.data import _rename_columns, load_prices_sentiment
from src.utils import is_cuda_available


client = TestClient(app)

def test_is_cuda_available():
    assert is_cuda_available() == True

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

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "API is up and running!"}

def test_price_history():
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

def test_predict_raw_from_file():
    with TestClient(app) as client:
        with open("test/payload_predict.json", "r", encoding="utf-8") as f:
            payload = json.load(f)

        response = client.post("/predict-raw", json=payload)
        assert response.status_code == 200
        data = response.json()

        assert "log_return" in data
        assert "current_price" in data
        assert "predicted_price" in data
