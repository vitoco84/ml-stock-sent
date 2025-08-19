import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.pipeline import Pipeline

from app.api.main import app
from src.config import Config
from src.data import _rename_columns, load_prices_sentiment, time_series_split
from src.features import convert_log_return, create_features_and_target
from src.llm import enrich_news_with_generated
from src.models.classical import LinearElasticNet
from src.preprocessing import get_preprocessor
from src.sentiment import FinBERT


@pytest.fixture(scope="session")
def client():
    return TestClient(app)

@pytest.fixture(scope="session")
def config():
    return Config(Path("config/config.yaml"))

def test_create_features_and_target_minimal():
    # Minimal dummy price data with required 'adj_close' and 'date'
    dates = pd.date_range("2024-01-01", periods=60, freq="B")
    df = pd.DataFrame({
        "date": dates,
        "adj_close": np.linspace(100, 150, len(dates))
    })

    features = create_features_and_target(df, forecast_horizon=3)

    assert not features.empty
    assert "target_1" in features.columns
    assert "log_return" in features.columns

def test_convert_log_return_basic():
    price = 100.0
    log_return = np.log(1.10)
    future_price = convert_log_return(price, log_return)

    assert abs(future_price - 110.0) < 0.01

def test_config_loads_yaml_and_resolves_paths():
    with tempfile.NamedTemporaryFile("w+", suffix=".yaml", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        tmp.write("""
        general:
          name: "simple-test"
        data:
          raw_dir: "../data/raw"
        model:
          path: "../data/models/model.pkl"
        urls:
          api: "http://localhost:8000"
        """)
        tmp.flush()

    try:
        config = Config(tmp_path)

        assert config.general.name == "simple-test"

        assert isinstance(config.data.raw_dir, Path)
        assert config.data.raw_dir.is_absolute()
        assert config.data.raw_dir.name == "raw"

        assert isinstance(config.model.path, Path)
        assert config.model.path.is_absolute()
        assert config.model.path.name == "model.pkl"

        assert isinstance(config.urls.api, str)
        assert config.urls.api == "http://localhost:8000"

    finally:
        tmp_path.unlink()

def test_rename_columns():
    df = pd.DataFrame(columns=["Open", "Adj Close ", " Volume"])
    _rename_columns(df)
    assert list(df.columns) == ["open", "adj_close", "volume"]

def test_time_series_split():
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=100),
        "feature": range(100),
        "target": range(100)
    })

    train, val, _, forecast = time_series_split(df, horizon=10)

    assert len(train) > len(val) > 0
    assert len(forecast) == 10
    assert forecast["date"].max() == df["date"].max()

def test_exception_and_logger():
    df = load_prices_sentiment("foobar")
    assert df.empty == True

def test_linear_elasticnet_multioutput(config):
    generator = np.random.default_rng(config.runtime.seed)
    X = pd.DataFrame(generator.random((10, 5)), columns=[f"f{i}" for i in range(5)])
    y = pd.DataFrame(generator.random((10, 3)))

    model = LinearElasticNet(horizon=3, multioutput=True)
    model.fit(X, y)

    preds = model.predict(X)

    assert isinstance(preds, pd.DataFrame)
    assert preds.shape == (10, 3)
    assert list(preds.columns) == ["target_0", "target_1", "target_2"]

def test_linear_elasticnet_singleoutput(config):
    generator = np.random.default_rng(config.runtime.seed)
    X = pd.DataFrame(generator.random((10, 5)), columns=[f"f{i}" for i in range(5)])
    y = pd.Series(generator.random(10))

    model = LinearElasticNet(horizon=1, multioutput=False)
    model.fit(X, y)

    preds = model.predict(X)

    assert isinstance(preds, pd.Series)
    assert preds.shape == (10,)

def test_root(client):
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"ok": True}

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
def test_news_history_integration(client, monkeypatch):
    key = os.getenv("NEWS_API_KEY") or "placeholder"
    monkeypatch.setenv("NEWS_API_KEY", key)
    client.app.state.news_api_key = key

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

@pytest.mark.integration
def test_predict_raw_from_file():
    payload_path = Path(__file__).parent / "payload_predict.json"
    with open(payload_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    with TestClient(app) as client:
        response = client.post("/predict-raw?enrich=false&symbol=DJIA", json=payload)

        assert response.status_code == 200, response.text
        data = response.json()
        assert {"log_return", "current_price", "predicted_price"} <= data.keys()

def test_enrich_news_fills_missing_dates(monkeypatch):
    def fake_generate(symbol, dates, model="llama3", url_llm=None):
        return [{"date": d, "rank": "top1", "headline": f"{symbol} test headline for {d}"} for d in dates]

    monkeypatch.setattr("src.llm.generate_local_headlines", fake_generate)

    price_dates = ["2024-08-01", "2024-08-02", "2024-08-03"]
    real_news = [{"date": "2024-08-01", "rank": "top1", "headline": "Real news"}]

    enriched = enrich_news_with_generated(price_dates, real_news, "AAPL", "url_llm", "llama3")

    assert len(enriched) == 3
    dates = [r["date"] for r in enriched]
    assert "2024-08-01" in dates and "2024-08-02" in dates and "2024-08-03" in dates

def test_get_preprocessor_returns_pipeline_and_features():
    df = pd.DataFrame({
        "log_return": [0.01, 0.02, None],
        "rsi": [55, 60, 58],
        "quarter": [1, 2, 1],
        "dow": [0, 1, 2],
        "date": pd.date_range("2024-01-01", periods=3),
        "target": [0.01, 0.02, 0.03]
    })

    pipeline, features = get_preprocessor(df)

    assert isinstance(pipeline, Pipeline)
    assert "pre" in pipeline.named_steps
    assert set(features) == {"log_return", "rsi", "quarter", "dow"}

def test_aggregate_daily_returns_expected_columns():
    df = pd.DataFrame({
        "date": ["2025-01-01", "2025-01-01", "2025-01-02"],
        "headline": ["A", "B", "C"],
        "pos": [0.8, 0.7, 0.1],
        "neu": [0.1, 0.1, 0.1],
        "neg": [0.1, 0.2, 0.9],
        "pos_minus_neg": [0.7, 0.5, -0.8],
        "emb_0": [0.1, 0.2, 0.3],
        "emb_1": [0.4, 0.5, 0.6]
    })

    daily = FinBERT.aggregate_daily(df)

    assert "headline_count" in daily.columns
    assert "pos" in daily.columns
    assert daily.shape[0] == 2

from src.utils import set_seed


def test_set_seed_returns_deterministic_rng():
    rng1 = set_seed(123)
    a = rng1.integers(0, 100)

    rng2 = set_seed(123)
    b = rng2.integers(0, 100)

    assert a == b
