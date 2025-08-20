"""
Tests: simplified with *minimal* changes, keeping your original structure and names.
- Kept all original test function names & behavior
- Light consistency pass (tiny helpers/constants, clearer asserts)
- No endpoint/IO logic changes
"""
import json
import os
import shutil
import tempfile
import time
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
from src.evaluation import SHAPExplainer
from src.features import convert_log_return, create_features_and_target, generate_full_feature_row
from src.llm import enrich_news_with_generated
from src.models.classical import LinearElasticNet
from src.preprocessing import get_preprocessor
from src.recursive import RecursiveForecaster
from src.sentiment import FinBERT
from src.train import ModelTrainer
from src.utils import set_seed


# === Tiny helpers to avoid repetition (non-breaking) ===
BUSINESS_DATES_60 = pd.date_range("2024-01-01", periods=60, freq="B")
BUSINESS_DATES_40 = pd.date_range("2024-01-01", periods=40, freq="B")

def mk_price_df(dates: pd.DatetimeIndex, start: float = 100.0, stop: float = 150.0) -> pd.DataFrame:
    return pd.DataFrame({
        "date": dates,
        "adj_close": np.linspace(start, stop, len(dates)),
        "open": 0.0, "high": 0.0, "low": 0.0, "close": 0.0, "volume": 0.0
    })

def init_finbert(config):
    sentiment_model = FinBERT(config, device="cpu")
    model_path = Path(config.model.path_dir) / config.model.enet_mo_best_30
    model, pre, _, _ = ModelTrainer.load(str(model_path))
    return model, pre, sentiment_model

# === Fixtures ===

@pytest.fixture(scope="session")
def client():
    return TestClient(app)

@pytest.fixture(scope="session")
def config():
    return Config(Path("config/config.yaml"))

@pytest.fixture
def rng(config):
    return np.random.default_rng(config.runtime.seed)

# === Core Utils ===

def test_set_seed_returns_deterministic_rng():
    rng1 = set_seed(123)
    rng2 = set_seed(123)
    assert rng1.integers(0, 100) == rng2.integers(0, 100)

# === Config ===

def test_config_loads_yaml_and_resolves_paths():
    with tempfile.NamedTemporaryFile("w+", suffix=".yaml", delete=False) as tmp:
        tmp.write(
            """
            general:
              name: "simple-test"
            data:
              raw_dir: "../data/raw"
            model:
              path: "../data/models/model.pkl"
            urls:
              api: "http://localhost:8000"
            """
        )
        tmp.flush()
        tmp_path = Path(tmp.name)

    try:
        config = Config(tmp_path)
        assert config.general.name == "simple-test"
        assert isinstance(config.data.raw_dir, Path) and config.data.raw_dir.is_absolute()
        assert isinstance(config.model.path, Path) and config.model.path.is_absolute()
        assert isinstance(config.urls.api, str)
    finally:
        tmp_path.unlink()

# === Data Preprocessing ===

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

# === Feature Engineering ===

def test_create_features_and_target_minimal():
    df = pd.DataFrame({
        "date": BUSINESS_DATES_60,
        "adj_close": np.linspace(100, 150, 60)
    })
    features = create_features_and_target(df, forecast_horizon=3)
    assert "target_1" in features.columns
    assert "log_return" in features.columns

def test_convert_log_return_basic():
    price = 100.0
    log_return = np.log(1.10)
    assert abs(convert_log_return(price, log_return) - 110.0) < 0.01

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

def test_generate_full_feature_row_no_sentiment():
    df = mk_price_df(BUSINESS_DATES_40)
    row = generate_full_feature_row(df, None, None, horizon=5)
    assert isinstance(row, pd.DataFrame)
    assert row.shape[0] == 1

# === Sentiment ===

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

def test_enrich_news_fills_missing_dates(monkeypatch):
    def fake_generate(symbol, dates, model="llama3", url_llm=None):
        return [{"date": d, "rank": "top1", "headline": f"{symbol} test headline for {d}"} for d in dates]

    monkeypatch.setattr("src.llm.generate_local_headlines", fake_generate)
    dates = ["2024-08-01", "2024-08-02", "2024-08-03"]
    real_news = [{"date": "2024-08-01", "rank": "top1", "headline": "Real news"}]
    enriched = enrich_news_with_generated(dates, real_news, "AAPL", "url", "llama3")
    assert len(enriched) == 3

def test_sentiment_affects_feature_row(config):
    df_price = mk_price_df(BUSINESS_DATES_60)

    df_pos = pd.DataFrame([{"date": "2024-03-01", "rank": "1", "headline": "great earnings results"}])
    df_neg = pd.DataFrame([{"date": "2024-03-01", "rank": "1", "headline": "lawsuit and fraud scandal"}])

    model = FinBERT(config, device="cpu")

    row_pos = generate_full_feature_row(df_price, df_pos, model, horizon=1)
    row_neg = generate_full_feature_row(df_price, df_neg, model, horizon=1)

    assert not np.allclose(
        row_pos["pos_minus_neg"], row_neg["pos_minus_neg"]
    ), "Positive vs negative news should affect sentiment features"

def test_finbert_caching_effectiveness(tmp_path, config):
    cache_dir = getattr(config.runtime, "cache_dir", None)
    if cache_dir:
        shutil.rmtree(cache_dir, ignore_errors=True)

    config.runtime.cache_dir = tmp_path / "finbert"
    config.runtime.cache_dir.mkdir(parents=True, exist_ok=True)

    sentiment_model = FinBERT(config, device="cpu")
    df = pd.DataFrame([{"date": "2025-01-01", "headline": "Apple stock jumps after record earnings report"}])

    # First run should take longer (no cache yet)
    start = time.time()
    _ = sentiment_model.transform(df)
    duration_first = time.time() - start

    # Second run should use cache and be faster
    start = time.time()
    result_cached = sentiment_model.transform(df)
    duration_cached = time.time() - start

    assert duration_cached < duration_first, "Cached run should be faster"
    assert "pos_minus_neg" in result_cached.columns
    assert "emb_0" in result_cached.columns

# === Models & Training ===

def test_linear_elasticnet_multioutput(rng):
    X = pd.DataFrame(rng.random((10, 5)), columns=[f"f{i}" for i in range(5)])
    y = pd.DataFrame(rng.random((10, 3)))
    model = LinearElasticNet(horizon=3, multioutput=True).fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (10, 3)

def test_linear_elasticnet_singleoutput(rng):
    X = pd.DataFrame(rng.random((10, 5)), columns=[f"f{i}" for i in range(5)])
    y = pd.Series(rng.random(10))
    model = LinearElasticNet(horizon=1, multioutput=False).fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (10,)

def test_model_trainer_fit_and_evaluate(rng):
    X = pd.DataFrame(rng.random((30, 5)), columns=[f"x{i}" for i in range(5)])
    y = pd.DataFrame(rng.random((30, 3)), columns=["target_0", "target_1", "target_2"])
    model = LinearElasticNet(horizon=3, multioutput=True)
    trainer = ModelTrainer(model=model, name="test_model", config={"optimization_metric": "rmse"})
    trainer.fit(X, y)
    results = trainer.evaluate(X, y)
    assert results["rmse"] > 0.0

# === Prediction ===

def test_prediction_changes_with_different_prices(config):
    price_df1 = mk_price_df(BUSINESS_DATES_60, 100, 150)
    price_df2 = mk_price_df(BUSINESS_DATES_60, 120, 170)  # same shape, different values

    model, pre, sentiment_model = init_finbert(config)

    X1 = pre.transform(generate_full_feature_row(price_df1, pd.DataFrame(), sentiment_model, horizon=30))
    X2 = pre.transform(generate_full_feature_row(price_df2, pd.DataFrame(), sentiment_model, horizon=30))

    preds1 = model.predict(X1)
    preds2 = model.predict(X2)

    assert not np.allclose(preds1, preds2), "Predictions should differ with different price input"

def test_deterministic_prediction_with_seed(config):
    set_seed(42)

    price_df = mk_price_df(BUSINESS_DATES_60)

    model, pre, sentiment_model = init_finbert(config)

    X = pre.transform(generate_full_feature_row(price_df, pd.DataFrame(), sentiment_model, horizon=30))

    preds1 = model.predict(X)
    set_seed(42)
    preds2 = model.predict(X)

    assert np.allclose(preds1, preds2), "Predictions with same seed should match"

# === Forecasting ===

def test_recursive_forecaster_forecasts_with_dummy_data():
    df = mk_price_df(BUSINESS_DATES_60)
    feat = create_features_and_target(df).dropna(subset=["target"])
    pre, _ = get_preprocessor(feat.drop(columns=["target"]))
    model = LinearElasticNet(horizon=1, multioutput=False).fit(pre.fit_transform(feat.drop(columns=["target"])),
                                                               feat["target"])
    forecaster = RecursiveForecaster(model=model, preprocessor=pre, sentiment_model=None)
    forecast = forecaster.forecast(df, news_df=None, horizon=5)
    assert forecast.shape == (5,)

# === Evaluation ===

def test_shap_explainer_outputs_values(rng):
    X = pd.DataFrame(rng.random((10, 4)), columns=[f"f{i}" for i in range(4)])
    y = pd.Series(rng.random(10))
    model = LinearElasticNet(horizon=1, multioutput=False).fit(X, y)
    pre, _ = get_preprocessor(X)
    pre.fit(X)
    explainer = SHAPExplainer(model, pre, X)
    shap_vals = explainer.explain(X)
    assert isinstance(shap_vals, (np.ndarray, list))

# === API Integration ===

def test_root(client):
    res = client.get("/healthz")
    assert res.status_code == 200
    assert res.json() == {"ok": True}

@pytest.mark.integration
def test_price_history(client):
    res = client.get("/price-history", params={"symbol": "^DJI", "end_date": "2025-08-01", "days": 10})
    assert res.status_code == 200
    assert isinstance(res.json().get("price"), list)

@pytest.mark.integration
def test_news_history_integration(client, monkeypatch):
    monkeypatch.setenv("NEWS_API_KEY", os.getenv("NEWS_API_KEY") or "placeholder")
    client.app.state.news_api_key = os.getenv("NEWS_API_KEY")
    res = client.get("/news-history",
                     params={"query": "Apple", "end_date": datetime.today().strftime("%Y-%m-%d"), "days": 7})
    assert res.status_code == 200
    assert "news" in res.json()

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

def test_exception_and_logger():
    df = load_prices_sentiment("foobar")
    assert df.empty
