from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.pipeline import Pipeline

from app.api.main import app
from config.config import Config
from src.data import _rename_columns, time_series_split
from src.evaluation import SHAPExplainer
from src.features import create_features_and_target, generate_full_feature_row
from src.llm import enrich_news_with_generated
from src.models.factory import Experiment
from src.models.linreg import LinearElasticNet
from src.preprocessing import get_preprocessor
from src.scaler import SafeStandardScaler
from src.sentiment import FinBERT
from src.train import ModelTrainer
from src.utils import set_seed


# === Constants ===
MAX_EMB_DIMS: int = 17
IS_CLOSE_ATOL: float = 1e-9

# === Helpers ===
def bdays(start: str, n: int) -> pd.DatetimeIndex:
    """Generate n business days starting from date string."""
    return pd.date_range(start, periods=n, freq="B")

BUSINESS_DATES_60 = bdays("2024-01-01", 60)
BUSINESS_DATES_40 = bdays("2024-01-01", 40)

def is_close(a: float, b: float = 0.0, atol: float = IS_CLOSE_ATOL) -> bool:
    """Check approximate equality with absolute tolerance."""
    return np.isclose(float(a), float(b), atol=atol)

def mk_price_df(dates: pd.DatetimeIndex, start: float = 100.0, seed: int = 42) -> pd.DataFrame:
    """Create a synthetic OHLCV price DataFrame with random walk adj_close."""
    rng = np.random.default_rng(seed)
    n = len(dates)

    rets = rng.normal(loc=0.001, scale=0.02, size=n)
    close = start * np.exp(np.cumsum(rets))
    open_ = np.r_[close[0], close[:-1]]
    wiggle = rng.uniform(0.001, 0.01, size=n)
    high = np.maximum(open_, close) * (1 + wiggle)
    low = np.minimum(open_, close) * (1 - wiggle)
    volu = (np.abs(close - open_) / close * 1e6).astype(int)

    return pd.DataFrame(
        {
            "date": dates,
            "adj_close": close,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volu,
        }
    )

def mk_news(dates: Iterable[pd.Timestamp], text: str = "headline") -> list[dict[str, str]]:
    """Generate dummy news headlines for given dates."""
    return [{"date": pd.to_datetime(d).strftime("%Y-%m-%d"), "headline": text} for d in dates]

def df_to_payload(df: pd.DataFrame) -> list[dict[str, str]]:
    """Convert a DataFrame into JSON payload with isoformatted dates."""
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    return out.to_dict(orient="records")  # type: ignore[return-value]

def init_finbert(config: Config) -> tuple[object, Pipeline, FinBERT]:
    """Initialize FinBERT sentiment model and load baseline linear model + preprocessor."""
    sentiment_model = FinBERT(config, device="cpu", max_embedding_dims=MAX_EMB_DIMS)
    model_path = Path(config.data.models_dir) / "linreg.pkl"
    model, pre, _, _ = ModelTrainer.load(str(model_path))
    return model, pre, sentiment_model

# === Fixtures ===
@pytest.fixture(scope="session")
def client() -> TestClient:
    return TestClient(app)

@pytest.fixture(scope="session")
def config() -> Config:
    return Config(Path("config/config.yaml"))

@pytest.fixture
def rng(config: Config) -> np.random.Generator:
    return np.random.default_rng(config.runtime.seed)

# === Config Tests ===
def test_set_seed_returns_deterministic_rng():
    rng1 = set_seed(123)
    rng2 = set_seed(123)
    assert rng1.integers(0, 100) == rng2.integers(0, 100)

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
        cfg = Config(tmp_path)
        assert cfg.general.name == "simple-test"
        assert isinstance(cfg.data.raw_dir, Path) and cfg.data.raw_dir.is_absolute()
        assert isinstance(cfg.model.path, Path) and cfg.model.path.is_absolute()
        assert isinstance(cfg.urls.api, str)
    finally:
        tmp_path.unlink()

# === Data Preprocessing Tests ===
def test_rename_columns():
    df = pd.DataFrame(columns=["Open", "Adj Close ", " Volume"])
    df_renamed = _rename_columns(df)
    assert list(df_renamed.columns) == ["open", "adj_close", "volume"]

def test_time_series_split_with_horizon_tail():
    n, H = 100, 10
    df = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=n, freq="D"),
        "adj_close": range(n),
        "open": 0.0, "high": 0.0, "low": 0.0, "close": 0.0, "volume": 0.0
    })
    df = create_features_and_target(df, forecast_horizon=H)
    train, val, test, future = time_series_split(df, train_ratio=0.7, val_ratio=0.2, horizon=H)

    n_feat = len(df)

    # assert sizes add up
    assert len(train) + len(val) + len(test) + len(future) == n_feat
    assert len(future) == H

    # assert chronological ordering is preserved
    assert train["date"].min() < train["date"].max() <= val["date"].min() <= val["date"].max() <= test["date"].max()

    # assert future tail starts right after last test row (positional continuity)
    effective_n = n_feat - H
    assert test.index.max() == effective_n - 1
    assert future.index.min() == effective_n

    # assert first future date is the day after the last test date
    assert future["date"].iloc[0] == df["date"].iloc[effective_n]

def test_time_series_split_no_overlap():
    n, H = 50, 5
    df = pd.DataFrame({
        "date": pd.date_range("2021-01-01", periods=n, freq="D"),
        "adj_close": np.arange(n, dtype=float),
        "open": 0.0, "high": 0.0, "low": 0.0, "close": 0.0, "volume": 0.0,
    })
    df = create_features_and_target(df, forecast_horizon=H)
    _, _, test, future = time_series_split(df, train_ratio=0.6, val_ratio=0.2, horizon=H)

    assert test.index.max() < future.index.min()
    assert test["date"].max() < future["date"].min()

# === Feature Engineering Tests ===
def test_create_features_and_target_minimal():
    df = pd.DataFrame({
        "date": BUSINESS_DATES_60,
        "adj_close": np.linspace(100, 150, 60),
        "open": 0.0, "high": 0.0, "low": 0.0, "close": 0.0, "volume": 0.0
    })
    features = create_features_and_target(df, forecast_horizon=3)
    assert {"target_1", "log_return"} <= set(features.columns)

def test_get_preprocessor_returns_pipeline_and_features():
    df = pd.DataFrame({
        "log_return": [0.01, 0.02, None],
        "dow": [0, 1, 2],
        "date": pd.date_range("2024-01-01", periods=3)
    })
    pipeline, features = get_preprocessor(df, "linreg")
    assert isinstance(pipeline, Pipeline)
    assert "pre" in pipeline.named_steps
    assert set(features) == {"log_return", "dow"}

def test_generate_full_feature_row_no_sentiment():
    df = mk_price_df(BUSINESS_DATES_40)
    row = generate_full_feature_row(df, None, None, forecast_horizon=5)
    assert isinstance(row, pd.DataFrame) and row.shape[0] == 1

def test_generate_full_feature_row_pad_neutral_last_day_zero(config: Config):
    price_df = mk_price_df(BUSINESS_DATES_40)

    # News on a day that is NOT the last, last day should be neutral if padded
    news_df = pd.DataFrame([{"date": price_df["date"].iloc[0], "headline": "some headline"}])
    _, _, sentiment_model = init_finbert(config)
    row = generate_full_feature_row(
        price_df, news_df, sentiment_model,
        forecast_horizon=1, max_embedding_dims=MAX_EMB_DIMS, fill_missing_neutral=True
    )
    assert is_close(row["pos_minus_neg"].values[0], 0.0)
    assert "emb_0" in row.columns and is_close(row["emb_0"].values[0], 0.0)

# === Sentiment Tests ===
@pytest.mark.slow
def test_sentiment_affects_feature_row(config: Config):
    df_price = mk_price_df(BUSINESS_DATES_60)
    last = df_price["date"].iloc[-1].strftime("%Y-%m-%d")

    df_pos = pd.DataFrame([{"date": last, "headline": "great earnings results and upbeat guidance"}])
    df_neg = pd.DataFrame([{"date": last, "headline": "lawsuit, accounting probe, and missed targets"}])

    model = FinBERT(config, device="cpu", max_embedding_dims=MAX_EMB_DIMS)
    row_pos = generate_full_feature_row(df_price, df_pos, model, forecast_horizon=1, max_embedding_dims=MAX_EMB_DIMS)
    row_neg = generate_full_feature_row(df_price, df_neg, model, forecast_horizon=1, max_embedding_dims=MAX_EMB_DIMS)

    # Expect different sentiment on the last day
    assert not np.allclose(row_pos["pos_minus_neg"], row_neg["pos_minus_neg"])

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
    assert {"headline_count", "pos"} <= set(daily.columns)

def test_enrich_news_fills_missing_dates(monkeypatch: pytest.MonkeyPatch):
    def fake_generate(
            symbol: str,
            dates: list[str],
            url: str,
            model: str = "llama3",
            seed_examples: list[str] | None = None
    ) -> list[dict[str, str]]:
        return [{"date": d, "headline": f"{symbol} test headline for {d}"} for d in dates]

    monkeypatch.setattr("src.llm.generate_local_headlines", fake_generate)
    dates = ["2024-08-01", "2024-08-02", "2024-08-03"]
    real_news = [{"date": "2024-08-01", "headline": "Real news"}]
    enriched = enrich_news_with_generated(dates, real_news, "AAPL", "url", "llama3")
    assert len(enriched) == 3

def test_enrich_requires_seed_raises():
    with pytest.raises(ValueError):
        enrich_news_with_generated(["2024-08-01"], [], "AAPL", "url", "llama3")

@pytest.mark.slow
def test_finbert_caching_effectiveness(tmp_path: Path, config: Config):
    cache_dir = getattr(config.runtime, "cache_dir", None)
    if cache_dir:
        shutil.rmtree(cache_dir, ignore_errors=True)

    config.runtime.cache_dir = tmp_path / "finbert"
    config.runtime.cache_dir.mkdir(parents=True, exist_ok=True)

    sentiment_model = FinBERT(config, device="cpu", max_embedding_dims=MAX_EMB_DIMS)
    df = pd.DataFrame([{"date": "2025-01-01", "headline": "Apple stock jumps after record earnings report"}])

    t0 = time.time()
    _ = sentiment_model.transform(df)
    t1 = time.time()
    result_cached = sentiment_model.transform(df)
    t2 = time.time()

    assert (t2 - t1) < (t1 - t0)
    assert {"pos_minus_neg", "emb_0"} <= set(result_cached.columns)
    emb_cols = [c for c in result_cached.columns if c.startswith("emb_")]
    assert len(emb_cols) <= MAX_EMB_DIMS

# === Models & Training Tests ===
@pytest.mark.parametrize("h,multi", [(3, True), (1, False)])
def test_linear_elasticnet_predictions_shape(rng: np.random.Generator, h: int, multi: bool):
    X = pd.DataFrame(rng.random((10, 5)), columns=[f"f{i}" for i in range(5)])
    y = pd.DataFrame(rng.random((10, h))) if multi else pd.Series(rng.random(10))
    model = LinearElasticNet(horizon=h, multioutput=multi).fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (10, h) if multi else preds.shape == (10,)

def test_model_trainer_fit_and_evaluate(rng: np.random.Generator):
    X = pd.DataFrame(rng.random((30, 5)), columns=[f"x{i}" for i in range(5)])
    y = pd.DataFrame(rng.random((30, 3)), columns=["target_0", "target_1", "target_2"])

    linreg_exp = Experiment(
        name="linreg",
        build=lambda horizon, seed: LinearElasticNet(horizon=horizon, random_state=seed, multioutput=True),
        include_sentiment=True
    )
    model = linreg_exp.build(3, 7)

    trainer = ModelTrainer(model=model, name="test_model", config={"optimization_metric": "rmse"})
    trainer.fit(X, y)
    results = trainer.evaluate(X, y)
    assert results["rmse"] > 0.0

@pytest.mark.parametrize("array_shape", [(3,), (3, 2)])
def test_safe_scaler_roundtrip(array_shape: tuple[int, ...]):
    s = SafeStandardScaler()
    y = np.array([1.0, 2.0, 3.0]) if len(array_shape) == 1 else np.array([[1.0, 0.1], [2.0, 0.2], [3.0, 0.3]])
    ys = s.fit_transform(y)
    y_back = s.inverse_transform(ys)
    assert y_back.shape == y.shape
    assert np.allclose(y_back, y)

# === Prediction Tests ===
def test_prediction_changes_with_different_prices(config: Config):
    price_df1 = mk_price_df(BUSINESS_DATES_60, 100, 150)
    price_df2 = mk_price_df(BUSINESS_DATES_60, 120, 170)

    model, pre, sentiment_model = init_finbert(config)

    X1 = pre.transform(generate_full_feature_row(
        price_df1, pd.DataFrame(), sentiment_model, forecast_horizon=30, max_embedding_dims=MAX_EMB_DIMS)
    )
    X2 = pre.transform(generate_full_feature_row(
        price_df2, pd.DataFrame(), sentiment_model, forecast_horizon=30, max_embedding_dims=MAX_EMB_DIMS)
    )

    preds1 = model.predict(X1)
    preds2 = model.predict(X2)
    assert not np.allclose(preds1, preds2)

def test_deterministic_prediction_with_seed(config: Config):
    set_seed(42)
    price_df = mk_price_df(BUSINESS_DATES_60)
    model, pre, sentiment_model = init_finbert(config)

    X = pre.transform(generate_full_feature_row(
        price_df, pd.DataFrame(), sentiment_model, forecast_horizon=30, max_embedding_dims=MAX_EMB_DIMS)
    )

    preds1 = model.predict(X)
    set_seed(42)
    preds2 = model.predict(X)
    assert np.allclose(preds1, preds2)

# === Evaluation (SHAP) Tests ===
def test_shap_explainer_outputs_values(rng: np.random.Generator):
    X = pd.DataFrame(rng.random((10, 4)), columns=[f"f{i}" for i in range(4)])
    y = pd.Series(rng.random(10))
    model = LinearElasticNet(horizon=1, multioutput=False).fit(X, y)
    pre, _ = get_preprocessor(X, "linreg")
    pre.fit(X)
    explainer = SHAPExplainer(model, pre, X, "linear")
    shap_vals = explainer.explain(X)
    assert isinstance(shap_vals, (np.ndarray, list))

# === API Integration Tests ===
def test_root(client: TestClient):
    res = client.get("/healthz")
    assert res.status_code == 200
    assert res.json() == {"ok": True}

def test_symbol_regex():
    pattern = re.compile(r"[A-Za-z0-9_.^-]+")
    assert pattern.fullmatch("AAPL")
    assert pattern.fullmatch("^DJI")
    assert not pattern.fullmatch("AAP L")

@pytest.mark.integration
def test_price_history(client: TestClient):
    res = client.get("/price-history", params={"symbol": "^DJI", "end_date": "2025-08-01", "days": 10})
    assert res.status_code == 200
    assert isinstance(res.json().get("price"), list)

@pytest.mark.integration
def test_news_history_integration(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("NEWS_API_KEY", os.getenv("NEWS_API_KEY") or "placeholder")
    client.app.state.news_api_key = os.getenv("NEWS_API_KEY")
    res = client.get(
        "/news-history", params={"query": "Apple", "end_date": datetime.today().strftime("%Y-%m-%d"), "days": 7}
    )
    assert res.status_code == 200
    assert "news" in res.json()

@pytest.mark.integration
def test_predict_raw_from_file():
    payload_path = Path(__file__).parent / "payload_predict.json"
    with open(payload_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    with TestClient(app) as c:
        r = c.post("/predict-raw?enrich=false&symbol=DJIA", json=payload)
        assert r.status_code == 200, r.text
        data = r.json()
        assert {"delta_price", "current_price", "predicted_price"} <= data.keys()

@pytest.mark.integration
def test_predict_raw_enrich_requires_seed(client: TestClient):
    price = mk_price_df(pd.date_range("2024-06-03", periods=10, freq="B"))
    payload = {"price": df_to_payload(price), "news": []}
    res = client.post("/predict-raw", params={"enrich": "true", "symbol": "^DJI"}, json=payload)
    assert res.status_code == 422

@pytest.mark.integration
def test_predict_raw_pad_requires_two(client: TestClient):
    price = mk_price_df(pd.date_range("2024-06-03", periods=10, freq="B"))
    news = [{"date": price["date"].iloc[0].strftime("%Y-%m-%d"), "headline": "seed only one"}]
    payload = {"price": df_to_payload(price), "news": news}
    res = client.post("/predict-raw", params={"pad_neutral": "true", "symbol": "^DJI"}, json=payload)
    assert res.status_code == 422

@pytest.mark.integration
def test_predict_raw_ignore_news_ok(client: TestClient):
    price = mk_price_df(pd.date_range("2024-06-03", periods=10, freq="B"))
    news = mk_news(price["date"].iloc[:2], "whatever")
    payload = {"price": df_to_payload(price), "news": news}
    res = client.post("/predict-raw", params={"ignore_news": "true", "symbol": "^DJI"}, json=payload)
    assert res.status_code == 200
