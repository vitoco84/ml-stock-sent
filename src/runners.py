from pathlib import Path

import numpy as np
import optuna
import pandas as pd

from src.data import time_series_split
from src.models.factory import build_model
from src.preprocessing import get_preprocessor
from src.train import ModelTrainer


def _sentiment_cols(df: pd.DataFrame) -> list[str]:
    base = ["pos", "neg", "neu", "pos_minus_neg"]
    emb = [c for c in df.columns if c.startswith("emb_")]
    return [c for c in base if c in df.columns] + emb

def _select_cols(df_full: pd.DataFrame, include_sentiment: bool):
    drop_cols = ["open", "high", "low", "close", "volume", "adj_close"]
    target_cols = [c for c in df_full.columns if c == "target" or c.startswith("target_")]
    feat_cols = [c for c in df_full.columns if c not in target_cols + ["date"] + drop_cols]
    if not include_sentiment:
        feat_cols = [c for c in feat_cols if c not in _sentiment_cols(df_full)]
    return feat_cols, target_cols

def _returns_to_prices(last_price: float, log_returns: np.ndarray) -> np.ndarray:
    cum = np.cumsum(np.asarray(log_returns, dtype=float))
    return last_price * np.exp(cum)

def run_experiment(
        kind: str,
        df_full: pd.DataFrame,
        include_sentiment: bool,
        out_dir: str,
        forecast_horizon: int = 30,
        random_state: int = 42,
        n_trials: int = 30
):
    """Train/validate/test with Optuna tuning. kind ∈ {'linreg','xgboost'}."""
    # Split and Features
    train, val, test, _ = time_series_split(df_full, train_ratio=0.8, val_ratio=0.1, horizon=forecast_horizon)

    drop_cols = ["open", "high", "low", "close", "volume", "adj_close"]
    target_cols = [c for c in df_full.columns if c == "target" or c.startswith("target_")]
    feature_cols = [c for c in df_full.columns if c not in target_cols + ["date"] + drop_cols]

    if not include_sentiment:
        sent = ["pos", "neg", "neu", "pos_minus_neg"] + [c for c in df_full.columns if c.startswith("emb_")]
        feature_cols = [c for c in feature_cols if c not in sent]

    X_train, y_train = train[feature_cols], train[target_cols]
    X_val, y_val = val[feature_cols], val[target_cols]
    X_test, y_test = test[feature_cols], test[target_cols]

    # Preprocessor
    preprocessor, _ = get_preprocessor(X_train)

    # Base Model and Trainer
    # DirectMultiStep auto-applied when horizon > 1
    base_model = build_model(
        kind,
        horizon=forecast_horizon,
        random_state=random_state,
        multioutput=True if kind == "linreg" else False,
        tree_method="hist" if kind == "xgboost" else None,
        n_jobs=-1 if kind == "xgboost" else None,
    )

    trainer = ModelTrainer(
        model=base_model,
        name=f"{kind}_h{forecast_horizon}",
        config={
            "optimization_metric": "rmse",
            "gap": 0,
            "seed": random_state,
        },
        preprocessor=preprocessor,
        y_scale=True
    )

    # Optuna
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
        pruner=optuna.pruners.SuccessiveHalvingPruner(min_resource=1, reduction_factor=3),
        study_name=f"{kind}_{'with' if include_sentiment else 'wo'}_sent_h{forecast_horizon}"
    )
    study.optimize(lambda tr: trainer.objective(tr, X_train, y_train, n_splits=3), n_trials=n_trials, timeout=1200)

    best_params = study.best_trial.user_attrs.get("best_params", {}) or {}
    best_params.setdefault("random_state", random_state)

    # Rebuild with best params and train on train and val
    best_model = build_model(kind, horizon=forecast_horizon, **best_params)
    trainer = ModelTrainer(
        best_model,
        name=f"{kind}_h{forecast_horizon}",
        config={
            "optimization_metric": "rmse",
            "gap": 0,
            "seed": random_state
        },
        preprocessor=preprocessor,
        y_scale=True
    )
    trainer.fit(X_train, y_train, X_val, y_val)

    # Predictions
    y_pred_test = np.asarray(trainer.predict(X_test))
    y_true_test = np.asarray(y_test)
    y_pred_last = y_pred_test[-1].ravel()
    y_true_last = y_true_test[-1].ravel()

    # Metrics
    metrics_train = trainer.evaluate(X_train, y_train)
    metrics_val = trainer.evaluate(X_val, y_val)
    metrics_test = trainer.evaluate(X_test, y_test)

    # Save artifacts (minimal)
    params_path = Path(out_dir) / f"best_params_{kind}_h{forecast_horizon}.csv"
    pd.Series(best_params).to_csv(params_path)
    model_path = trainer.save()

    return {
        "kind": kind,
        "horizon": forecast_horizon,
        "include_sentiment": include_sentiment,

        "paths": {"model": str(model_path), "params_csv": str(params_path)},
        "best_params": best_params,
        "features": {"feature_cols": feature_cols, "target_cols": target_cols},
        "metrics": {"train": metrics_train, "val": metrics_val, "test": metrics_test},

        "trainer": trainer,
        "test_index": X_test.index,

        "y_pred_test": y_pred_test,
        "y_true_test": y_true_test,
        "y_pred_last": y_pred_last,
        "y_true_last": y_true_last
    }
