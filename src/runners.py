from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import optuna
import pandas as pd

from src.data import time_series_split
from src.metrics import metrics
from src.models.factory import Experiment
from src.preprocessing import get_preprocessor
from src.train import ModelTrainer


def run_experiments(
        df: pd.DataFrame,
        out_dir: Path,
        experiments: List[Experiment],
        forecast_horizon: int = 30,
        random_state: int = 42,
        n_trials: int = 30,
        n_splits: int = 2,
        gap: int = 30
) -> List[Dict]:
    results = []
    for exp in experiments:
        res = _run(
            df_full=df,
            exp=exp,
            out_dir=str(out_dir),
            forecast_horizon=forecast_horizon,
            random_state=random_state,
            n_trials=n_trials,
            n_splits=n_splits,
            gap=gap
        )
        results.append(res)
    return results

def _run(
        df_full: pd.DataFrame,
        exp: Experiment,
        out_dir: str,
        forecast_horizon: int = 30,
        random_state: int = 42,
        n_trials: int = 30,
        n_splits: int = 2,
        gap: int = 30
):
    """Generic training/tuning/eval runner."""
    # Silence optuna experimental warnings
    warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)

    # Ensure gap is at least the horizon (avoid leakage in CV)
    gap = max(gap, forecast_horizon)

    # Split and Features
    train, val, test, forecast = time_series_split(df_full, train_ratio=0.8, val_ratio=0.1, horizon=forecast_horizon)

    drop_cols = ["open", "high", "low", "close", "volume", "adj_close"]
    target_cols = [c for c in df_full.columns if c == "target" or c.startswith("target_")]
    feature_cols = [c for c in df_full.columns if c not in target_cols + ["date"] + drop_cols]

    # Only sequence models consume lag_* (CNN/LSTM).
    if "cnn" not in exp.name.lower() and "lstm" not in exp.name.lower():
        feature_cols = [c for c in feature_cols if not c.startswith("lag_") and c != "log_return"]

    # Optional drop sentiment features
    if not exp.include_sentiment:
        sent = {"pos", "neg", "neu", "pos_minus_neg", "headline_count", "headline", "title"}
        sent |= {c for c in df_full.columns if c.startswith("emb_")}
        feature_cols = [c for c in feature_cols if c not in sent]

    X_train, y_train = train[feature_cols], train[target_cols]
    X_val, y_val = val[feature_cols], val[target_cols]
    X_test, y_test = test[feature_cols], test[target_cols]
    X_forecast = forecast[feature_cols]

    exp_name = exp.name
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"feature": feature_cols}).to_csv(Path(out_dir) / f"{exp_name}_features.csv", index=False)
    X_test.to_parquet(Path(out_dir) / f"{exp_name}_X_test.parquet", index=False)
    X_train.to_parquet(Path(out_dir) / f"{exp_name}_X_train.parquet", index=False)
    X_forecast.to_parquet(Path(out_dir) / f"{exp_name}_X_forecast.parquet", index=False)

    # Preprocessor
    preprocessor, _ = get_preprocessor(X_train, exp_name)

    # Config
    model_config = {"optimization_metric": "mae", "gap": gap, "seed": random_state}

    # Scale y for linear/NN models, tree models predict in original scale.
    y_scale_flag = exp_name.lower() not in {"random_forest", "xgboost"}

    # Base Model and Trainer
    base_model = exp.build(forecast_horizon, random_state)

    if getattr(base_model, "input_mode", "tabular") == "sequence":
        assert any(c.startswith("lag_") for c in feature_cols), "CNN and LSTM require lag_* features."

    # Training
    trainer = ModelTrainer(
        model=base_model,
        name=f"{exp_name}",
        config=model_config,
        preprocessor=preprocessor,
        y_scale=y_scale_flag
    )

    # Optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",  # minimize for mae/rmse/mse/smape; maximize for r2/accuracy
        sampler=optuna.samplers.TPESampler(
            seed=random_state,
            n_startup_trials=15,
            multivariate=True,
            group=True,
            constant_liar=True
        ),
        pruner=optuna.pruners.HyperbandPruner(min_resource=1, max_resource=n_splits, reduction_factor=3)
    )
    study.optimize(
        lambda tr: trainer.objective(tr, X_train, y_train, n_splits=n_splits),
        n_trials=n_trials,
        timeout=None,
        n_jobs=1
    )

    best_params = study.best_trial.user_attrs.get("best_params", {}) or {}
    best_params.setdefault("random_state", random_state)

    # Rebuild with best params and fit on train and val
    best_model = base_model.__class__(**{**base_model.get_params(), **best_params})
    trainer = ModelTrainer(
        best_model,
        name=f"{exp_name}",
        config=model_config,
        preprocessor=preprocessor,
        y_scale=y_scale_flag
    )
    trainer.fit(X_train, y_train, X_val, y_val)
    joblib.dump(trainer.preprocessor, Path(out_dir) / f"{exp_name}_preprocessor.joblib")

    # Predictions
    y_pred_val = np.asarray(trainer.predict(X_val))
    y_pred_test = np.asarray(trainer.predict(X_test))
    y_pred_last = np.asarray(trainer.predict(X_forecast.iloc[[0]])).ravel()

    np.savez_compressed(
        Path(out_dir) / f"{exp_name}_preds.npz",
        y_pred_val=y_pred_val,
        y_pred_test=y_pred_test,
        y_pred_last=y_pred_last
    )
    np.save(Path(out_dir) / f"{exp_name}_test_index.npy", X_test.index.to_numpy())

    # Metrics
    metrics_test = trainer.evaluate(X_test, y_test)
    metrics_path = Path(out_dir) / f"{exp_name}_metrics_test.csv"
    pd.DataFrame(
        [{"name": exp_name, **{k: float(v) for k, v in metrics_test.items()}}]
    ).to_csv(metrics_path, index=False)

    # Naive baseline: predict zeros (delta=0) for comparison
    y_test_np = np.asarray(y_test)
    y_train_np = np.asarray(y_train)
    y_pred_naive = np.zeros_like(y_test_np)

    baseline_metrics = metrics(y_test_np, y_pred_naive, y_insample=y_train_np)
    pd.DataFrame([{"name": "naive", **baseline_metrics}]).to_csv(
        Path(out_dir) / f"{exp_name}_metrics_test_naive.csv", index=False
    )

    # Save artifacts
    params_path = Path(out_dir) / f"{exp_name}_best_params.csv"
    pd.Series(best_params).to_csv(params_path)
    model_path = trainer.save()

    return {
        "kind": exp_name,
        "study": study,
        "horizon": forecast_horizon,
        "include_sentiment": exp.include_sentiment,
        "best_params": best_params,
        "metrics": {"test": metrics_test, "naive": baseline_metrics},
        "trainer": trainer,
        "paths": {"model": str(model_path), "params_csv": str(params_path), "metrics_csv": str(metrics_path)},
        "test_index": X_test.index.to_numpy(),
        "y_pred_val": y_pred_val,
        "y_pred_test": y_pred_test,
        "y_pred_last": y_pred_last
    }
