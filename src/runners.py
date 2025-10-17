from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from src.data import time_series_split
from src.metrics import metrics
from src.models.factory import Experiment
from src.preprocessing import get_preprocessor
from src.train import ModelTrainer
from src.utils import set_seed


def sign_strategy_returns(y_true: pd.DataFrame, y_pred: np.ndarray) -> pd.DataFrame:
    """
    Compute cumulative returns for a simple sign-based strategy.
    - Long if prediction > 0, short if prediction < 0.
    - Compare vs buy&hold and riskless asset.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    signal = np.sign(y_pred)
    strat_returns = signal * y_true

    df = pd.DataFrame({
        "strategy": np.cumsum(strat_returns),
        "buy_hold": np.cumsum(y_true),
        "riskless": np.zeros_like(y_true),
    })
    return df

def run_experiments(
        df: pd.DataFrame,
        out_dir: Path,
        experiments: list[Experiment],
        forecast_horizon: int,
        random_state: int,
        n_trials: int,
        n_splits: int,
        gap: int,
        subsample_train: int,
        train_ratio: int,
        val_ratio: int,
        n_jobs: int,
        target_mode: str,
        save: bool = True
) -> list[dict[str, Any]]:
    """Run multiple experiments sequentially."""
    results: list[dict[str, Any]] = []
    for exp in experiments:
        results.append(
            _run(
                df_full=df,
                exp=exp,
                out_dir=str(out_dir),
                forecast_horizon=forecast_horizon,
                random_state=random_state,
                n_trials=n_trials,
                n_splits=n_splits,
                gap=gap,
                subsample_train=subsample_train,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                n_jobs=n_jobs,
                target_mode=target_mode,
                save=save
            )
        )
    return results

def _run(
        df_full: pd.DataFrame,
        exp: Experiment,
        out_dir: str,
        forecast_horizon: int,
        random_state: int,
        n_trials: int,
        n_splits: int,
        gap: int,
        subsample_train: int,
        train_ratio: int,
        val_ratio: int,
        n_jobs: int,
        target_mode: str,
        save: bool = True
) -> dict[str, Any]:
    """
    Single experiment: split → preprocess → tune → retrain → evaluate → save artifacts.
    Targets are log returns. Evaluation includes MAE, MSE, RMSE, R^2, and directional accuracy.
    """
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)

    set_seed(random_state)
    gap = max(gap, forecast_horizon)

    # Data split and features
    train, val, test, forecast = time_series_split(
        df_full, train_ratio=train_ratio, val_ratio=val_ratio, horizon=forecast_horizon
    )

    drop_cols = ["open", "high", "low", "close", "volume", "adj_close"]
    if target_mode == "rolling":
        target_cols = [c for c in df_full.columns if c.startswith("target_")]
    else:
        target_cols = [c for c in df_full.columns if c == "target" or c.startswith("target_")]
    feature_cols = [c for c in df_full.columns if c not in target_cols + ["date"] + drop_cols]

    if exp.name.lower() in {"cnn", "lstm"}:
        feature_cols = [c for c in feature_cols if c.startswith("lag_")]
    else:
        feature_cols = [c for c in feature_cols if not c.startswith("lag_") and c != "log_return"]

    if not exp.include_sentiment:
        sent = {"pos", "neg", "neu", "pos_minus_neg", "headline_count", "headline", "title"}
        sent |= {c for c in df_full.columns if c.startswith("emb_")}
        feature_cols = [c for c in feature_cols if c not in sent]

    # Full data
    X_train, y_train = train[feature_cols], train[target_cols]
    X_val, y_val = val[feature_cols], val[target_cols]
    X_test, y_test = test[feature_cols], test[target_cols]
    X_forecast = forecast[feature_cols]

    # Subsample
    train_sub = (
        train.sample(min(len(train), subsample_train), random_state=random_state)
        if subsample_train else train
    )
    X_train_sub, y_train_sub = train_sub[feature_cols], train_sub[target_cols]

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if save:
        X_test.to_parquet(out_path / f"{exp.name}_X_test.parquet", index=False)

    # Preprocessor and Config
    preprocessor, _ = get_preprocessor(X_train_sub, exp.name)
    model_config = {"optimization_metric": "mae", "gap": gap, "seed": random_state}
    y_scale_flag = exp.name.lower() not in {"xgboost", "random_forest"}

    # Base Model and Trainer
    base_model = exp.build(forecast_horizon, random_state, n_jobs)

    if getattr(base_model, "input_mode", "tabular") == "sequence":
        assert any(c.startswith("lag_") for c in feature_cols), "CNN and LSTM require lag_* features."

    trainer = ModelTrainer(
        model=base_model,
        name=f"{exp.name}",
        config=model_config,
        preprocessor=preprocessor,
        y_scale=y_scale_flag
    )

    # Optuna Hyperparameter tuning on a supsample
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize" if model_config["optimization_metric"].lower() == "r2" else "minimize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    )
    walk_forward = True if exp.name in ["lstm", "cnn"] else False
    study.optimize(
        lambda tr: trainer.objective(tr, X_train_sub, y_train_sub, n_splits=n_splits, walk_forward=walk_forward),
        n_trials=n_trials
    )

    best_params = study.best_trial.user_attrs.get("best_params", {}) or {}
    best_params.setdefault("random_state", random_state)

    # Rebuild with best params and fit on train and val
    best_model = base_model.__class__(**{**base_model.get_params(), **best_params})
    trainer = ModelTrainer(
        best_model,
        name=f"{exp.name}",
        config=model_config,
        preprocessor=preprocessor,
        y_scale=y_scale_flag
    )
    trainer.fit(X_train, y_train, X_val, y_val)
    if save:
        joblib.dump(trainer.preprocessor, out_path / f"{exp.name}_preprocessor.joblib")

    # Predictions
    y_pred_val = np.asarray(trainer.predict(X_val))
    y_pred_test = np.asarray(trainer.predict(X_test))
    y_pred_last = np.asarray(trainer.predict(X_forecast.iloc[[0]])).ravel()

    np.savez_compressed(
        out_path / f"{exp.name}_preds.npz",
        y_pred_val=y_pred_val,
        y_pred_test=y_pred_test,
        y_pred_last=y_pred_last
    )
    if save:
        np.save(out_path / f"{exp.name}_test_index.npy", X_test.index.to_numpy())

    # Metrics
    metrics_test = trainer.evaluate(X_test, y_test)

    # Economic sanity check: sign-based strategy
    strat_df = sign_strategy_returns(y_test, y_pred_test)
    strat_cum = strat_df.iloc[-1].to_dict()

    # Baseline: predict last observed return at each horizon
    if target_mode == "rolling":
        y_pred_naive = np.zeros_like(y_test)
    else:
        last_return = np.asarray(y_train.iloc[-1])
        y_pred_naive = np.tile(last_return, (len(y_test), 1))

    baseline_metrics = metrics(np.asarray(y_test), y_pred_naive, y_insample=np.asarray(y_train))

    model_path = ""
    if save:
        model_path = trainer.save()

    result = {
        "kind": exp.name,
        "study": study,
        "horizon": forecast_horizon,
        "include_sentiment": exp.include_sentiment,
        "best_params": best_params,
        "metrics": {"test": metrics_test, "baseline": baseline_metrics},
        "strategy": strat_cum,
        "trainer": trainer,
        "paths": {
            "model": str(model_path),
            "params_csv": str(out_path / f"{exp.name}_best_params.csv"),
            "metrics_json": str(out_path / f"{exp.name}_metrics_test.json"),
            "baseline_metrics_json": str(out_path / f"{exp.name}_metrics_test_naive.json"),
            "preds_npz": str(out_path / f"{exp.name}_preds.npz"),
            "test_index_npy": str(out_path / f"{exp.name}_test_index.npy"),
            "features_csv": str(out_path / f"{exp.name}_features.csv"),
            "X_forecast_parquet": str(out_path / f"{exp.name}_X_forecast.parquet")
        },
        "test_index": X_test.index.to_numpy(),
        "y_pred_val": y_pred_val,
        "y_pred_test": y_pred_test,
        "y_pred_last": y_pred_last
    }

    result_light = {
        k: (str(v) if k in {"study", "trainer"} else v)
        for k, v in result.items()
        if k not in {"y_pred_val", "y_pred_test", "y_pred_last", "test_index"}
    }
    if save:
        with open(out_path / f"{exp.name}_result.json", "w") as f:
            json.dump(result_light, f, indent=2)

    return result
