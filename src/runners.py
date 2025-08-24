from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import optuna
import pandas as pd

from src.data import time_series_split
from src.models.factory import Experiment
from src.preprocessing import get_preprocessor
from src.train import ModelTrainer


def run_experiments(
        df: pd.DataFrame,
        out_dir: Path,
        experiments: List[Experiment],
        forecast_horizon: int = 30,
        random_state: int = 42,
        n_trials: int = 30
) -> List[Dict]:
    results = []
    for exp in experiments:
        res = run(
            df_full=df,
            exp=exp,
            out_dir=str(out_dir),
            forecast_horizon=forecast_horizon,
            random_state=random_state,
            n_trials=n_trials
        )
        results.append(res)
    return results

def run(
        df_full: pd.DataFrame,
        exp: Experiment,
        out_dir: str,
        forecast_horizon: int = 30,
        random_state: int = 42,
        n_trials: int = 30
):
    """
    Generic training/tuning/eval runner.
    """
    # Split and Features
    train, val, test, _ = time_series_split(df_full, train_ratio=0.8, val_ratio=0.1, horizon=forecast_horizon)

    drop_cols = ["open", "high", "low", "close", "volume", "adj_close"]
    target_cols = [c for c in df_full.columns if c == "target" or c.startswith("target_")]
    feature_cols = [c for c in df_full.columns if c not in target_cols + ["date"] + drop_cols]

    if not exp.include_sentiment:
        sent = {"pos", "neg", "neu", "pos_minus_neg", "headline_count", "headline", "title"}
        sent |= {c for c in df_full.columns if c.startswith("emb_")}
        feature_cols = [c for c in feature_cols if c not in sent]

    X_train, y_train = train[feature_cols], train[target_cols]
    X_val, y_val = val[feature_cols], val[target_cols]
    X_test, y_test = test[feature_cols], test[target_cols]

    exp_name = exp.name
    X_test.to_parquet(Path(out_dir) / f"X_test_{exp_name}.parquet", index=False)

    # Preprocessor
    preprocessor, _ = get_preprocessor(X_train)
    joblib.dump(preprocessor, Path(out_dir) / f"preprocessor_{exp_name}.joblib")

    # Config
    model_config = {"optimization_metric": "rmse", "gap": 0, "seed": random_state}

    # Base Model and Trainer
    base_model = exp.build(forecast_horizon, random_state)
    trainer = ModelTrainer(
        model=base_model,
        name=f"{exp_name}",
        config=model_config,
        preprocessor=preprocessor,
        y_scale=True
    )

    # Optuna (use SQLite storage so the Study itself is reusable)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
        pruner=optuna.pruners.SuccessiveHalvingPruner(min_resource=1, reduction_factor=3)
    )
    study.optimize(lambda tr: trainer.objective(tr, X_train, y_train, n_splits=3), n_trials=n_trials, timeout=1200)

    best_params = study.best_trial.user_attrs.get("best_params", {}) or {}
    best_params.setdefault("random_state", random_state)

    # Rebuild with best params and train on train and val
    best_model = base_model.__class__(**{**base_model.get_params(), **best_params})
    trainer = ModelTrainer(
        best_model,
        name=f"{exp_name}",
        config=model_config,
        preprocessor=preprocessor,
        y_scale=True
    )
    trainer.fit(X_train, y_train, X_val, y_val)

    # Predictions
    y_pred_test = np.asarray(trainer.predict(X_test))
    y_pred_last = y_pred_test[-1].ravel()

    np.savez_compressed(Path(out_dir) / f"preds_{exp_name}.npz", y_pred_test=y_pred_test, y_pred_last=y_pred_last)
    np.save(Path(out_dir) / f"test_index_{exp_name}.npy", X_test.index.to_numpy())

    # Metrics
    metrics_test = trainer.evaluate(X_test, y_test)
    metrics_path = Path(out_dir) / f"metrics_test_{exp_name}.csv"
    pd.DataFrame(
        [{"name": exp_name, **{k: float(v) for k, v in metrics_test.items()}}]
    ).to_csv(metrics_path, index=False)

    # Save artifacts
    params_path = Path(out_dir) / f"best_params_{exp_name}.csv"
    pd.Series(best_params).to_csv(params_path)
    model_path = trainer.save()

    return {
        "kind": exp_name,
        "study": study,
        "horizon": forecast_horizon,
        "include_sentiment": exp.include_sentiment,
        "best_params": best_params,
        "metrics": {"test": metrics_test},
        "trainer": trainer,
        "paths": {"model": str(model_path), "params_csv": str(params_path), "metrics_csv": str(metrics_path)},
        "test_index": X_test.index.to_numpy(),
        "y_pred_test": y_pred_test,
        "y_pred_last": y_pred_last
    }
