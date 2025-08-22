from pathlib import Path

import optuna
import pandas as pd

from src.data import time_series_split
from src.models.factory import build_model
from src.preprocessing import get_preprocessor
from src.train import ModelTrainer


def get_sentiment_cols(df: pd.DataFrame) -> list[str]:
    base = ["pos", "neg", "neu", "pos_minus_neg"]
    emb = [c for c in df.columns if c.startswith("emb_")]
    return [c for c in base if c in df.columns] + emb

def run_experiment(
        kind: str,
        df_full: pd.DataFrame,
        include_sentiment: bool,
        path_dir: str, *,
        horizon: int = 30,
        random_state: int = 42
):
    """
    kind e.g.: 'linreg' or 'xgboost'
    include_sentiment: True/False
    returns: (results_df, study, final_trainer, context_dict)
    """
    # Train, Val, Split
    train, val, test, forecast = time_series_split(df_full, train_ratio=0.8, val_ratio=0.1, horizon=horizon)

    # Features
    drop_cols = ["open", "high", "low", "close", "volume", "adj_close"]
    target_cols = [c for c in df_full.columns if c == "target" or c.startswith("target_")]
    feat_cols = [c for c in df_full.columns if c not in target_cols + ["date"] + drop_cols]
    sentiment_cols = get_sentiment_cols(df_full)

    if not include_sentiment:
        feat_cols = [c for c in feat_cols if c not in sentiment_cols]

    X_train, y_train = train[feat_cols], train[target_cols]
    X_val, y_val = val[feat_cols], val[target_cols]
    X_test, y_test = test[feat_cols], test[target_cols]
    X_forecast = forecast[feat_cols]

    sent_tag = "with_sent" if include_sentiment else "wo_sent"
    Path(path_dir).mkdir(parents=True, exist_ok=True)
    X_test.to_parquet(Path(path_dir) / f"X_test_{kind}_{sent_tag}_h{horizon}.parquet", index=False)

    # Preprocessor Pipeline
    preprocessor, _ = get_preprocessor(X_train)

    # Model kwargs + y_scale
    if kind == "xgboost":
        model_kwargs = {
            "horizon": horizon,
            "random_state": random_state,
            "tree_method": "hist",
            "n_jobs": -1
        }
        y_scale = True
    elif kind == "linreg":
        model_kwargs = {
            "horizon": horizon,
            "random_state": random_state,
            "multioutput": True,
            "max_iter": 2000
        }
        y_scale = True
    else:
        raise ValueError(f"Unknown kind: {kind}")

    # Base model and Trainer
    base_model = build_model(kind, **model_kwargs)
    trainer = ModelTrainer(
        model=base_model,
        name=f"{kind}_h{horizon}",
        config={
            "optimization_metric": "rmse",
            "gap": 0,
            "seed": random_state,
            "tune_targets": [0, 7, 14, 29],
            "max_train_size": 5000
        },
        preprocessor=preprocessor,
        y_scale=y_scale
    )

    # Tune Optuna
    pruner = optuna.pruners.SuccessiveHalvingPruner(min_resource=1, reduction_factor=3)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
        pruner=pruner,
        study_name=f"{kind}_{'with' if include_sentiment else 'wo'}_sent_h{horizon}"
    )
    n_trials = 60 if kind == "xgboost" else 30
    study.optimize(lambda tr: trainer.objective(tr, X_train, y_train, n_splits=3), n_trials=n_trials, timeout=1200)

    best_params = study.best_trial.user_attrs.get("best_params", {}) or {}
    best_params.setdefault("random_state", random_state)

    if kind == "xgboost":
        final_model_kwargs = {**model_kwargs, **best_params, "tree_method": "hist", "device": "cuda", "n_jobs": 1}
    else:
        final_model_kwargs = {**model_kwargs, **best_params}

    # Rebuild best and final Trainer
    best_model = build_model(kind, **final_model_kwargs)
    final_trainer = ModelTrainer(
        best_model,
        name=f"{kind}_h{horizon}",
        config={
            "optimization_metric": "rmse",
            "gap": 0,
            "seed": random_state
        },
        preprocessor=preprocessor,
        y_scale=y_scale,
    )
    final_trainer.fit(X_train, y_train, X_val, y_val)

    # Metrics
    meta = {
        "scenario": f"direct_{'with' if include_sentiment else 'wo'}_sent",
        "model": f"{kind}_h{horizon}",
        "with_sentiment": include_sentiment,
        "horizon": horizon,
        "n_features": len(feat_cols)
    }
    results = {
        "train": final_trainer.evaluate(X_train, y_train),
        "val": final_trainer.evaluate(X_val, y_val),
        "test": final_trainer.evaluate(X_test, y_test),
    }
    rows = [{"split": k, **meta, **v} for k, v in results.items()]
    res_df = pd.DataFrame(rows)

    # Save artifacts
    pd.Series(best_params).to_csv(Path(path_dir) / f"final_params_{kind}_{meta['scenario']}_h{horizon}.csv")
    model_path = final_trainer.save()
    print("Best params:", best_params)
    print("Saved model:", model_path)

    # Context for overlays
    ctx = {
        "df_full": df_full,
        "train": train, "val": val, "test": test, "forecast": forecast,
        "X_test": X_test, "y_test": y_test, "X_forecast": X_forecast,
        "feat_cols": feat_cols, "target_cols": target_cols
    }
    return res_df, study, final_trainer, ctx
