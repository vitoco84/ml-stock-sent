from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import optuna
from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit

from src.logger import get_logger
from src.metrics import metrics
from src.models.search_spaces import SEARCH_SPACES
from src.scaler import SafeStandardScaler


class ModelTrainer:
    """
    High-level training/orchestration wrapper for time-series regression models.
    Hyperparameter Tuning with Optuna using TimeSeriesSplit.
    """

    def __init__(
            self,
            model: Any,
            name: str,
            config: Dict[str, Any],
            output_path: str = "../data/models",
            preprocessor: Any = None,
            y_scale: bool = True
    ):
        self.model = model
        self.name = name
        self.config = config
        self.output_path = Path(output_path)
        self.preprocessor = preprocessor or SafeStandardScaler()
        self.y_scale = y_scale
        self.y_scaler: Optional[SafeStandardScaler] = None

        self.logger = get_logger(self.__class__.__name__)
        self.logger.info(f"Initialized ModelTrainer for model: {name}")

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        self.logger.info("Starting model training...")
        scaler: SafeStandardScaler | object = clone(self.preprocessor)
        X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
        X_val_scaled = scaler.transform(X_val).astype(np.float32) if X_val is not None else None

        if self.y_scale:
            self.y_scaler = SafeStandardScaler()
            y_train_scaled = self.y_scaler.fit_transform(y_train)
            y_val_scaled = self.y_scaler.transform(y_val) if y_val is not None else None
        else:
            y_train_scaled = y_train
            y_val_scaled = y_val

        if hasattr(self.model, "train"):
            self.model.train(X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled)
        else:
            self.model.fit(X_train_scaled, y_train_scaled)

        self.preprocessor = scaler

    def evaluate(self, X, y):
        self.logger.info("Evaluating model...")
        X_scaled = self.preprocessor.transform(X).astype(np.float32)
        preds = self.model.predict(X_scaled)
        preds = np.asarray(preds)
        if self.y_scale and self.y_scaler is not None:
            preds = self.y_scaler.inverse_transform(preds)
        return metrics(np.asarray(y), preds)

    def predict(self, X):
        X_scaled = self.preprocessor.transform(X).astype(np.float32)
        preds = self.model.predict(X_scaled)
        preds = np.asarray(preds)
        if self.y_scale and self.y_scaler is not None:
            preds = self.y_scaler.inverse_transform(preds)
        return preds

    def save(self) -> Path:
        self.output_path.mkdir(parents=True, exist_ok=True)
        model_path = self.output_path / f"{self.name}.pkl"
        joblib.dump(
            {
                "model": self.model,
                "preprocessor": self.preprocessor,
                "y_scaler": self.y_scaler,
                "y_scale": self.y_scale,
            },
            str(model_path),
        )
        return model_path

    @classmethod
    def load(cls, path: str):
        obj = joblib.load(path)
        return obj["model"], obj["preprocessor"], obj.get("y_scaler"), obj.get("y_scale", False)

    def objective(self, trial, X, y, n_splits: int = 3):
        tune_k = self.config.get("tune_targets", 0)
        if isinstance(tune_k, list) and hasattr(y, "iloc"):
            y = y.iloc[:, tune_k]
        elif isinstance(tune_k, int) and tune_k > 0:
            y = y.iloc[:, :tune_k]

        space_key = getattr(self.model, "name", None)
        if hasattr(self.model, "base_cls"):
            space_key = getattr(self.model.base_cls, "name", space_key)

        space_fn = SEARCH_SPACES.get(space_key)
        params = space_fn(trial) if space_fn else {}
        params["random_state"] = self.config.get("seed", 42)

        def build(hp):
            if hasattr(self.model, "base_cls"):
                base_cls = self.model.base_cls
                merged = {**getattr(self.model, "base_params", {}), **hp}
                cand = self.model.__class__(
                    base_cls=base_cls,
                    base_params=merged,
                    horizon=self.model.horizon,
                    random_state=hp["random_state"],
                )
            else:
                base_params = self.model.get_params()
                cand = self.model.__class__(**{**base_params, **hp})
            setattr(cand, "_trial", trial)
            return cand

        tscv = TimeSeriesSplit(
            n_splits=n_splits,
            gap=int(self.config.get("gap", 0)),
            max_train_size=self.config.get("max_train_size")
        )
        metric_name = self.config.get("optimization_metric", "rmse")
        higher_is_better = metric_name.lower() in {"r2"}

        scores = []
        for tr_idx, va_idx in tscv.split(X):
            candidate = build(params)

            X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

            pre = clone(self.preprocessor)
            X_tr_s = pre.fit_transform(X_tr).astype(np.float32)
            X_va_s = pre.transform(X_va).astype(np.float32)

            if self.y_scale:
                y_sclr = SafeStandardScaler()
                y_tr_s = y_sclr.fit_transform(y_tr)
                y_va_s = y_sclr.transform(y_va)
            else:
                y_tr_s, y_va_s = y_tr, y_va

            if hasattr(candidate, "train"):
                candidate.train(X_tr_s, y_tr_s, X_va_s, y_va_s)
            else:
                candidate.fit(X_tr_s, y_tr_s)

            preds = np.asarray(candidate.predict(X_va_s))
            if self.y_scale:
                preds = y_sclr.inverse_transform(preds)

            m = metrics(np.asarray(y_va), preds)[metric_name]
            scores.append(-m if higher_is_better else m)

            trial.report(scores[-1], step=len(scores))
            if trial.should_prune():
                raise optuna.TrialPruned()

        trial.set_user_attr("best_params", params)
        return float(np.mean(scores))
