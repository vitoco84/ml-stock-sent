from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
from sklearn.base import clone, TransformerMixin
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from src.logger import get_logger
from src.metrics import metrics
from src.models.base import Base


class ModelTrainer:
    def __init__(
            self,
            model: Base,
            name: str,
            config: Dict[str, Any],
            output_path: str = "../data/models",
            preprocessor: Optional[TransformerMixin] = None,
            y_scale: bool = True
    ):
        self.model = model
        self.name = name
        self.config = config
        self.output_path = Path(output_path)
        self.preprocessor = preprocessor or StandardScaler()
        self.y_scale = y_scale
        self.y_scaler: Optional[StandardScaler] = None

        self.logger = get_logger(self.__class__.__name__)
        self.logger.info(f"Initialized ModelTrainer for model: {name}")

    def fit(self, X_train, y_train, X_val=None, y_val=None, best_params: Optional[dict] = None):
        self.logger.info("Starting model training...")
        scaler: StandardScaler | object = clone(self.preprocessor)
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val) if X_val is not None else None

        if self.y_scale:
            self.y_scaler = StandardScaler()
            y_train_scaled = self.y_scaler.fit_transform(np.asarray(y_train))
            y_val_scaled = self.y_scaler.transform(np.asarray(y_val)) if y_val is not None else None
        else:
            y_train_scaled = y_train
            y_val_scaled = y_val

        if best_params:
            self.logger.info(f"Training final model with best params: {best_params}")
            if hasattr(self.model, "set_params"):
                self.model.set_params(**best_params)
            else:
                self.model = self.model.__class__(**best_params)

        if hasattr(self.model, "train"):
            self.model.train(X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled)
        else:
            self.model.fit(X_train_scaled, y_train_scaled)

        self.preprocessor = scaler

    def evaluate(self, X, y):
        self.logger.info("Evaluating model...")
        X_scaled = self.preprocessor.transform(X)
        preds = self.model.predict(X_scaled)

        if self.y_scale and self.y_scaler is not None:
            preds = self.y_scaler.inverse_transform(np.asarray(preds))

        return metrics(np.asarray(y), preds)

    def predict(self, X):
        X_scaled = self.preprocessor.transform(X)
        preds = self.model.predict(X_scaled)
        if self.y_scale and self.y_scaler is not None:
            preds = self.y_scaler.inverse_transform(np.asarray(preds))
        return preds

    def save(self) -> Path:
        self.output_path.mkdir(parents=True, exist_ok=True)
        model_path = self.output_path / f"{self.name}.pkl"
        joblib.dump({
            "model": self.model,
            "preprocessor": self.preprocessor,
            "y_scaler": self.y_scaler,  # <-- ADD
            "y_scale": self.y_scale,  # <-- ADD
        }, str(model_path))
        return model_path

    @classmethod
    def load(cls, path: str):
        obj = joblib.load(path)
        return obj["model"], obj["preprocessor"], obj.get("y_scaler"), obj.get("y_scale", False)

    def objective(self, trial, X, y, n_splits: int = 3):
        params = self.model.suggest_hyperparameters(trial)
        candidate = self.model.__class__(**{**self.model.get_params(), **params})

        inner_gap = int(self.config.get("gap", 0))
        tscv = TimeSeriesSplit(n_splits=n_splits, gap=inner_gap)

        metric_name = self.config.get("optimization_metric", "rmse")
        higher_is_better = metric_name.lower() in {"r2"}

        scores = []
        for tr_idx, va_idx in tscv.split(X):
            X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

            pre = clone(self.preprocessor)
            X_tr_s = pre.fit_transform(X_tr)
            X_va_s = pre.transform(X_va)

            if self.y_scale:
                y_sclr = StandardScaler()
                y_tr_s = y_sclr.fit_transform(np.asarray(y_tr))
                y_va_s = y_sclr.transform(np.asarray(y_va))
            else:
                y_tr_s, y_va_s = y_tr, y_va

            if hasattr(candidate, "train"):
                candidate.train(X_tr_s, y_tr_s, X_va_s, y_va_s)
            else:
                candidate.fit(X_tr_s, y_tr_s)

            preds = candidate.predict(X_va_s)

            if self.y_scale:
                preds = y_sclr.inverse_transform(np.asarray(preds))

            m = metrics(np.asarray(y_va), preds)[metric_name]
            scores.append(-m if higher_is_better else m)

        trial.set_user_attr("best_params", params)
        return float(np.mean(scores))
