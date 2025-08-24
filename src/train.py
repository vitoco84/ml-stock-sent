from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import optuna
from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from src.logger import get_logger
from src.metrics import metrics
from src.scaler import SafeStandardScaler


class ModelTrainer:
    """Wrapper Class for Training and Tuning with Optuna."""

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
        self.preprocessor = preprocessor or StandardScaler()
        self.y_scale = y_scale
        self.y_scaler: Optional[SafeStandardScaler] = None

        self.logger = get_logger(self.__class__.__name__)
        self.logger.info(f"Initialized ModelTrainer for model: {name}")

    def _prep_X(self, pre, X_tr, X_va=None):
        pre_est = pre or StandardScaler()
        pre_ = clone(pre_est)
        X_tr_s = pre_.fit_transform(X_tr).astype(np.float32)
        X_va_s = pre_.transform(X_va).astype(np.float32) if X_va is not None else None
        return pre_, X_tr_s, X_va_s

    def _prep_y(self, y_tr, y_va=None):
        if not self.y_scale:
            return None, y_tr, y_va
        s = SafeStandardScaler()
        y_tr_s = s.fit_transform(y_tr)
        y_va_s = s.transform(y_va) if y_va is not None else None
        return s, y_tr_s, y_va_s

    def _build_candidate(self, params: Dict[str, Any], trial) -> Any:
        base_params = self.model.get_params()
        cand = self.model.__class__(**{**base_params, **params})
        setattr(cand, "_trial", trial)
        return cand

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        self.logger.info("Starting model training...")
        pre_, X_tr_s, X_va_s = self._prep_X(self.preprocessor, X_train, X_val)
        y_s, y_tr_s, y_va_s = self._prep_y(y_train, y_val)
        self.y_scaler = y_s
        self.preprocessor = pre_

        if hasattr(self.model, "train") and X_val is not None and y_val is not None:
            self.model.train(X_tr_s, y_tr_s, X_va_s, y_va_s)
        else:
            self.model.fit(X_tr_s, y_tr_s)
        return self

    def predict(self, X):
        X_s = self.preprocessor.transform(X).astype(np.float32)
        pred = self.model.predict(X_s)
        pred = np.asarray(pred)
        if pred.ndim == 1:
            pred = pred.reshape(-1, 1)
        if self.y_scale and self.y_scaler is not None:
            pred = self.y_scaler.inverse_transform(pred)
        return pred

    def evaluate(self, X, y):
        self.logger.info("Evaluating model...")
        preds = self.predict(X)
        return metrics(np.asarray(y), np.asarray(preds))

    def save(self) -> Path:
        self.output_path.mkdir(parents=True, exist_ok=True)
        path = self.output_path / f"{self.name}.pkl"
        joblib.dump(
            {
                "model": self.model,
                "preprocessor": self.preprocessor,
                "y_scaler": self.y_scaler,
                "y_scale": self.y_scale
            }, str(path),
        )
        return path

    @classmethod
    def load(cls, path: str):
        blob = joblib.load(path)
        return blob["model"], blob["preprocessor"], blob.get("y_scaler"), blob.get("y_scale", False)

    def _get_search_params(self, trial) -> Dict[str, Any]:
        space_fn = getattr(self.model.__class__, "search_space", None)
        params = space_fn(trial) if callable(space_fn) else {}
        params["random_state"] = self.config.get("seed", 42)
        return params

    @staticmethod
    def _fit_or_train(X_tr_s, X_va_s, candidate, y_tr_s, y_va_s) -> None:
        if hasattr(candidate, "train"):
            candidate.train(X_tr_s, y_tr_s, X_va_s, y_va_s)
        else:
            candidate.fit(X_tr_s, y_tr_s)

    def _maybe_inverse(self, pred, y_s):
        if self.y_scale and y_s is not None:
            pred = y_s.inverse_transform(pred)
        return pred

    @staticmethod
    def _score_metric(y_true, y_pred, metric_name: str) -> float:
        return metrics(np.asarray(y_true), np.asarray(y_pred))[metric_name]

    @staticmethod
    def _is_maximize(metric_name: str) -> bool:
        return metric_name.lower() in {"r2"}

    def objective(self, trial, X, y, n_splits: int = 3):
        self.logger.info("Starting model tuning...")

        params = self._get_search_params(trial)

        tscv = TimeSeriesSplit(n_splits=n_splits, gap=int(self.config.get("gap", 0)))
        metric_name = self.config.get("optimization_metric", "rmse")

        scores = []
        for tr_idx, va_idx in tscv.split(X):
            # Split
            X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

            # Candidate Prep
            candidate = self._build_candidate(params, trial)
            _, X_tr_s, X_va_s = self._prep_X(self.preprocessor, X_tr, X_va)
            y_s, y_tr_s, y_va_s = self._prep_y(y_tr, y_va)

            # Train
            self._fit_or_train(X_tr_s, X_va_s, candidate, y_tr_s, y_va_s)

            # Predict and inverse
            pred = candidate.predict(X_va_s)
            pred = self._maybe_inverse(pred, y_s)

            # Score, Report, Prune
            folde_score = self._score_metric(y_va, pred, metric_name)
            scores.append(folde_score)

            trial.report(folde_score, step=len(scores))
            if trial.should_prune():
                raise optuna.TrialPruned()

        mean_score = float(np.mean(scores))
        trial.set_user_attr("best_params", params)
        trial.set_user_attr("cv_scores", scores)

        return -mean_score if self._is_maximize(metric_name) else mean_score
