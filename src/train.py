from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any, Optional, Tuple

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from src.logger import get_logger
from src.metrics import metrics
from src.scaler import SafeStandardScaler


class ModelTrainer:
    """
    Trainer wrapper for ML models (sklearn & custom).

    Features
    --------
    - Works with `.fit()` (sklearn) or `.train()` (custom NN models).
    - Handles feature preprocessing (`StandardScaler`, pipelines, or custom).
    - Optionally scales target `y` using `SafeStandardScaler`.
    - Provides evaluation with common metrics.
    - Integrates with Optuna for hyperparameter search.
    - Save/load support with joblib.
    """

    def __init__(
            self,
            model: Any,
            name: str,
            config: dict[str, Any],
            output_path: str | Path = "../data/models",
            preprocessor: Any = None,
            y_scale: bool = True
    ) -> None:
        self.model = model
        self.name = name
        self.config = config
        self.output_path = Path(output_path)
        self.preprocessor = preprocessor
        self.y_scale = y_scale
        self.y_scaler: Optional[SafeStandardScaler] = None
        self.is_sequence: bool = getattr(model, "input_mode", "tabular") == "sequence"

        self.logger = get_logger(self.__class__.__name__)
        self.logger.info(f"Initialized trainer for {name}")

    def _prep_X(
            self,
            pre: Any,
            X_tr: pd.DataFrame,
            X_va: Optional[pd.DataFrame] = None,
    ) -> Tuple[Any, np.ndarray, Optional[np.ndarray]]:
        """
        Prepare features (fit/transform).
        For sequence models, preprocessing is skipped.
        """
        if self.is_sequence:
            return None, X_tr, X_va

        pre_est = pre or StandardScaler()
        pre_ = clone(pre_est)
        X_tr_s = pre_.fit_transform(X_tr)
        X_va_s = pre_.transform(X_va) if X_va is not None else None
        return pre_, X_tr_s, X_va_s

    def _prep_y(
            self,
            y_tr: Any,
            y_va: Optional[Any] = None
    ) -> Tuple[Optional[SafeStandardScaler], np.ndarray, Optional[np.ndarray]]:
        """
        Prepare target (fit/transform).
        Returns scaler, transformed train, transformed val.
        """
        if not self.y_scale:
            return None, np.asarray(y_tr), np.asarray(y_va) if y_va is not None else None
        s = SafeStandardScaler()
        y_tr_s = s.fit_transform(y_tr)
        y_va_s = s.transform(y_va) if y_va is not None else None
        return s, y_tr_s, y_va_s

    def fit(
            self,
            X_train: pd.DataFrame,
            y_train: Any,
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[pd.DataFrame | np.ndarray] = None
    ) -> ModelTrainer:
        """Fit model on train set (and val set if provided)."""
        pre_, X_tr_s, X_va_s = self._prep_X(self.preprocessor, X_train, X_val)
        y_s, y_tr_s, y_va_s = self._prep_y(y_train, y_val)
        self.y_scaler = y_s
        self.preprocessor = pre_

        if hasattr(self.model, "train") and X_val is not None and y_val is not None:
            self.model.train(X_tr_s, y_tr_s, X_va_s, y_va_s)
        else:
            self.model.fit(X_tr_s, y_tr_s)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict with trained model. Inverse-scales if needed."""
        X_s = (
            self.preprocessor.transform(X)
            if (self.preprocessor is not None and not self.is_sequence)
            else X
        )
        pred = np.asarray(self.model.predict(X_s))
        if pred.ndim == 1:
            pred = pred.reshape(-1, 1)
        if self.y_scale and self.y_scaler is not None:
            pred = self.y_scaler.inverse_transform(pred)
        return pred

    def evaluate(self, X: pd.DataFrame, y: Any) -> dict[str, Any]:
        """Evaluate model using standard metrics."""
        self.logger.info("Evaluating model...")
        preds = self.predict(X)
        return metrics(np.asarray(y), np.asarray(preds))

    def save(self) -> Path:
        """Save model, preprocessor, scalers."""
        self.output_path.mkdir(parents=True, exist_ok=True)
        path = self.output_path / f"{self.name}.pkl"
        joblib.dump(
            {
                "model": self.model,
                "preprocessor": self.preprocessor,
                "y_scaler": self.y_scaler,
                "y_scale": self.y_scale
            },
            str(path),
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> Tuple[Any, Any, Optional[SafeStandardScaler], bool]:
        """Load components from disk (returns tuple)."""
        blob = joblib.load(path)
        return (
            blob["model"],
            blob["preprocessor"],
            blob.get("y_scaler"),
            blob.get("y_scale", False)
        )

    def _get_search_params(self, trial: optuna.Trial) -> dict[str, Any]:
        """Return candidate params from model's `search_space` if defined."""
        space_fn = getattr(self.model.__class__, "search_space", None)
        params = space_fn(trial) if callable(space_fn) else {}
        params["random_state"] = self.config.get("seed", 42)
        return params

    def _build_candidate(self, params: dict[str, Any], trial: optuna.Trial) -> Any:
        """Clone model with candidate parameters."""
        base_params = self.model.get_params()
        cand = self.model.__class__(**{**base_params, **params})
        setattr(cand, "_trial", trial)
        return cand

    @staticmethod
    def _fit_or_train(
            X_tr_s: np.ndarray,
            X_va_s: Optional[np.ndarray],
            candidate: Any,
            y_tr_s: np.ndarray,
            y_va_s: Optional[np.ndarray]
    ) -> None:
        """Fit or train depending on candidate interface."""
        if hasattr(candidate, "train"):
            candidate.train(X_tr_s, y_tr_s, X_va_s, y_va_s)
        else:
            candidate.fit(X_tr_s, y_tr_s)

    @staticmethod
    def _score_metric(y_true: np.ndarray, y_pred: np.ndarray, metric_name: str) -> float:
        """Compute a single metric."""
        all_metrics = metrics(np.asarray(y_true), np.asarray(y_pred))
        if "aggregate" in all_metrics:
            return all_metrics["aggregate"][metric_name]
        return all_metrics[metric_name]

    @staticmethod
    def _walk_forward_splits(n_samples: int, train_size: int, test_size: int, step_size: int):
        """Generator for walk-forward (rolling origin) splits."""
        start = 0
        while start + train_size + test_size <= n_samples:
            train_idx = np.arange(0, start + train_size)
            val_idx = np.arange(start + train_size, start + train_size + test_size)
            yield train_idx, val_idx
            start += step_size

    def objective(
            self,
            trial: optuna.Trial,
            X: pd.DataFrame,
            y: Any,
            n_splits: int,
            walk_forward: bool
    ) -> float:
        """Optuna objective function using TimeSeries CV."""
        try:
            params = self._get_search_params(trial)
            metric_name = self.config.get("optimization_metric", "r2")

            if walk_forward:
                n_samples = len(X)
                train_size = int(n_samples * 0.7)
                val_size = int(n_samples * 0.1)
                step_size = val_size
                splits = self._walk_forward_splits(n_samples, train_size, val_size, step_size)
            else:
                tscv = TimeSeriesSplit(n_splits=n_splits, gap=int(self.config.get("gap", 0)))
                splits = tscv.split(X)

            scores: list[float] = []
            for fold, (tr_idx, va_idx) in enumerate(splits, start=1):
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
                pred = np.asarray(candidate.predict(X_va_s))
                if pred.ndim == 1:
                    pred = pred.reshape(-1, 1)

                # Inverse-scale
                if self.y_scale and y_s is not None:
                    y_va = y_s.inverse_transform(pd.DataFrame(y_va_s))
                    pred = y_s.inverse_transform(pd.DataFrame(pred))
                else:
                    y_va = np.asarray(y_va).ravel()
                    pred = pred.ravel()

                # Score, Report, Prune
                fold_score = self._score_metric(y_va, pred, metric_name)
                scores.append(fold_score)

                trial.report(fold_score, step=fold)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            mean_score = float(np.mean(scores))
            trial.set_user_attr("best_params", params)
            trial.set_user_attr("cv_scores", scores)
            return mean_score
        except ValueError as e:
            self.logger.info(f"Trial failed {e}")
            traceback.print_exc()
            raise
