from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Self

import numpy as np
import pandas as pd
import torch
from optuna.integration import XGBoostPruningCallback
from xgboost import XGBRegressor
from xgboost.callback import EarlyStopping

from src.models.base import Base


@dataclass
class XGBoost(Base):
    """
    XGBoost regressor with optional multi-step (multi-horizon) support.

    - Trains either:
        • Single model (if y is 1D or shape (n, 1))
        • One model per horizon (if y is (n, H))
    - Supports early stopping (with Optuna pruning if attached).
    - Currently CPU-only, sequential training across horizons.
    """

    name: str = field(default="xgboost", init=False)

    n_jobs: int
    random_state: int
    horizon: int = 1

    # Core params
    n_estimators: int = 300
    learning_rate: float = 0.1
    max_depth: int = 4
    min_child_weight: float = 1.0
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    gamma: float = 0.0
    importance_type: str = "gain"
    eval_metric: str = "rmse"
    objective: str = "reg:squarederror"
    device: str = "cpu"

    # Sampling
    subsample: float = 0.8
    colsample_bytree: float = 0.8

    # Tree/backend
    tree_method: str = "hist"
    max_bin: int = 256
    grow_policy: str = "depthwise"
    max_leaves: int = 0

    # Threading
    outer_n_jobs: int = 1

    # Training
    early_stopping_rounds: int = 50
    multioutput: bool = False

    # Runtime state
    _single: Optional[XGBRegressor] = field(default=None, init=False, repr=False)
    _multi: Optional[List[XGBRegressor]] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        super().__init__(horizon=self.horizon, random_state=self.random_state, n_jobs=self.n_jobs)

    @staticmethod
    def _as_2d(y) -> np.ndarray:
        y = np.asarray(y)
        return y.reshape(-1, 1) if y.ndim == 1 else y

    @staticmethod
    def _as_float32(a):
        return np.asarray(a, dtype=np.float32, order="C")

    def _new_estimator(self, seed_offset: int = 0) -> XGBRegressor:
        params = {
            "n_estimators": int(self.n_estimators),
            "learning_rate": float(self.learning_rate),
            "max_depth": int(self.max_depth),
            "subsample": float(self.subsample),
            "colsample_bytree": float(self.colsample_bytree),
            "min_child_weight": float(self.min_child_weight),
            "reg_alpha": float(self.reg_alpha),
            "reg_lambda": float(self.reg_lambda),
            "gamma": float(self.gamma),
            "n_jobs": self.n_jobs,
            "random_state": int(self.random_state + seed_offset),
            "objective": self.objective,
            "importance_type": self.importance_type,
            "eval_metric": self.eval_metric,
            "max_bin": int(self.max_bin),
            "grow_policy": self.grow_policy,
            "max_leaves": int(self.max_leaves),
            "tree_method": self.tree_method,
            "device": self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        }
        if self.grow_policy == "lossguide":
            params["max_depth"] = 0
            if params["max_leaves"] <= 0:
                params["max_leaves"] = 128

        return XGBRegressor(**params)

    def _fit_with_val_single(self, model: XGBRegressor, X, y, Xv, yv):
        """Fit a single estimator with early stopping and optional Optuna pruning."""
        X, y = self._as_float32(X), self._as_float32(y)
        Xv, yv = self._as_float32(Xv), self._as_float32(yv)

        # Try modern callback API
        try:
            callbacks = [EarlyStopping(rounds=int(self.early_stopping_rounds), save_best=True, maximize=False)]
            trial = getattr(self, "_trial", None)
            if trial is not None:
                callbacks.append(XGBoostPruningCallback(trial, f"validation_1-{self.eval_metric}"))
            model.fit(X, y, eval_set=[(X, y), (Xv, yv)], callbacks=callbacks, verbose=False)
            return
        except TypeError:
            pass

        # Fallback to legacy early_stopping_rounds
        try:
            model.fit(
                X, y,
                eval_set=[(X, y), (Xv, yv)],
                early_stopping_rounds=int(self.early_stopping_rounds),
                verbose=False,
            )
            return
        except TypeError:
            pass

        # Last resort: fit without validation
        model.fit(X, y)

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> Self:
        Y = self._as_2d(y)
        X = self._as_float32(X)

        if Y.shape[1] == 1:
            self._single, self._multi = self._new_estimator(), None
            self._single.fit(X, self._as_float32(Y.ravel()))
            return self

        n_targets = Y.shape[1]
        self._single, self._multi = None, []
        for h in range(n_targets):
            m = self._new_estimator(seed_offset=h)
            m.fit(X, self._as_float32(Y[:, h]))
            self._multi.append(m)
        return self

    def train(
            self,
            X_tr: pd.DataFrame,
            y_tr: np.ndarray,
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[np.ndarray] = None
    ) -> Self:
        if X_val is None or y_val is None or self.early_stopping_rounds <= 0:
            return self.fit(X_tr, y_tr)

        Ytr = self._as_2d(y_tr)
        Yva = self._as_2d(y_val)
        X_tr = self._as_float32(X_tr)
        X_val = self._as_float32(X_val)

        if Ytr.shape[1] == 1:
            self._single, self._multi = self._new_estimator(), None
            self._fit_with_val_single(self._single, X_tr, Ytr.ravel(), X_val, Yva.ravel())
            return self

        n_targets = Ytr.shape[1]
        self._single, self._multi = None, []
        for h in range(n_targets):
            m = self._new_estimator(seed_offset=h)
            self._fit_with_val_single(m, X_tr, Ytr[:, h], X_val, Yva[:, h])
            self._multi.append(m)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = self._as_float32(X)
        if self._single is not None:
            return np.asarray(self._single.predict(X)).reshape(-1)
        if self._multi:
            return np.column_stack([m.predict(X) for m in self._multi])
        raise RuntimeError("XGBoost: call fit/train before predict.")

    @staticmethod
    def search_space(trial) -> dict:
        space = {
            "n_estimators": trial.suggest_categorical("n_estimators", [200, 300, 400]),
            "learning_rate": trial.suggest_categorical("learning_rate", [0.05, 0.1, 0.15]),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 2.0),
            "gamma": trial.suggest_float("gamma", 0.0, 1.0),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "eval_metric": trial.suggest_categorical("eval_metric", ["rmse", "mae"]),
            "tree_method": "hist",
            "grow_policy": "depthwise"
        }
        return space
