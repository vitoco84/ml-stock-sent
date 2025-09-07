from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Self

import numpy as np
import pandas as pd
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

    name: str = "xgboost"

    # Core params
    random_state: int = 42
    n_estimators: int = 800
    learning_rate: float = 0.05
    max_depth: int = 5
    min_child_weight: float = 1.0
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    gamma: float = 0.0
    importance_type: str = "gain"
    eval_metric: str = "rmse"
    objective: str = "reg:squarederror"

    # Sampling
    subsample: float = 1.0
    colsample_bytree: float = 1.0

    # Tree/backend
    tree_method: str = "hist"
    max_bin: int = 256
    grow_policy: str = "depthwise"
    max_leaves: int = 0

    # Threading
    n_jobs: int = -1  # threads per estimator
    outer_n_jobs: int = 1  # horizons trained sequentially

    # Training
    early_stopping_rounds: int = 200
    horizon: int = 1
    multioutput: bool = False

    # Runtime state
    _single: Optional[XGBRegressor] = field(default=None, init=False, repr=False)
    _multi: Optional[List[XGBRegressor]] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        super().__init__(horizon=self.horizon, random_state=self.random_state)

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
            "device": "cpu",
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
            "n_estimators": trial.suggest_int("n_estimators", 600, 2000, step=200),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 20.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "max_bin": trial.suggest_int("max_bin", 128, 512, step=64),
            "grow_policy": trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"]),
            "objective": trial.suggest_categorical("objective", ["reg:squarederror", "reg:absoluteerror"]),
            "eval_metric": trial.suggest_categorical("eval_metric", ["rmse", "mae"]),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "tree_method": "hist"
        }
        if space["grow_policy"] == "lossguide":
            space["max_leaves"] = trial.suggest_int("max_leaves", 64, 1024, step=64)
            space["max_depth"] = 0
        else:
            space["max_leaves"] = 0
        return space
