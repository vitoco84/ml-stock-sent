from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from joblib import delayed, Parallel
from optuna.integration import XGBoostPruningCallback
from xgboost.callback import EarlyStopping

from src.models.base import Base
from xgboost import XGBRegressor


@dataclass
class XGBoost(Base):
    """
    XGBoost with native multi-step support (H>1 trains one estimator per horizon).
    Early stopping + Optuna pruning per step. Parallelized across horizons on CPU.

    Note (XGBoost >= 2.0):
      - For GPU training use: tree_method="hist", device="cuda" (no gpu_hist).
    """
    name = "xgboost"

    # Core params
    random_state: int = 42
    n_estimators: int = 400
    learning_rate: float = 0.05
    max_depth: int = 5
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    gamma: float = 0.0
    min_child_weight: float = 1.0
    importance_type: str = "gain"
    eval_metric: str = "rmse"
    objective: str = "reg:squarederror"

    # Tree/backend
    device: str = "cpu"  # "cuda" for GPU
    tree_method: str = "hist"
    max_bin: int = 128
    grow_policy: str = "depthwise"
    max_leaves: int = 0

    # Threading
    n_jobs: int = -1  # global thread budget
    outer_n_jobs: int = 1  # horizon-level parallel workers (CPU)

    # Training
    early_stopping_rounds: int = 30
    horizon: int = 1
    multioutput: bool = False

    # Runtime state
    _single: Optional[XGBRegressor] = field(default=None, init=False, repr=False)
    _multi: Optional[List[XGBRegressor]] = field(default=None, init=False, repr=False)
    _last_outer_jobs: int = field(default=1, init=False, repr=False)

    def __post_init__(self):
        super().__init__(horizon=self.horizon, random_state=self.random_state)

    @staticmethod
    def _as_2d(y) -> np.ndarray:
        y = np.asarray(y)
        return y.reshape(-1, 1) if y.ndim == 1 else y

    @staticmethod
    def _as_float32(a):
        return np.asarray(a, dtype=np.float32, order="C")

    def _effective_outer_jobs(self, n_targets: int) -> int:
        """How many horizons to train in parallel."""
        if self.device == "cuda":
            return 1
        if self.outer_n_jobs in (None, 0):
            return 1
        if self.outer_n_jobs < 0:
            return min(n_targets, os.cpu_count() or 1)
        return min(n_targets, self.outer_n_jobs)

    def _effective_inner_jobs(self) -> int:
        """Threads per estimator; divide global budget by actual outer workers."""
        if self.device == "cuda":
            return 1
        cpu_total = os.cpu_count() or 1
        if self.n_jobs in (None, 0):
            base = cpu_total
        elif self.n_jobs < 0:
            base = cpu_total
        else:
            base = self.n_jobs
        outer = max(1, getattr(self, "_last_outer_jobs", 1))
        return max(1, base // outer)

    def _new_estimator(self, seed_offset: int = 0) -> XGBRegressor:
        """Create a fresh XGBRegressor with current hyperparams (2.x-friendly GPU)."""
        return XGBRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            gamma=self.gamma,
            min_child_weight=self.min_child_weight,
            n_jobs=self._effective_inner_jobs(),
            random_state=int(self.random_state + seed_offset),
            objective=self.objective,
            importance_type=self.importance_type,
            eval_metric=self.eval_metric,
            max_bin=int(self.max_bin),
            grow_policy=self.grow_policy,
            max_leaves=int(self.max_leaves),
            tree_method="hist",
            device="cuda" if self.device == "cuda" else "cpu",
        )

    def _fit_with_val_single(self, model: XGBRegressor, X, y, Xv, yv):
        """Fit a single estimator with early stopping and optional Optuna pruning."""
        X, y = self._as_float32(X), self._as_float32(y)
        Xv, yv = self._as_float32(Xv), self._as_float32(yv)

        try:
            callbacks = [EarlyStopping(rounds=int(self.early_stopping_rounds), save_best=True, maximize=False)]
            trial = getattr(self, "_trial", None)
            if trial is not None:
                callbacks.append(XGBoostPruningCallback(trial, f"validation_1-{self.eval_metric}"))
            model.fit(X, y, eval_set=[(X, y), (Xv, yv)], callbacks=callbacks, verbose=False)
            return
        except TypeError:
            pass

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

        model.fit(X, y)

    def fit(self, X, y) -> XGBoost:
        Y = self._as_2d(y)
        X = self._as_float32(X)

        if Y.shape[1] == 1:
            self._single, self._multi = self._new_estimator(), None
            self._single.fit(X, self._as_float32(Y.ravel()))
            return self

        n_targets = Y.shape[1]
        jobs = self._effective_outer_jobs(n_targets)
        self._last_outer_jobs = jobs
        self._single, self._multi = None, [None] * n_targets

        def _fit_one(j):
            m = self._new_estimator(seed_offset=j)
            m.fit(X, self._as_float32(Y[:, j]))
            return m

        if jobs == 1:
            self._multi = [_fit_one(j) for j in range(n_targets)]
        else:
            self._multi = Parallel(n_jobs=jobs, prefer="threads")(
                delayed(_fit_one)(j) for j in range(n_targets)
            )
        return self

    def train(self, X_tr, y_tr, X_val=None, y_val=None) -> XGBoost:
        Ytr = self._as_2d(y_tr)
        X_tr = self._as_float32(X_tr)

        if X_val is None or y_val is None or self.early_stopping_rounds <= 0:
            return self.fit(X_tr, y_tr)

        Yva = self._as_2d(y_val)
        X_val = self._as_float32(X_val)

        if Ytr.shape[1] == 1:
            self._single, self._multi = self._new_estimator(), None
            self._fit_with_val_single(self._single, X_tr, Ytr.ravel(), X_val, Yva.ravel())
            return self

        n_targets = Ytr.shape[1]
        jobs = self._effective_outer_jobs(n_targets)
        self._last_outer_jobs = jobs
        self._single, self._multi = None, [None] * n_targets

        def _fit_one_with_val(j):
            m = self._new_estimator(seed_offset=j)
            self._fit_with_val_single(m, X_tr, Ytr[:, j], X_val, Yva[:, j])
            return m

        if jobs == 1:
            self._multi = [_fit_one_with_val(j) for j in range(n_targets)]
        else:
            self._multi = Parallel(n_jobs=jobs, prefer="threads")(
                delayed(_fit_one_with_val)(j) for j in range(n_targets)
            )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = self._as_float32(X)
        if self._single is not None:
            return np.asarray(self._single.predict(X)).reshape(-1)
        if self._multi:
            return np.column_stack([m.predict(X) for m in self._multi])
        raise RuntimeError("XGBoost: call fit/train before predict.")

    @staticmethod
    def search_space(trial):
        space = {
            "n_estimators": trial.suggest_int("n_estimators", 400, 1200, step=200),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 10.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 5.0),
            "early_stopping_rounds": 100,
            "max_bin": trial.suggest_int("max_bin", 128, 512, step=64),
            "grow_policy": trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"]),
            "objective": trial.suggest_categorical("objective", ["reg:squarederror", "reg:absoluteerror"]),
            "eval_metric": trial.suggest_categorical("eval_metric", ["rmse", "mae"]),
            "tree_method": "hist",
        }
        if space["grow_policy"] == "lossguide":
            space["max_leaves"] = trial.suggest_int("max_leaves", 64, 512, step=64)
            space["max_depth"] = 0
        else:
            space["max_leaves"] = 0
        return space
