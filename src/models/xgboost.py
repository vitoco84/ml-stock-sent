from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from optuna.integration import XGBoostPruningCallback
from xgboost.callback import EarlyStopping

from src.models.base import Base
from src.utils import is_cuda_available
from xgboost import XGBRegressor


@dataclass
class XGBoost(Base):
    """Single-output XGBoost. Uses ES when validation is provided."""
    name = "xgboost"

    random_state: int = 42
    n_estimators: int = 100
    learning_rate: float = 0.05
    max_depth: int = 6
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    gamma: float = 0.0
    min_child_weight: float = 1.0
    tree_method: str = "hist"
    n_jobs: int = -1
    importance_type: str = "gain"
    eval_metric: str = "rmse"
    objective: str = "reg:squarederror"

    max_bin: int | None = None
    grow_policy: str | None = None
    max_leaves: int | None = None
    device: str | None = None

    early_stopping_rounds: int = 50
    verbose_fit: bool = False

    model: XGBRegressor = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self._build()

    def _build(self):
        n_jobs = 1 if self.device == "cuda" else self.n_jobs

        extra = {}
        if self.device is not None:
            extra["device"] = self.device

        # Optional hyperparams
        if self.max_bin is not None:
            extra["max_bin"] = int(self.max_bin)
        if self.grow_policy is not None:
            extra["grow_policy"] = self.grow_policy
        if self.max_leaves is not None:
            extra["max_leaves"] = int(self.max_leaves)

        self.model = XGBRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            gamma=self.gamma,
            min_child_weight=self.min_child_weight,
            tree_method=self.tree_method,
            n_jobs=n_jobs,
            random_state=self.random_state,
            objective=self.objective,
            importance_type=self.importance_type,
            eval_metric=self.eval_metric,
            **extra
        )

    @staticmethod
    def _flatten_1d(y):
        y = np.asarray(y)
        return y.ravel() if (y.ndim == 2 and y.shape[1] == 1) else y

    def _fit_single_with_val(self, X, y, Xv, yv):
        try:
            cbs = [EarlyStopping(rounds=int(self.early_stopping_rounds), save_best=True, maximize=False)]
            trial = getattr(self, "_trial", None)
            if trial is not None:
                cbs.append(XGBoostPruningCallback(trial, f"validation_1-{self.eval_metric}"))
            self.model.fit(
                X, y,
                eval_set=[(X, y), (Xv, yv)],
                callbacks=cbs,
                verbose=False
            )
            return
        except TypeError:
            pass

        # Older XGBoost fallback
        try:
            self.model.fit(
                X, y,
                eval_set=[(X, y), (Xv, yv)],
                early_stopping_rounds=int(self.early_stopping_rounds),
                verbose=False
            )
        except TypeError:
            self.model.fit(X, y, eval_set=[(Xv, yv)], verbose=False)

    def fit(self, X: pd.DataFrame, y) -> XGBoost:
        self._build()
        y1 = self._flatten_1d(y)
        self.model.fit(X, y1)
        return self

    def train(self, X_tr, y_tr, X_val=None, y_val=None) -> XGBoost:
        self._build()
        ytr = self._flatten_1d(y_tr)
        if X_val is not None and y_val is not None and self.early_stopping_rounds > 0:
            yva = self._flatten_1d(y_val)
            self._fit_single_with_val(X_tr, ytr, X_val, yva)
        else:
            self.model.fit(X_tr, ytr)
        return self

    def predict(self, X: pd.DataFrame):
        return np.asarray(self.model.predict(X)).reshape(-1)
