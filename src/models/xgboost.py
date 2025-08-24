from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd
from optuna.integration import XGBoostPruningCallback
from xgboost.callback import EarlyStopping

from src.models.base import Base
from xgboost import XGBRegressor


@dataclass
class XGBoost(Base):
    """
    XGBoost with native multi-step support (H>1 trains one estimator per horizon).
    Early stopping + Optuna pruning per step. All params are fixed (not optional).
    """
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

    device: str = "cpu"  # "cuda" for GPU
    max_bin: int = 256
    grow_policy: str = "depthwise"
    max_leaves: int = 0

    early_stopping_rounds: int = 50

    horizon: int = 1
    multioutput: bool = False

    _single: Optional[XGBRegressor] = field(default=None, init=False, repr=False)
    _multi: Optional[List[XGBRegressor]] = field(default=None, init=False, repr=False)

    def _new_estimator(self) -> XGBRegressor:
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
            tree_method=self.tree_method,
            n_jobs=self.n_jobs if self.device != "cuda" else 1,
            random_state=self.random_state,
            objective=self.objective,
            importance_type=self.importance_type,
            eval_metric=self.eval_metric,
            device=self.device,
            max_bin=int(self.max_bin),
            grow_policy=self.grow_policy,
            max_leaves=int(self.max_leaves),
        )

    @staticmethod
    def _as_2d(y) -> np.ndarray:
        y = np.asarray(y)
        return y.reshape(-1, 1) if y.ndim == 1 else y

    def fit(self, X: pd.DataFrame, y) -> XGBoost:
        Y = self._as_2d(y)
        if Y.shape[1] == 1:
            self._single, self._multi = self._new_estimator(), None
            self._single.fit(X, Y.ravel())
        else:
            self._single, self._multi = None, []
            for j in range(Y.shape[1]):
                m = self._new_estimator()
                self._multi.append(m)
                m.fit(X, Y[:, j])
        return self

    def _fit_with_val_single(self, model: XGBRegressor, X, y, Xv, yv):
        try:
            cbs = [EarlyStopping(rounds=int(self.early_stopping_rounds), save_best=True, maximize=False)]
            trial = getattr(self, "_trial", None)
            if trial is not None:
                cbs.append(XGBoostPruningCallback(trial, f"validation_1-{self.eval_metric}"))
            model.fit(X, y, eval_set=[(X, y), (Xv, yv)], callbacks=cbs, verbose=False)
            return
        except TypeError:
            pass
        try:
            model.fit(
                X, y,
                eval_set=[(X, y), (Xv, yv)],
                early_stopping_rounds=int(self.early_stopping_rounds),
                verbose=False
            )
            return
        except TypeError:
            pass
        model.fit(X, y)

    def train(self, X_tr, y_tr, X_val=None, y_val=None) -> XGBoost:
        Ytr = self._as_2d(y_tr)
        if X_val is None or y_val is None or self.early_stopping_rounds <= 0:
            return self.fit(X_tr, y_tr)

        Yva = self._as_2d(y_val)
        if Ytr.shape[1] == 1:
            self._single, self._multi = self._new_estimator(), None
            self._fit_with_val_single(self._single, X_tr, Ytr.ravel(), X_val, Yva.ravel())
        else:
            self._single, self._multi = None, []
            for j in range(Ytr.shape[1]):
                m = self._new_estimator()
                self._multi.append(m)
                self._fit_with_val_single(m, X_tr, Ytr[:, j], X_val, Yva[:, j])
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._single is not None:
            return np.asarray(self._single.predict(X)).reshape(-1)
        if self._multi:
            return np.column_stack([m.predict(X) for m in self._multi])
        raise RuntimeError("XGBoost: call fit/train before predict.")

    @staticmethod
    def search_space(trial):
        space = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 1500, step=200),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.10, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 10.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 5.0),
            "early_stopping_rounds": 100,
            "max_bin": trial.suggest_int("max_bin", 128, 512, step=64),
            "grow_policy": trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"]),
            "objective": "reg:squarederror",
            "tree_method": "hist",
        }
        if space["grow_policy"] == "lossguide":
            space["max_leaves"] = trial.suggest_int("max_leaves", 64, 512, step=64)
        else:
            space["max_leaves"] = 0
        return space
