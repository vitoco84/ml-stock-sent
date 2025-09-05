from __future__ import annotations

from dataclasses import dataclass, field
from typing import Self

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.model_selection import BaseCrossValidator
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

from src.models.base import Base


class PartitionedTimeSeriesSplit(BaseCrossValidator):
    """
    Partitioned (non-overlapping) time series split for stacking CV.
    Unlike rolling splits, this partitions the dataset into consecutive chunks.
    """

    def __init__(self, n_splits: int):
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        self.n_splits = n_splits

    def split(self, X: pd.DataFrame, y=None, groups=None):
        n = len(X)
        fold_sizes = np.full(self.n_splits, n // self.n_splits)
        fold_sizes[: n % self.n_splits] += 1
        start = 0
        for f in fold_sizes:
            stop = start + f
            test_idx = np.arange(start, stop)
            train_idx = np.setdiff1d(np.arange(n), test_idx, assume_unique=True)
            yield train_idx, test_idx
            start = stop

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

@dataclass
class StackingEnsemble(Base):
    """Stacking ensemble of base regressors with Ridge meta-learner."""

    name: str = "stacking"

    horizon: int = 30
    random_state: int = 42
    multioutput: bool = True
    cv_n_splits: int = 2
    ridge_alpha: float = 1.0
    n_jobs: int = 1

    # Random Forest
    rf_estimators: int = 50
    rf_max_depth: int | None = 12
    rf_max_samples: float = 0.6

    # XGBoost
    xgb_estimators: int = 80
    xgb_max_depth: int = 5
    xgb_learning_rate: float = 0.08
    xgb_subsample: float = 1.0
    xgb_colsample_bytree: float = 1.0
    xgb_device: str = "cpu"

    drop_rf: bool = False
    drop_lin: bool = False
    passthrough: bool = True

    model: StackingRegressor | MultiOutputRegressor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        super().__init__(horizon=self.horizon, random_state=self.random_state)
        self._build()

    def _build(self) -> None:
        lin = ElasticNet(
            alpha=1e-3,
            l1_ratio=0.2,
            max_iter=800,
            random_state=self.random_state
        )
        rf = RandomForestRegressor(
            n_estimators=self.rf_estimators,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            min_samples_leaf=2,
            max_features="sqrt",
            max_depth=self.rf_max_depth,
            max_samples=self.rf_max_samples
        )
        xgb = XGBRegressor(
            n_estimators=self.xgb_estimators,
            learning_rate=self.xgb_learning_rate,
            max_depth=self.xgb_max_depth,
            subsample=self.xgb_subsample,
            colsample_bytree=self.xgb_colsample_bytree,
            tree_method="hist",
            device=self.xgb_device,
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )

        base_learners: list[tuple[str, object]] = []
        if not self.drop_lin:
            base_learners.append(("lin", lin))
        if not self.drop_rf:
            base_learners.append(("rf", rf))
        base_learners.append(("xgb", xgb))

        cv = PartitionedTimeSeriesSplit(n_splits=self.cv_n_splits)
        meta = Ridge(alpha=self.ridge_alpha, random_state=self.random_state)

        stack = StackingRegressor(
            estimators=base_learners,
            final_estimator=meta,
            passthrough=self.passthrough,
            n_jobs=self.n_jobs,
            cv=cv
        )

        self.model = MultiOutputRegressor(stack, n_jobs=self.n_jobs) if self.multioutput else stack

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> Self:
        y = np.asarray(y)
        if not self.multioutput and y.ndim == 2 and y.shape[1] == 1:
            y = y.ravel()
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        yhat = self.model.predict(X)
        return np.asarray(yhat)

    @staticmethod
    def search_space(trial) -> dict:
        return {
            "ridge_alpha": trial.suggest_float("ridge_alpha", 1e-3, 10.0, log=True),
            "rf_estimators": trial.suggest_int("rf_estimators", 50, 200, step=25),
            "xgb_estimators": trial.suggest_int("xgb_estimators", 60, 200, step=20),
            "xgb_max_depth": trial.suggest_int("xgb_max_depth", 3, 7),
            "xgb_learning_rate": trial.suggest_float("xgb_learning_rate", 0.05, 0.2, log=True),
            "drop_rf": trial.suggest_categorical("drop_rf", [False, True]),
            "drop_lin": trial.suggest_categorical("drop_lin", [False, True]),
            "passthrough": trial.suggest_categorical("passthrough", [True, False])
        }
