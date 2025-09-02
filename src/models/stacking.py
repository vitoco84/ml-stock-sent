from __future__ import annotations

from dataclasses import dataclass
from typing import Generator, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.model_selection import BaseCrossValidator
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

from src.models.base import Base


@dataclass
class StackingEnsemble(Base):
    """Stacking ensemble of base regressors with Ridge meta-learner."""
    name = "stacking"

    horizon: int = 30
    random_state: int = 42
    cv_n_splits: int = 5
    cv_gap: Optional[int] = None
    ridge_alpha: float = 1.0
    n_jobs: int = 1
    multioutput: bool = True

    def __post_init__(self):
        super().__init__(horizon=self.horizon, random_state=self.random_state)
        self._build()

    def _build(self):
        base_learners = [
            ("lin", ElasticNet(
                alpha=1e-3,
                l1_ratio=0.2,
                max_iter=2000,
                random_state=self.random_state
            )),
            ("rf", RandomForestRegressor(
                n_estimators=600,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                min_samples_leaf=2,
                max_features="sqrt"
            )),
            ("xgb", XGBRegressor(
                n_estimators=800,
                learning_rate=0.05,
                max_depth=5,
                tree_method="hist",
                random_state=self.random_state,
                n_jobs=max(1, self.n_jobs if self.n_jobs != -1 else 0)
            ))
        ]

        cv = PartitionedTimeSeriesSplit(n_splits=self.cv_n_splits)
        meta = Ridge(alpha=self.ridge_alpha, random_state=self.random_state)

        stack = StackingRegressor(
            estimators=base_learners,
            final_estimator=meta,
            passthrough=True,
            n_jobs=self.n_jobs,
            cv=cv
        )

        self.model = MultiOutputRegressor(stack) if self.multioutput else stack

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> StackingEnsemble:
        self.model.fit(X, np.asarray(y))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        yhat = self.model.predict(X)
        return np.asarray(yhat)

    @staticmethod
    def search_space(trial):
        return {
            "ridge_alpha": trial.suggest_float("ridge_alpha", 1e-3, 10.0, log=True)
        }

class PartitionedTimeSeriesSplit(BaseCrossValidator):
    """Partition Time Series Split for Stacking CV Issues."""
    def __init__(self, n_splits: int):
        self.n_splits = n_splits

    def split(
            self,
            X: pd.DataFrame,
            y: Optional[np.ndarray] = None,
            groups: Optional[np.ndarray] = None
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        n_samples = len(X)
        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits)
        fold_sizes[:n_samples % self.n_splits] += 1

        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            test_idx = np.arange(start, stop)
            train_idx = np.setdiff1d(np.arange(n_samples), test_idx)
            yield train_idx, test_idx
            current = stop

    def get_n_splits(
            self,
            X: Optional[pd.DataFrame] = None,
            y: Optional[np.ndarray] = None,
            groups: Optional[np.ndarray] = None
    ) -> int:
        return self.n_splits
