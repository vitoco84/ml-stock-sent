from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

from src.models.base import Base


@dataclass
class StackingEnsemble(Base):
    """Stacking ensemble of base regressors with Ridge meta-learner."""
    name = "stacking"

    horizon: int = 30
    random_state: int = 42
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
                n_jobs=-1,
                min_samples_leaf=2,
                max_features="sqrt"
            )),
            ("xgb", XGBRegressor(
                n_estimators=800,
                learning_rate=0.05,
                max_depth=5,
                tree_method="hist",
                device="cpu",
                random_state=self.random_state, n_jobs=1
            ))
        ]

        meta = Ridge(alpha=1.0, random_state=self.random_state)
        stack = StackingRegressor(estimators=base_learners, final_estimator=meta, passthrough=True, n_jobs=-1)
        self.model = MultiOutputRegressor(stack)

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> StackingEnsemble:
        self.model.fit(X, np.asarray(y))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        yhat = self.model.predict(X)
        return np.asarray(yhat)

    @staticmethod
    def search_space(trial):
        return {
            "ridge_alpha": trial.suggest_float("ridge_alpha", 1e-3, 10.0, log=True),
        }
