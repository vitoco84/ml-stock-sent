from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge

from src.models.base import Base
from src.models.linreg import LinearElasticNet
from src.models.random_forest import RandomForest
from src.models.xgboost import XGBoost


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
            ("linreg", LinearElasticNet(horizon=self.horizon)),
            ("rf", RandomForest(horizon=self.horizon)),
            ("xgb", XGBoost(horizon=self.horizon))
        ]

        meta = Ridge(alpha=1.0, random_state=self.random_state)
        self.model = StackingRegressor(
            estimators=[(n, m.model) for n, m in base_learners],
            final_estimator=meta,
            passthrough=True,
            n_jobs=-1
        )

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
