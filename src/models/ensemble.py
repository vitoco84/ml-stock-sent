from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Self

import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from src.models.base import Base
from src.models.linreg import LinearElasticNet
from src.models.random_forest import RandomForest
from src.models.xgboost import XGBoost


@dataclass
class Ensemble(Base):
    """
    Simple ensemble of Elastic Net, Random Forest, and XGBoost.
    """

    name: str = field(default="ensemble", init=False)
    n_jobs: int
    horizon: int
    random_state: int

    models: List[Base] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        super().__init__(horizon=self.horizon, random_state=self.random_state, n_jobs=self.n_jobs)
        self._build()

    def _build(self) -> None:
        self.models = [
            LinearElasticNet(horizon=self.horizon, random_state=self.random_state, n_jobs=self.n_jobs),
            RandomForest(horizon=self.horizon, random_state=self.random_state, n_jobs=self.n_jobs),
            XGBoost(horizon=self.horizon, random_state=self.random_state, n_jobs=self.n_jobs)
        ]

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> Self:
        fitted = 0
        for m in self.models:
            try:
                m.fit(X, y)
                fitted += 1
            except Exception:
                continue
        if fitted == 0:
            raise RuntimeError("Ensemble.fit(): all submodels failed to train.")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = []
        for m in self.models:
            try:
                check_is_fitted(m.model if hasattr(m, "model") else m)
                preds.append(m.predict(X))
            except (NotFittedError, Exception):
                continue
        if not preds:
            raise RuntimeError("Ensemble.predict(): all submodels failed to predict.")
        return np.mean(preds, axis=0)

    @staticmethod
    def search_space(trial) -> dict:
        return {}
