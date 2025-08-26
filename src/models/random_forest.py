from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.models.base import Base


@dataclass
class RandomForest(Base):
    """
    RandomForestRegressor with native multi-output support.
    y can be shape (n,) or (n, H)
    """
    name = "random_forest"

    horizon: int = 30
    random_state: int = 42
    n_estimators: int = 800
    max_depth: Optional[int] = None
    min_samples_split: int = 2
    min_samples_leaf: int = 2
    max_features: str | float | int | None = "sqrt"
    bootstrap: bool = True
    n_jobs: int = -1
    max_samples: Optional[int | float] = None

    def __post_init__(self):
        super().__init__(horizon=self.horizon, random_state=self.random_state)
        self._build()

    def _build(self):
        max_samples = self.max_samples if self.bootstrap else None
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            max_samples=max_samples,
            n_jobs=self.n_jobs,
            random_state=self.random_state
        )

    def fit(self, X: pd.DataFrame, y: Any) -> RandomForest:
        self.model.fit(X, np.asarray(y))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        yhat = self.model.predict(X)
        return np.asarray(yhat)

    @staticmethod
    def search_space(trial):
        bootstrap = trial.suggest_categorical("bootstrap", [True, False])
        use_max_depth = trial.suggest_categorical("use_max_depth", [True, False])

        return {
            "n_estimators": trial.suggest_int("n_estimators", 600, 2000, step=200),
            "max_depth": trial.suggest_int("max_depth", 6, 40) if use_max_depth else None,
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical(
                "max_features", ["sqrt", "log2", 1.0, 0.8, 0.5]
            ),
            "bootstrap": bootstrap,
            "max_samples": trial.suggest_float("max_samples", 0.5, 1.0) if bootstrap else None,
        }
