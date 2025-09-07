from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Self

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.models.base import Base


@dataclass
class RandomForest(Base):
    """
    Wrapper around sklearn's RandomForestRegressor.
    Supports single-output or multi-output regression.
    """

    name: str = "random_forest"

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
    criterion: str = "squared_error"

    model: RandomForestRegressor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        super().__init__(horizon=self.horizon, random_state=self.random_state)
        self._build()

    def _build(self) -> None:
        """Initialize the underlying sklearn RandomForest model."""
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            max_samples=self.max_samples if self.bootstrap else None,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            criterion=self.criterion
        )

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> Self:
        self.model.fit(X, np.asarray(y))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        yhat = self.model.predict(X)
        return np.asarray(yhat)

    @staticmethod
    def search_space(trial) -> dict:
        bootstrap = trial.suggest_categorical("bootstrap", [True, False])

        max_depth = trial.suggest_int("max_depth", 6, 20)
        if trial.suggest_categorical("use_max_depth", [True, False]) is False:
            max_depth = None

        return {
            "n_estimators": trial.suggest_int("n_estimators", 300, 1200, step=300),
            "max_depth": max_depth,
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5, 0.8]),
            "bootstrap": bootstrap,
            "max_samples": trial.suggest_float("max_samples", 0.5, 0.9) if bootstrap else None,
            "criterion": trial.suggest_categorical("criterion", ["squared_error", "absolute_error"]),
        }
