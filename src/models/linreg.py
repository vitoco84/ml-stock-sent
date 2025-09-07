from __future__ import annotations

from dataclasses import dataclass, field
from typing import Self

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.multioutput import MultiOutputRegressor

from src.models.base import Base


@dataclass
class LinearElasticNet(Base):
    """
    Linear regression with combined L1/L2 regularization (ElasticNet).

    Supports:
    - Single target regression
    - Multi-output regression (e.g., horizon=30 vector forecast)
    """

    name = "linreg"

    n_jobs = -1
    horizon: int = 30
    random_state: int = 42
    alpha: float = 1e-3
    l1_ratio: float = 0.2
    selection: str = "cyclic"
    max_iter: int = 2000
    multioutput: bool = True

    model: ElasticNet | MultiOutputRegressor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        super().__init__(horizon=self.horizon, random_state=self.random_state)
        self._build()

    def _build(self) -> None:
        base = ElasticNet(
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
            selection=self.selection,
            random_state=self.random_state,
            max_iter=self.max_iter
        )
        self.model = MultiOutputRegressor(base, n_jobs=self.n_jobs) if self.multioutput else base

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> Self:
        """Train ElasticNet (single or multi-output)."""
        if not self.multioutput and getattr(y, "ndim", 1) == 2 and y.shape[1] == 1:
            y = np.asarray(y).ravel()
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions."""
        yhat = self.model.predict(X)
        return np.asarray(yhat)

    @staticmethod
    def search_space(trial) -> dict:
        return {
            "alpha": trial.suggest_float("alpha", 1e-5, 1e-2, log=True),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
            "max_iter": trial.suggest_int("max_iter", 1000, 4000, step=500),
            "selection": trial.suggest_categorical("selection", ["cyclic"])
        }
