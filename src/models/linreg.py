from __future__ import annotations

from dataclasses import dataclass, field
from typing import Self

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor

from src.models.base import Base

@dataclass
class LinearElasticNet(Base):
    """
    Linear regression with combined L1/L2 regularization (ElasticNet penalty via SGDRegressor).

    Supports:
    - Single target regression
    - Multi-output regression (e.g., horizon=20 vector forecast)
    - Incremental updates via warm_start or partial_fit
    """

    name: str = field(default="linreg", init=False)

    n_jobs: int
    horizon: int
    random_state: int

    alpha: float = 1e-3
    l1_ratio: float = 0.5
    max_iter: int = 1500
    tol: float = 1e-3
    learning_rate: str = "constant"
    eta0: float = 0.01
    penalty: str = "elasticnet"
    multioutput: bool = True

    model: SGDRegressor | MultiOutputRegressor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        super().__init__(horizon=self.horizon, random_state=self.random_state, n_jobs=self.n_jobs)
        self._build()

    def _build(self) -> None:
        base = SGDRegressor(
            penalty=self.penalty,
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
            learning_rate=self.learning_rate,
            eta0=self.eta0,
            warm_start=True
        )
        self.model = MultiOutputRegressor(base, n_jobs=self.n_jobs) if self.multioutput else base

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> Self:
        """Train LinearSGD (single or multi-output)."""
        if not self.multioutput and getattr(y, "ndim", 1) == 2 and y.shape[1] == 1:
            y = np.asarray(y).ravel()
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions."""
        yhat = self.model.predict(X)
        return np.asarray(yhat)

    def fine_tune(self, X: pd.DataFrame, y: np.ndarray) -> Self:
        """Incrementally update model with new stock data."""
        if not self.multioutput and getattr(y, "ndim", 1) == 2 and y.shape[1] == 1:
            y = np.asarray(y).ravel()
        if hasattr(self.model, "partial_fit"):
            self.model.partial_fit(X, y)
        else:
            for est, y_col in zip(self.model.estimators_, y.T):
                est.partial_fit(X, y_col)
        return self

    @staticmethod
    def search_space(trial) -> dict:
        return {
            "alpha": trial.suggest_float("alpha", 1e-5, 1e-2, log=True),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
            "max_iter": trial.suggest_int("max_iter", 1000, 3000, step=500),
            "eta0": trial.suggest_float("eta0", 1e-3, 1e-1, log=True),
            "learning_rate": trial.suggest_categorical(
                "learning_rate", ["constant", "optimal"]
            )
        }
