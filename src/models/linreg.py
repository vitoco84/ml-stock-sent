from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.multioutput import MultiOutputRegressor

from src.models.base import Base


@dataclass
class LinearElasticNet(Base):
    """
    Linear regression with combined L1/L2 regularization (ElasticNet).
    Supports single-target or multi-target (e.g., 30-step vector) via MultiOutputRegressor.
    """
    name = "linreg"

    horizon: int = 30
    random_state: int = 42

    alpha: float = 1e-3
    l1_ratio: float = 0.2
    selection: str = "cyclic"
    max_iter: int = 2000

    multioutput: bool = True

    def __post_init__(self):
        self._build()

    def _build(self):
        base = ElasticNet(
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
            selection=self.selection,
            random_state=self.random_state,
            max_iter=self.max_iter,
        )
        self.model = MultiOutputRegressor(base) if self.multioutput else base

    def fit(self, X: pd.DataFrame, y: Any) -> LinearElasticNet:
        if not self.multioutput and getattr(y, "ndim", 1) == 2 and y.shape[1] == 1:
            y = np.asarray(y).ravel()
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> Any:
        yhat = self.model.predict(X)
        if self.multioutput and getattr(yhat, "ndim", 1) == 2:
            return pd.DataFrame(yhat, columns=[f"target_{i}" for i in range(yhat.shape[1])])
        return pd.Series(yhat)
