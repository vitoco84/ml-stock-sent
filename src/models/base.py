from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator


class Base(ABC, BaseEstimator):
    """Base Class Interface for TimeSeries Regressors."""
    name = "base"

    def __init__(self, horizon: int = 30, random_state: int = 42):
        self.horizon = horizon
        self.random_state = random_state

    @abstractmethod
    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray) -> Base:
        raise NotImplementedError

    @abstractmethod
    def predict(self, X_test: pd.DataFrame) -> Any:
        raise NotImplementedError

    def fit_with_val(
            self,
            X_train: pd.DataFrame,
            y_train: np.ndarray,
            X_val: pd.DataFrame,
            y_val: np.ndarray,
    ) -> Base:
        return self.fit(X_train, y_train)

    def train(
            self,
            X_train: pd.DataFrame,
            y_train: np.ndarray,
            X_val: pd.DataFrame = None,
            y_val: np.ndarray = None,
    ) -> Base:
        if X_val is None or y_val is None:
            return self.fit(X_train, y_train)
        return self.fit_with_val(X_train, y_train, X_val, y_val)

    def save(self, path: Union[str, Path]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, p, compress=True)

    @classmethod
    def load(cls, path: Union[str, Path]) -> Base:
        return joblib.load(Path(path))
