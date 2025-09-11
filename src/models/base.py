from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Self, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator


class Base(ABC, BaseEstimator):
    """Abstract base class for time series regressors."""

    name = "base"

    def __init__(self, horizon: int = 20, random_state: int = 42):
        self.horizon = horizon
        self.random_state = random_state

    @abstractmethod
    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray) -> Self:
        """Train the model on provided data."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """Generate predictions on the test data."""
        raise NotImplementedError

    def fit_with_val(
            self,
            X_train: pd.DataFrame,
            y_train: np.ndarray,
            X_val: pd.DataFrame,
            y_val: np.ndarray
    ) -> Self:
        """Optional validation-aware training method."""
        return self.fit(X_train, y_train)

    def train(
            self,
            X_train: pd.DataFrame,
            y_train: np.ndarray,
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[np.ndarray] = None
    ) -> Self:
        """Train the model, with or without validation data."""
        if X_val is None or y_val is None:
            return self.fit(X_train, y_train)
        return self.fit_with_val(X_train, y_train, X_val, y_val)

    def save(self, path: Union[str, Path], compress: int | bool = True) -> None:
        """Save the trained model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path, compress=compress)

    @classmethod
    def load(cls, path: Union[str, Path]) -> Self:
        """Load a model from disk."""
        model = joblib.load(Path(path))
        if not isinstance(model, cls):
            raise TypeError(f"Loaded object is not of type {cls.__name__}")
        return model
