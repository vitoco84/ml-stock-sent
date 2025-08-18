from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import optuna


class Base(ABC):
    """Base class for all models."""
    name = 'base_model'

    def __init__(self, horizon: int = 30, random_state: int = 42):
        self.horizon = horizon
        self.random_state = random_state

    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "Base":
        raise NotImplementedError

    @abstractmethod
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        raise NotImplementedError

    def train(self, X_train, y_train, X_val=None, y_val=None):
        return self.fit(X_train, y_train)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return {"horizon": self.horizon, "random_state": self.random_state}

    def save(self, path: Path) -> None:
        joblib.dump(self, Path(path))

    @classmethod
    def load(cls, path: Path) -> "Base":
        return joblib.load(Path(path))

    def __repr__(self):
        return f"{self.__class__.__name__}(horizon={self.horizon}, random_state={self.random_state})"
