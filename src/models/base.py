import os
from abc import ABC, abstractmethod

import joblib
import numpy as np


class Base(ABC):
    """Base class for all models."""

    name = 'base_model'

    def __init__(self, horizon: int = 30, random_state: int = 42, **kwargs):
        self.horizon = horizon
        self.random_state = random_state

    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "Base":
        raise NotImplementedError

    @abstractmethod
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "Base":
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        return joblib.load(path)

    def __repr__(self):
        attrs = vars(self)
        attr_str = ", ".join(f"{k}={v}" for k, v in attrs.items() if not k.startswith("_"))
        return f"{self.__class__.__name__}({attr_str})"
