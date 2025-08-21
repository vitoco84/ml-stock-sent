from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Union

import joblib
import optuna
import pandas as pd


class Base(ABC):
    """Base class for all models."""
    name = 'base_model'

    def __init__(self, horizon: int = 30, random_state: int = 42):
        self.horizon = horizon
        self.random_state = random_state

    @abstractmethod
    def fit(self, X_train: pd.DataFrame, y_train: Any) -> "Base":
        raise NotImplementedError

    @abstractmethod
    def predict(self, X_test: pd.DataFrame) -> Any:
        raise NotImplementedError

    @abstractmethod
    def suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: Any, X_val: pd.DataFrame = None,
              y_val: pd.DataFrame = None) -> "Base":
        raise NotImplementedError

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return {"horizon": self.horizon, "random_state": self.random_state}

    def save(self, path: Union[str, Path]) -> None:
        p = Path(path)
        if p.parent and not p.parent.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, p, compress=True)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Base":
        return joblib.load(Path(path))

    def __repr__(self):
        return f"{self.__class__.__name__}(horizon={self.horizon}, random_state={self.random_state})"
