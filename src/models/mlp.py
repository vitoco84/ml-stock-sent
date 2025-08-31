from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor

from src.models.base import Base


@dataclass
class MLP(Base):
    """Feed-forward Multi Layer Perceptron (FF MLP)."""
    name = "mlp"

    horizon: int = 30
    random_state: int = 42
    hidden_layer_sizes: Tuple[int, ...] = (256, 128, 64)
    activation: str = "relu"
    solver: str = "adam"
    alpha: float = 1e-4  # L2
    learning_rate_init: float = 1e-3
    learning_rate: str = "adaptive"
    batch_size: int = 256
    max_iter: int = 2000
    early_stopping: bool = True
    n_iter_no_change: int = 20
    validation_fraction: float = 0.15
    shuffle: bool = False
    tol: float = 1e-5

    def __post_init__(self):
        super().__init__(horizon=self.horizon, random_state=self.random_state)
        self._build()

    def _build(self):
        es = self.early_stopping and (self.solver != "lbfgs")
        self.model = MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            solver=self.solver,
            alpha=self.alpha,
            learning_rate_init=self.learning_rate_init,
            learning_rate=self.learning_rate if self.solver != "lbfgs" else "constant",
            batch_size=self.batch_size if self.solver != "lbfgs" else "auto",
            max_iter=self.max_iter,
            early_stopping=es,
            n_iter_no_change=self.n_iter_no_change if es else 10,
            validation_fraction=self.validation_fraction if es else 0.1,
            shuffle=self.shuffle,
            tol=self.tol,
            random_state=self.random_state,
        )

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> MLP:
        self.model.fit(X, np.asarray(y))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        yhat = self.model.predict(X)
        return np.asarray(yhat)

    @staticmethod
    def search_space(trial):
        layers = trial.suggest_categorical(
            "hidden_layers",
            [
                [128], [256],
                [128, 64], [256, 128],
                [256, 128, 64], [128, 64, 32],
                [512, 256]
            ]
        )
        solver = trial.suggest_categorical("solver", ["adam", "lbfgs"])
        params = {
            "hidden_layer_sizes": tuple(layers),
            "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
            "solver": solver,
            "alpha": trial.suggest_float("alpha", 1e-6, 1e-1, log=True),
            "max_iter": trial.suggest_int("max_iter", 1200, 3000, step=300),
            "tol": trial.suggest_float("tol", 1e-6, 1e-4, log=True)
        }
        if solver == "adam":
            params.update({
                "learning_rate_init": trial.suggest_float("learning_rate_init", 5e-4, 1e-2, log=True),
                "learning_rate": trial.suggest_categorical("learning_rate", ["constant", "adaptive"]),
                "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
                "early_stopping": True,
                "n_iter_no_change": trial.suggest_int("n_iter_no_change", 10, 30),
                "validation_fraction": 0.15,
                "shuffle": False
            })
        else:
            params.update({
                "early_stopping": False,
                "learning_rate_init": 1e-3,
                "learning_rate": "constant",
                "batch_size": "auto",
                "shuffle": False
            })
        return params
