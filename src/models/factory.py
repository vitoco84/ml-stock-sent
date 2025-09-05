from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from src.models.base import Base
from src.models.cnn import CNNModel
from src.models.linreg import LinearElasticNet
from src.models.lstm import LSTMModel
from src.models.mlp import MLP
from src.models.random_forest import RandomForest
from src.models.stacking import StackingEnsemble
from src.models.xgboost import XGBoost


class ModelBuilder(Protocol):
    """Protocol for experiment model builders."""

    def __call__(self, horizon: int, seed: int) -> Base: ...

@dataclass(frozen=True, slots=True)
class Experiment:
    """Factory container for experiment setup."""
    name: str
    build: ModelBuilder
    include_sentiment: bool

# Default experiment registry
EXPERIMENTS: list[Experiment] = [
    Experiment(
        name="linreg",
        build=lambda horizon, seed: LinearElasticNet(
            horizon=horizon, random_state=seed, multioutput=True
        ),
        include_sentiment=True
    ),
    Experiment(
        name="linreg_wo_sent",
        build=lambda horizon, seed: LinearElasticNet(
            horizon=horizon, random_state=seed, multioutput=True
        ),
        include_sentiment=False
    ),
    Experiment(
        name="xgboost",
        build=lambda horizon, seed: XGBoost(random_state=seed),
        include_sentiment=True
    ),
    Experiment(
        name="random_forest",
        build=lambda horizon, seed: RandomForest(horizon=horizon, random_state=seed),
        include_sentiment=True
    ),
    Experiment(
        name="mlp",
        build=lambda horizon, seed: MLP(horizon=horizon, random_state=seed),
        include_sentiment=True
    ),
    Experiment(
        name="cnn",
        build=lambda horizon, seed: CNNModel(horizon=horizon, random_state=seed),
        include_sentiment=True
    ),
    Experiment(
        name="lstm",
        build=lambda horizon, seed: LSTMModel(horizon=horizon, random_state=seed),
        include_sentiment=True
    ),
    Experiment(
        name="stacking",
        build=lambda horizon, seed: StackingEnsemble(horizon=horizon, random_state=seed),
        include_sentiment=True
    )
]
