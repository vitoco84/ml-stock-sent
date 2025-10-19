from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Type

from config.config import Config
from src.models.base import Base
from src.models.linreg import LinearElasticNet
from src.models.lstm import LSTMModel
from src.models.random_forest import RandomForest
from src.models.xgboost import XGBoost


cfg = Config.load()
SEED = cfg.runtime.seed
HORIZON = cfg.runtime.horizon
N_JOBS = cfg.runtime.n_jobs

class ModelBuilder(Protocol):
    """Protocol for experiment model builders."""

    def __call__(self, horizon: int, seed: int, n_jobs: int | None) -> Base: ...

@dataclass(frozen=True, slots=True)
class Experiment:
    """Factory container for experiment setup."""
    name: str
    build: ModelBuilder
    include_sentiment: bool

def make_builder(model_cls: Type[Base]) -> ModelBuilder:
    return lambda horizon, seed, n_jobs: model_cls(
        horizon=HORIZON,
        random_state=SEED,
        n_jobs=N_JOBS,
    )

base_models: dict[str, Type[Base]] = {
    "linreg": LinearElasticNet,
    "xgboost": XGBoost,
    "random_forest": RandomForest,
    "lstm": LSTMModel
}

experiments: list[Experiment] = []
for name, cls in base_models.items():
    builder = make_builder(cls)
    experiments.append(Experiment(name=name, build=builder, include_sentiment=True))
    experiments.append(Experiment(name=f"{name}_wo_sent", build=builder, include_sentiment=False))
