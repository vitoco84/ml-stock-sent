from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from config.config import Config
from src.models.base import Base
from src.models.cnn import CNNModel
from src.models.linreg import LinearElasticNet
from src.models.lstm import LSTMModel
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

experiments = [
    Experiment(
        name="linreg",
        build=lambda horizon, seed, n_jobs: LinearElasticNet(
            horizon=HORIZON,
            random_state=SEED,
            n_jobs=N_JOBS,
            multioutput=True
        ),
        include_sentiment=True
    ),
    Experiment(
        name="linreg_wo_sent",
        build=lambda horizon, seed, n_jobs: LinearElasticNet(
            horizon=HORIZON,
            random_state=SEED,
            n_jobs=N_JOBS,
            multioutput=True
        ),
        include_sentiment=False
    ),
    Experiment(
        name="xgboost",
        build=lambda horizon, seed, n_jobs: XGBoost(
            horizon=HORIZON,
            random_state=SEED,
            n_jobs=N_JOBS
        ),
        include_sentiment=True,
    ),
    Experiment(
        name="cnn",
        build=lambda horizon, seed, n_jobs: CNNModel(
            horizon=HORIZON,
            random_state=SEED,
            n_jobs=N_JOBS
        ),
        include_sentiment=True
    ),
    Experiment(
        name="lstm",
        build=lambda horizon, seed, n_jobs: LSTMModel(
            horizon=HORIZON,
            random_state=SEED,
            n_jobs=N_JOBS
        ),
        include_sentiment=True
    )
]
