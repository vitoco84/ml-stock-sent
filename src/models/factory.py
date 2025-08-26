from dataclasses import dataclass
from typing import Any, Callable

from src.models.linreg import LinearElasticNet
from src.models.random_forest import RandomForest
from src.models.xgboost import XGBoost


@dataclass
class Experiment:
    """Factory Class for experiments."""
    name: str
    build: Callable[[int, int], Any]
    include_sentiment: bool

# e.g.: how to use in the Jupyter Notebooks for the run_experiments method
experiments = [
    Experiment(
        name="linreg",
        build=lambda horizon, seed: LinearElasticNet(horizon=horizon, random_state=seed, multioutput=True),
        include_sentiment=True
    ),
    Experiment(
        name="linreg_wo_sent",
        build=lambda horizon, seed: LinearElasticNet(horizon=horizon, random_state=seed, multioutput=True),
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
    )
]
