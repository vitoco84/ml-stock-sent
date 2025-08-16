from typing import Any, Dict

import numpy as np
from sklearn.linear_model import ElasticNet

from src.models.base import Base


class LinearElasticNet(Base):
    """
    Linear regression with combined L1 and L2 priors as regularizer.
    Multi target regression.
    """

    name = "linear_elasticnet"

    def __init__(self, horizon: int = 30, alpha: float = 0.1, l1_ratio: float = 0.2, random_state: int = 42, **kwargs):
        super().__init__(horizon, random_state, **kwargs)

        # For Multi Target
        # base = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=random_state, max_iter=2000)
        # self.model = MultiOutputRegressor(base)
        self.model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=random_state, max_iter=2000)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "LinearElasticNet":
        self.logger.info("Starting model training...")
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return self.model.predict(X_test)

    def suggest_hyperparameters(self, trial) -> Dict[str, Any]:
        return {
            "alpha": trial.suggest_float("alpha", 1e-4, 10.0, log=True),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0)
        }
