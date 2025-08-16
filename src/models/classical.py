import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.multioutput import MultiOutputRegressor

from src.models.base import Base


class LinearElasticNetForecaster(Base):
    """
    Linear regression with combined L1 and L2 priors as regularizer.
    Multi target regression.
    """

    name = "linear_elasticnet"

    def __init__(self, horizon: int = 30, alpha: float = 0.1, l1_ratio: float = 0.2, random_state: int = 42, **kwargs):
        super().__init__(horizon, random_state, **kwargs)

        base = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=random_state, max_iter=2000)
        self.model = MultiOutputRegressor(base)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "LinearElasticNetForecaster":
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return self.model.predict(X_test)
