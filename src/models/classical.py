from typing import Any, Dict

import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.multioutput import MultiOutputRegressor

from src.models.base import Base
from src.annotations import tested


@tested
class LinearElasticNet(Base):
    """
    Linear regression with combined L1/L2 regularization (ElasticNet).
    Supports single-target or multi-target (e.g., 30-step vector) via MultiOutputRegressor.
    Use recursive roll-forward outside this class to produce a multi-day path.
    """

    name = "linear_elasticnet"

    def __init__(
            self,
            horizon: int = 30,
            alpha: float = 0.1,
            l1_ratio: float = 0.2,
            random_state: int = 42,
            selection: str = "cyclic",  # 'cyclic' or 'random'
            max_iter: int = 2000,
            multioutput: bool = True
    ):
        super().__init__(horizon=horizon, random_state=random_state)
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.selection = selection
        self.max_iter = max_iter
        self.multioutput = multioutput
        self._build_estimator()

    def _build_estimator(self) -> None:
        base = ElasticNet(
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
            selection=self.selection,
            random_state=self.random_state,
            max_iter=self.max_iter
        )
        self.model = MultiOutputRegressor(base) if self.multioutput else base

    def _build(self) -> None:
        self._build_estimator()

    def fit(self, X_train: pd.DataFrame, y_train: Any) -> "LinearElasticNet":
        if not self.multioutput and y_train.ndim == 2 and y_train.shape[1] == 1:
            y_train = y_train.ravel()
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X_test: pd.DataFrame) -> Any:
        yhat = self.model.predict(X_test)

        if self.multioutput:
            return pd.DataFrame(yhat, columns=[f"target_{i}" for i in range(self.horizon)])
        else:
            return pd.Series(yhat)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return {
            "horizon": self.horizon,
            "random_state": self.random_state,
            "alpha": self.alpha,
            "l1_ratio": self.l1_ratio,
            "selection": self.selection,
            "max_iter": self.max_iter,
            "multioutput": self.multioutput,
        }

    def set_params(self, **params):
        known = {
            "horizon", "random_state",
            "alpha", "l1_ratio", "selection", "max_iter",
            "multioutput",
        }
        for k, v in list(params.items()):
            if k in known:
                setattr(self, k, v)
                params.pop(k)
        self._build_estimator()
        return self

    def suggest_hyperparameters(self, trial):
        return {
            "alpha": trial.suggest_float("alpha", 1e-6, 1.0, log=True),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
            "selection": trial.suggest_categorical("selection", ["cyclic", "random"])
        }
