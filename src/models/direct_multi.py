from __future__ import annotations

from typing import Any, List, Type

import numpy as np
import pandas as pd

from src.models.base import Base


class DirectMultiStep(Base):
    """
    Direct Multi-Step Wrapper for:
    - XGBoost
    - Random Forest
    Train one Single-Target model per horizon step and stacks predictions.
    """
    name = "direct_multi"

    def __init__(
            self,
            base_cls: Type[Base],
            base_params: dict[str, Any] | None = None,
            horizon: int = 30,
            random_state: int = 42
    ):
        super().__init__(horizon=horizon, random_state=random_state)
        self.base_cls = base_cls
        self.base_params = {**(base_params or {}), "random_state": random_state}
        self.name = getattr(base_cls, "name", base_cls.__name__.lower())
        self.models: List[Base] = []

    @staticmethod
    def _to_2d(y) -> np.ndarray:
        y = np.asarray(y)
        return y.reshape(-1, 1) if y.ndim == 1 else y

    def fit(self, X_train: pd.DataFrame, y_train: Any) -> DirectMultiStep:
        Y = self._to_2d(y_train)
        self.models = []
        for i in range(Y.shape[1]):
            m = self.base_cls(**self.base_params)
            if hasattr(self, "_trial"):
                setattr(m, "_trial", getattr(self, "_trial"))
            m.fit(X_train, Y[:, i])
            self.models.append(m)
        return self

    def train(self, X_tr, y_tr, X_val=None, y_val=None) -> DirectMultiStep:
        if X_val is not None and y_val is not None:
            return self.fit_with_val(X_tr, y_tr, X_val, y_val)
        return self.fit(X_tr, y_tr)

    def fit_with_val(self, X_train, y_train, X_val, y_val) -> DirectMultiStep:
        Ytr, Yva = self._to_2d(y_train), self._to_2d(y_val)
        self.models = []
        for i in range(Ytr.shape[1]):
            y_tr_i = pd.DataFrame(
                Ytr[:, i],
                index=getattr(X_train, "index", None),
                columns=[f"target_{i}"],
            )
            y_va_i = pd.DataFrame(
                Yva[:, i],
                index=getattr(X_val, "index", None),
                columns=[f"target_{i}"],
            )
            m = self.base_cls(**self.base_params)
            if hasattr(self, "_trial"):
                setattr(m, "_trial", getattr(self, "_trial"))
            if hasattr(m, "train"):
                m.train(X_train, y_tr_i, X_val, y_va_i)
            elif hasattr(m, "fit_with_val"):
                m.fit_with_val(X_train, y_tr_i, X_val, y_va_i)
            else:
                m.fit(X_train, y_tr_i)

            self.models.append(m)
        return self

    def predict(self, X_test: pd.DataFrame):
        preds = np.column_stack([m.predict(X_test) for m in self.models])
        return pd.DataFrame(preds, columns=[f"target_{i}" for i in range(preds.shape[1])])
