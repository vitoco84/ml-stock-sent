from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator
from torch import nn
from torch.optim import Adam


class Base(ABC, BaseEstimator):
    """Base Class for TimeSeries Regressors."""
    name = "base"

    def __init__(self, horizon: int = 30, random_state: int = 42):
        self.horizon = horizon
        self.random_state = random_state

    @abstractmethod
    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray) -> Base:
        raise NotImplementedError

    @abstractmethod
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def fit_with_val(
            self,
            X_train: pd.DataFrame,
            y_train: np.ndarray,
            X_val: pd.DataFrame,
            y_val: np.ndarray,
    ) -> Base:
        return self.fit(X_train, y_train)

    def train(
            self,
            X_train: pd.DataFrame,
            y_train: np.ndarray,
            X_val: pd.DataFrame = None,
            y_val: np.ndarray = None,
    ) -> Base:
        if X_val is None or y_val is None:
            return self.fit(X_train, y_train)
        return self.fit_with_val(X_train, y_train, X_val, y_val)

    def save(self, path: Union[str, Path]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, p, compress=True)

    @classmethod
    def load(cls, path: Union[str, Path]) -> Base:
        return joblib.load(Path(path))

    @staticmethod
    def _reshape(X: pd.DataFrame) -> torch.Tensor:
        """Used for CNN and LSTM."""
        Xnp = np.asarray(X, dtype=np.float32).reshape((len(X), X.shape[1], 1))
        return torch.from_numpy(Xnp)

class TorchBaseNN(Base):
    """Base class for PyTorch forecasting models."""

    def __init__(self, horizon=30, random_state=42, lr=1e-3, epochs=50, batch_size=64, weight_decay=1e-5, device=None):
        super().__init__(horizon=horizon, random_state=random_state)
        self.horizon = horizon
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._net = None

    @abstractmethod
    def _build_net(self, input_dim: int, output_dim: int) -> nn.Module:
        raise NotImplementedError

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        X_t = self._reshape(X).to(self.device)
        y_t = torch.as_tensor(y, dtype=torch.float32).to(self.device)
        if y_t.ndim == 1:
            y_t = y_t.unsqueeze(1)

        self._net = self._build_net(input_dim=1, output_dim=y_t.shape[1]).to(self.device)

        opt = Adam(self._net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.MSELoss()

        self._net.train()
        for _ in range(self.epochs):
            for i in range(0, len(X_t), self.batch_size):
                xb = X_t[i:i + self.batch_size]
                yb = y_t[i:i + self.batch_size]
                opt.zero_grad()
                loss = loss_fn(self._net(xb), yb)
                loss.backward()
                opt.step()
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._net.eval()
        X_t = self._reshape(X).to(self.device)
        with torch.no_grad():
            return self._net(X_t).cpu().numpy()
