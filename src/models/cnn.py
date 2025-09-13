from __future__ import annotations

from dataclasses import dataclass, field

import torch.nn as nn
from torch import Tensor

from src.models.basenn import TorchBaseNN


class _CNNNet(nn.Module):
    """2-layer 1D CNN for time series forecasting."""

    def __init__(self, output_dim: int, filters: int, kernel_size: int, dense_units: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=filters, kernel_size=kernel_size)
        self.conv2 = nn.Conv1d(in_channels=filters, out_channels=filters, kernel_size=max(2, kernel_size // 2))
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Linear(filters, dense_units)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dense_units, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        # input (N, T, 1) -> Conv1d expects (N, C, T)
        x = x.transpose(1, 2)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

@dataclass
class CNNModel(TorchBaseNN):
    """2-layer Convolutional Neural Network (CNN) forecaster."""

    name: str = field(default="cnn", init=False)
    input_mode: str = field(default="sequence", init=False)

    n_jobs: int
    horizon: int
    random_state: int

    filters: int = 16
    kernel_size: int = 3
    dense_units: int = 32
    dropout: float = 0.1
    lr: float = 1e-3
    epochs: int = 200
    batch_size: int = 32
    weight_decay: float = 1e-5
    patience: int = 10
    min_delta: float = 1e-4
    scheduler_patience: int = 5
    clip_grad_norm: float = 1.0
    use_amp: bool = False

    def __post_init__(self) -> None:
        super().__init__(horizon=self.horizon, random_state=self.random_state, n_jobs=self.n_jobs)

    def _build_net(self, input_dim: int, output_dim: int) -> nn.Module:
        return _CNNNet(
            output_dim=output_dim,
            filters=self.filters,
            kernel_size=self.kernel_size,
            dense_units=self.dense_units,
            dropout=self.dropout
        )

    @staticmethod
    def search_space(trial) -> dict:
        return {
            "filters": trial.suggest_int("filters", 8, 64, step=8),
            "kernel_size": trial.suggest_int("kernel_size", 2, 5),
            "dense_units": trial.suggest_int("dense_units", 16, 64, step=16),
            "dropout": trial.suggest_float("dropout", 0.0, 0.3),
            "lr": trial.suggest_float("lr", 5e-4, 3e-3, log=True),
            "epochs": trial.suggest_int("epochs", 20, 100, step=20),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32]),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)
        }
