from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn

from src.models.base import TorchBaseNN


class _CNNNet(nn.Module):
    def __init__(self, output_dim, filters, kernel_size, dense_units, dropout):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=1, out_channels=filters, kernel_size=kernel_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(filters, dense_units)
        self.fc2 = nn.Linear(dense_units, output_dim)

    def forward(self, x):
        x = x.transpose(1, 2)
        out = self.conv(x)
        out = out.mean(dim=2)
        out = self.dropout(self.relu(self.fc1(out)))
        return self.fc2(out)

@dataclass
class CNNModel(TorchBaseNN):
    """Convolutional Neural Network (CNN)."""
    name: str = "cnn"

    horizon: int = 30
    random_state: int = 42
    filters: int = 64
    kernel_size: int = 3
    dense_units: int = 64
    dropout: float = 0.2
    lr: float = 1e-3
    epochs: int = 50
    batch_size: int = 64
    weight_decay: float = 1e-5

    def __post_init__(self):
        super().__init__(horizon=self.horizon, random_state=self.random_state)

    def _build_net(self, input_dim: int, output_dim: int) -> nn.Module:
        return _CNNNet(
            output_dim=output_dim,
            filters=self.filters,
            kernel_size=self.kernel_size,
            dense_units=self.dense_units,
            dropout=self.dropout
        )

    @staticmethod
    def search_space(trial):
        return {
            "filters": trial.suggest_int("filters", 32, 256, step=32),
            "kernel_size": trial.suggest_int("kernel_size", 2, 5),
            "dense_units": trial.suggest_int("dense_units", 32, 256, step=32),
            "dropout": trial.suggest_float("dropout", 0.1, 0.5),
            "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
            "epochs": trial.suggest_int("epochs", 30, 100, step=10),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        }
