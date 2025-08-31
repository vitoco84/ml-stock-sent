from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn

from src.models.base import TorchBaseNN


class _LSTMNet(nn.Module):
    def __init__(self, output_dim, units, dense_units, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=units, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(units, dense_units)
        self.fc2 = nn.Linear(dense_units, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.relu(self.fc1(out))
        return self.fc2(out)

@dataclass
class LSTMModel(TorchBaseNN):
    """Long Short-term Memory (LSTM)"""
    name: str = "lstm"

    units: int = 64
    dense_units: int = 64
    dropout: float = 0.2

    def _build_net(self, input_dim: int, output_dim: int) -> nn.Module:
        return _LSTMNet(
            output_dim=output_dim,
            units=self.units,
            dense_units=self.dense_units,
            dropout=self.dropout
        )

    @staticmethod
    def search_space(trial):
        return {
            "units": trial.suggest_int("units", 32, 256, step=32),
            "dense_units": trial.suggest_int("dense_units", 32, 256, step=32),
            "dropout": trial.suggest_float("dropout", 0.1, 0.5),
            "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
            "epochs": trial.suggest_int("epochs", 30, 100, step=10),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        }
