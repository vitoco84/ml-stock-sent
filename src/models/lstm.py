from __future__ import annotations

from dataclasses import dataclass, field

import torch.nn as nn
from torch import Tensor

from src.models.basenn import TorchBaseNN


class _LSTMNet(nn.Module):
    """LSTM forecaster head."""

    def __init__(
            self,
            output_dim: int,
            units: int,
            dense_units: int,
            dropout: float,
            bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=units, batch_first=True, bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(units * (2 if bidirectional else 1), dense_units),
            nn.ReLU(),
            nn.Linear(dense_units, output_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for name, p in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(p)
            elif "weight_hh" in name:
                nn.init.orthogonal_(p)
            elif "bias" in name:
                nn.init.zeros_(p)
        for m in self.head:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        return self.head(out)

@dataclass
class LSTMModel(TorchBaseNN):
    """Long Short-Term Memory (LSTM) forecaster."""

    name: str = field(default="lstm", init=False)
    input_mode: str = field(default="sequence", init=False)

    n_jobs: int
    horizon: int
    random_state: int

    bidirectional: bool = False

    units: int = 32
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
        return _LSTMNet(
            output_dim=output_dim,
            units=self.units,
            dense_units=self.dense_units,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
        )

    @staticmethod
    def search_space(trial) -> dict:
        return {
            "units": trial.suggest_categorical("units", [64, 128, 256]),
            "dense_units": trial.suggest_categorical("dense_units", [32, 64, 128]),
            "dropout": trial.suggest_float("dropout", 0.0, 0.15),
            "lr": trial.suggest_float("lr", 1e-5, 5e-4, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-7, 1e-4, log=True),
            "epochs": trial.suggest_int("epochs", 300, 600, step=100),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
            "bidirectional": trial.suggest_categorical("bidirectional", [False, True])
        }
