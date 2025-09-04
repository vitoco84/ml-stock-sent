from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn

from src.models.basenn import TorchBaseNN


class _MLPNet(nn.Module):
    def __init__(self, input_dim: int, out_dim: int, hidden: Tuple[int, ...], dropout: float, use_bn: bool = True):
        super().__init__()
        layers: list[nn.Module] = [nn.Flatten(start_dim=1)]  # (N, F, 1) -> (N, F)

        # First hidden layer: LazyLinear infers in_features (so we don't care about input_dim=1)
        h0 = hidden[0]
        layers.append(nn.LazyLinear(h0))
        if use_bn:
            layers.append(nn.BatchNorm1d(h0))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        # Remaining hidden layers
        for fin, fout in zip(hidden[:-1], hidden[1:]):
            layers.append(nn.Linear(fin, fout))
            if use_bn:
                layers.append(nn.BatchNorm1d(fout))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        # Output layer
        layers.append(nn.Linear(hidden[-1], out_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

@dataclass
class MLP(TorchBaseNN):
    """Feed-forward Multi Layer Perceptron using tabular data."""
    name: str = "mlp"
    input_mode: str = "tabular"

    horizon: int = 30
    random_state: int = 42
    lr: float = 1e-3
    epochs: int = 100
    batch_size: int = 256
    weight_decay: float = 1e-4
    hidden: Tuple[int, ...] = (256, 128, 64)
    dropout: float = 0.1
    use_bn: bool = True
    patience: int = 20
    min_delta: float = 0.0
    scheduler_patience: int = 5
    clip_grad_norm: float = 1.0
    use_amp: bool = False

    def __post_init__(self):
        super().__init__(horizon=self.horizon, random_state=self.random_state)

    def _build_net(self, input_dim: int, output_dim: int) -> nn.Module:
        return _MLPNet(
            input_dim=input_dim,
            out_dim=output_dim,
            hidden=self.hidden,
            dropout=self.dropout,
            use_bn=self.use_bn
        )

    @staticmethod
    def search_space(trial):
        hidden_key = trial.suggest_categorical(
            "hidden_key",
            ["128", "256",
             "256,128", "128,64",
             "256,128,64", "128,64,32",
             "512,256"]
        )
        layers = tuple(int(x) for x in hidden_key.split(","))
        return {
            "hidden": layers,
            "dropout": trial.suggest_float("dropout", 0.0, 0.3),
            "use_bn": True,
            "lr": trial.suggest_float("lr", 5e-4, 3e-3, log=True),
            "epochs": trial.suggest_int("epochs", 80, 200, step=20),
            "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 5e-4, log=True),
        }
