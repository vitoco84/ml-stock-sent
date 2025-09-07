from __future__ import annotations

from abc import abstractmethod
from typing import Iterator, Optional, Self, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch import nn, Tensor
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.models.base import Base


class TorchBaseNN(Base):
    """
    Base class for PyTorch models (CNN, MLP, LSTM).

    Features:
    - input_mode: "tabular" => (N, F, 1), "sequence" => (N, T, 1) from lag_* columns
    - Training: AMP, grad-clipping, ReduceLROnPlateau, early stopping
    """

    input_mode: str = "tabular"  # "tabular" or "sequence" (lags-only)

    def __init__(self, horizon: int = 30, random_state: int = 42, device: Optional[str] = None) -> None:
        super().__init__(horizon=horizon, random_state=random_state)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._net: nn.Module | None = None

    @abstractmethod
    def _build_net(self, input_dim: int, output_dim: int) -> nn.Module:
        """Return an nn.Module mapping (N, D, 1) -> (N, output_dim)."""
        raise NotImplementedError

    def fit(
            self,
            X: pd.DataFrame,
            y: np.ndarray,
            lr: Optional[float] = None,
            epochs: Optional[int] = None,
            batch_size: Optional[int] = None,
            weight_decay: Optional[float] = None,
            loss_fn: Optional[nn.Module] = None
    ) -> Self:
        hp = self._resolve_hparams(lr, epochs, batch_size, weight_decay, loss_fn)
        X_t, y_t = self._to_tensor(X), self._to_target(y)
        input_dim = X.shape[1] if self.input_mode == "tabular" else 1
        self._init_net(output_dim=y_t.shape[1], input_dim=input_dim)
        self._train_loop(X_t, y_t, None, None, hp)
        return self

    def fit_with_val(
            self,
            X_train: pd.DataFrame,
            y_train: np.ndarray,
            X_val: pd.DataFrame,
            y_val: np.ndarray,
            lr: Optional[float] = None,
            epochs: Optional[int] = None,
            batch_size: Optional[int] = None,
            weight_decay: Optional[float] = None,
            loss_fn: Optional[nn.Module] = None
    ) -> Self:
        hp = self._resolve_hparams(lr, epochs, batch_size, weight_decay, loss_fn)
        X_tr_t, y_tr_t = self._to_tensor(X_train), self._to_target(y_train)
        X_va_t, y_va_t = self._to_tensor(X_val), self._to_target(y_val)
        input_dim = X_train.shape[1] if self.input_mode == "tabular" else 1
        self._init_net(output_dim=y_tr_t.shape[1], input_dim=input_dim)
        self._train_loop(X_tr_t, y_tr_t, X_va_t, y_va_t, hp)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._net is None:
            raise RuntimeError("Model not trained. Call fit/train first.")
        self._net.eval()
        X_t = self._to_tensor(X).to(self.device)
        with torch.no_grad(), torch.amp.autocast(self.device, enabled=True):
            return self._net(X_t).cpu().numpy()

    def _resolve_hparams(
            self,
            lr: Optional[float],
            epochs: Optional[int],
            batch_size: Optional[int],
            weight_decay: Optional[float],
            loss_fn: Optional[nn.Module]
    ) -> dict:
        return {
            "lr": lr or getattr(self, "lr", 1e-3),
            "epochs": epochs or getattr(self, "epochs", 50),
            "batch_size": batch_size or getattr(self, "batch_size", 64),
            "weight_decay": weight_decay or getattr(self, "weight_decay", 1e-5),
            "loss_fn": loss_fn or getattr(self, "loss_fn", nn.SmoothL1Loss(beta=0.01)),
            "patience": int(getattr(self, "patience", 10)),
            "min_delta": float(getattr(self, "min_delta", 0.0)),
            "scheduler_patience": int(getattr(self, "scheduler_patience", 5)),
            "clip_grad_norm": float(getattr(self, "clip_grad_norm", 1.0)),
            "use_amp": bool(getattr(self, "use_amp", False))
        }

    def _init_net(self, output_dim: int, input_dim: int) -> None:
        self._net = self._build_net(input_dim=input_dim, output_dim=output_dim).to(self.device)

    def _train_loop(
            self,
            X_t: Tensor,
            y_t: Tensor,
            Xv_t: Optional[Tensor],
            yv_t: Optional[Tensor],
            hp: dict,
    ) -> None:
        if self._net is None:
            raise RuntimeError("Network not initialized. Call _init_net first.")

        X_t, y_t = X_t.to(self.device), y_t.to(self.device)
        Xv_t = Xv_t.to(self.device) if Xv_t is not None else None
        yv_t = yv_t.to(self.device) if yv_t is not None else None
        use_val = Xv_t is not None and yv_t is not None

        opt: Optimizer = Adam(self._net.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
        scheduler = (
            ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=hp["scheduler_patience"], min_lr=1e-6)
            if use_val
            else None
        )
        scaler = torch.amp.GradScaler(enabled=hp["use_amp"])
        loss_fn = hp["loss_fn"]

        best_val = float("inf")
        best_state: Optional[dict[str, Tensor]] = None
        wait = 0

        for _ in range(hp["epochs"]):
            self._train_epoch(X_t, y_t, opt, loss_fn, scaler, hp["clip_grad_norm"], hp["batch_size"])

            if use_val:
                val_loss = self._val_loss(Xv_t, yv_t, loss_fn, scaler)
                if scheduler is not None:
                    scheduler.step(val_loss)

                if val_loss < best_val - hp["min_delta"]:
                    best_val = val_loss
                    wait = 0
                    best_state = {k: v.detach().clone() for k, v in self._net.state_dict().items()}
                else:
                    wait += 1
                    if wait >= hp["patience"]:
                        break

        if best_state is not None:
            self._net.load_state_dict(best_state)

    @staticmethod
    def _batch_iter(X: Tensor, y: Tensor, batch_size: int, *, shuffle: bool = True) -> Iterator[Tuple[Tensor, Tensor]]:
        n = int(X.size(0))
        idx = torch.randperm(n, device=X.device) if shuffle else torch.arange(n, device=X.device)
        for i in range(0, n, batch_size):
            yield X.index_select(0, idx[i: i + batch_size]), y.index_select(0, idx[i: i + batch_size])

    def _train_epoch(
            self,
            X_t: Tensor,
            y_t: Tensor,
            opt: Optimizer,
            loss_fn: nn.Module,
            scaler: torch.amp.GradScaler,
            clip_norm: float,
            batch_size: int,
    ) -> float:
        self._net.train()
        running = 0.0
        n_batches = 0
        for xb, yb in self._batch_iter(X_t, y_t, batch_size):
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(self.device, enabled=scaler.is_enabled()):
                pred = self._net(xb)
                loss = loss_fn(pred, yb)
            scaler.scale(loss).backward()
            if clip_norm > 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(self._net.parameters(), max_norm=clip_norm)
            scaler.step(opt)
            scaler.update()
            running += float(loss.item())
            n_batches += 1
        return running / max(1, n_batches)

    @torch.no_grad()
    def _val_loss(
            self,
            Xv: Tensor,
            yv: Tensor,
            loss_fn: nn.Module,
            scaler: torch.amp.GradScaler,
    ) -> float:
        self._net.eval()
        with torch.amp.autocast(self.device, enabled=scaler.is_enabled()):
            return float(loss_fn(self._net(Xv), yv).item())

    @staticmethod
    def _to_target(y: np.ndarray) -> Tensor:
        t = torch.as_tensor(y, dtype=torch.float32)
        return t.unsqueeze(1) if t.ndim == 1 else t

    def _to_tensor(self, X: Union[pd.DataFrame, np.ndarray]) -> Tensor:
        if self.input_mode == "sequence":
            if isinstance(X, pd.DataFrame):
                lag_cols = [c for c in X.columns if c.startswith("lag_")]
                if not lag_cols:
                    raise ValueError("sequence mode requires lag_* columns in X.")
                lag_cols = sorted(lag_cols, key=lambda s: int(s.split("_")[1]), reverse=True)
                arr = X[lag_cols].to_numpy(dtype=np.float32)
            else:
                arr = np.asarray(X, dtype=np.float32)
            return torch.from_numpy(arr[:, :, None])  # (N, T, 1)

        # tabular mode
        arr = X.to_numpy(dtype=np.float32) if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=np.float32)
        return torch.from_numpy(arr[:, :, None])

