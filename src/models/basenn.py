from __future__ import annotations

from abc import abstractmethod
from typing import Iterator, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn, Tensor
from torch.optim import Adam
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

    def __init__(self, horizon: int = 30, random_state: int = 42, device: Optional[str] = None):
        super().__init__(horizon=horizon, random_state=random_state)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._net: nn.Module | None = None

    @abstractmethod
    def _build_net(self, input_dim: int, output_dim: int) -> nn.Module:
        raise NotImplementedError

    def fit(
            self,
            X: pd.DataFrame,
            y: np.ndarray,
            lr: Optional[float] = None,
            epochs: Optional[int] = None,
            batch_size: Optional[int] = None,
            weight_decay: Optional[float] = None
    ):
        hp = self._resolve_hparams(lr, epochs, batch_size, weight_decay)
        X_t, y_t = self._to_tensor(X), self._to_target(y)
        self._init_net(output_dim=y_t.shape[1])
        self._train_loop(X_t, y_t, None, None, hp)
        return self

    def fit_with_val(
            self,
            X_train: pd.DataFrame,
            y_train: np.ndarray,
            X_val: pd.DataFrame, y_val: np.ndarray,
            lr: Optional[float] = None,
            epochs: Optional[int] = None,
            batch_size: Optional[int] = None,
            weight_decay: Optional[float] = None
    ) -> Base:
        hp = self._resolve_hparams(lr, epochs, batch_size, weight_decay)
        X_tr_t, y_tr_t = self._to_tensor(X_train), self._to_target(y_train)
        X_va_t, y_va_t = self._to_tensor(X_val), self._to_target(y_val)
        self._init_net(output_dim=y_tr_t.shape[1])
        self._train_loop(X_tr_t, y_tr_t, X_va_t, y_va_t, hp)
        return self

    def train(self, X_train: pd.DataFrame, y_train: np.ndarray,
              X_val: pd.DataFrame = None, y_val: np.ndarray = None) -> Base:
        if X_val is None or y_val is None:
            return self.fit(X_train, y_train)
        return self.fit_with_val(X_train, y_train, X_val, y_val)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._net is None:
            raise RuntimeError("Model not trained. Call fit/train first.")
        self._net.eval()
        X_t = self._to_tensor(X).to(self.device)
        with torch.no_grad():
            return self._net(X_t).cpu().numpy()

    def _resolve_hparams(self, lr, epochs, batch_size, weight_decay) -> dict:
        return {
            "lr": lr if lr is not None else getattr(self, "lr", 1e-3),
            "epochs": epochs if epochs is not None else getattr(self, "epochs", 50),
            "batch_size": batch_size if batch_size is not None else getattr(self, "batch_size", 64),
            "weight_decay": weight_decay if weight_decay is not None else getattr(self, "weight_decay", 1e-5),
            "patience": int(getattr(self, "patience", 10)),
            "min_delta": float(getattr(self, "min_delta", 0.0)),
            "scheduler_patience": int(getattr(self, "scheduler_patience", 5)),
            "clip_grad_norm": float(getattr(self, "clip_grad_norm", 1.0)),
            "use_amp": bool(getattr(self, "use_amp", self.device.startswith("cuda"))),
        }

    def _init_net(self, output_dim: int) -> None:
        # input_dim is 1 for sequence conv/lstm; MLP typically ignores it via LazyLinear.
        self._net = self._build_net(input_dim=1, output_dim=output_dim).to(self.device)

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
        use_val = (Xv_t is not None) and (yv_t is not None)

        opt = Adam(self._net.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
        loss_fn = nn.SmoothL1Loss(beta=0.001)
        scheduler = ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=hp["scheduler_patience"], min_lr=1e-6
        ) if use_val else None
        scaler = torch.amp.GradScaler(enabled=bool(hp["use_amp"]))

        best_val = float("inf")
        best_state: Optional[dict[str, Tensor]] = None
        wait = 0

        for _ in range(hp["epochs"]):
            self._train_epoch(
                X_t=X_t, y_t=y_t, opt=opt, loss_fn=loss_fn, scaler=scaler,
                clip_norm=hp["clip_grad_norm"], batch_size=hp["batch_size"]
            )

            if use_val:
                val_loss = self._val_loss(Xv_t, yv_t, loss_fn, scaler)
                if scheduler is not None:
                    scheduler.step(val_loss)

                improved = val_loss < (best_val - float(hp["min_delta"]))
                if improved:
                    best_val = float(val_loss)
                    wait = 0
                    best_state = {k: v.detach().clone() for k, v in self._net.state_dict().items()}
                else:
                    wait += 1
                    if wait >= int(hp["patience"]):
                        break

        if best_state is not None:
            self._net.load_state_dict(best_state)

    @staticmethod
    def _batch_iter(X: Tensor, y: Tensor, batch_size: int, *, shuffle: bool = True) -> Iterator[Tuple[Tensor, Tensor]]:
        n = int(X.size(0))

        if n == 1:
            idx = torch.tensor([0, 0], device=X.device)
            yield X.index_select(0, idx), y.index_select(0, idx)
            return

        idx = torch.randperm(n, device=X.device) if shuffle else torch.arange(n, device=X.device)

        i = 0
        while i < n:
            end = min(i + batch_size, n)
            if end - i == 1:
                i = n - 2
                end = n
            sel = idx[i:end]
            yield X.index_select(0, sel), y.index_select(0, sel)
            if end == n:
                break
            i = end

    def _train_epoch(
            self,
            X_t: Tensor,
            y_t: Tensor,
            opt: torch.optim.Optimizer,
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
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                pred = self._net(xb)
                loss = loss_fn(pred, yb)
            scaler.scale(loss).backward()
            if clip_norm and clip_norm > 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(self._net.parameters(), max_norm=float(clip_norm))
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
        with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
            loss = loss_fn(self._net(Xv), yv).item()
        return float(loss)

    @staticmethod
    def _to_target(y: np.ndarray) -> Tensor:
        t = torch.as_tensor(y, dtype=torch.float32)
        if t.ndim == 1:
            t = t.unsqueeze(1)
        return t

    def _to_tensor(self, X: pd.DataFrame) -> Tensor:
        """
        tabular  -> (N, F, 1)
        sequence -> (N, T, 1) using only sorted lag_* columns
        """
        if self.input_mode == "sequence":
            lag_cols = [c for c in X.columns if c.startswith("lag_")]
            if not lag_cols:
                raise ValueError("sequence mode requires lag_* columns in X.")
            # oldest -> most recent: lag_T, ..., lag_1
            lag_cols = sorted(lag_cols, key=lambda s: int(s.split("_")[1]), reverse=True)
            arr = X[lag_cols].to_numpy(dtype=np.float32)  # (N, T)
            return torch.from_numpy(arr[:, :, None])  # (N, T, 1)
        arr = X.to_numpy(dtype=np.float32) if hasattr(X, "to_numpy") else np.asarray(X, dtype=np.float32)
        arr = arr.reshape((len(arr), arr.shape[1], 1))
        return torch.from_numpy(arr)
