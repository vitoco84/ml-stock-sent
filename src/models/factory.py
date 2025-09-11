from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from src.models.base import Base


class ModelBuilder(Protocol):
    """Protocol for experiment model builders."""

    def __call__(self, horizon: int, seed: int) -> Base: ...

@dataclass(frozen=True, slots=True)
class Experiment:
    """Factory container for experiment setup."""
    name: str
    build: ModelBuilder
    include_sentiment: bool
