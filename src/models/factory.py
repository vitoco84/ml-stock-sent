from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class Experiment:
    name: str
    build: Callable[[int, int], Any]
    include_sentiment: bool
