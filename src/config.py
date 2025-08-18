from pathlib import Path
from types import SimpleNamespace

import yaml


class Config:
    """Configuration Class: Loads configuration from YAML file."""

    def __init__(self, path: Path):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            cfg_dict = yaml.safe_load(f)
        self._config = self._to_namespace(cfg_dict)

    def _to_namespace(self, obj):
        if isinstance(obj, dict):
            return SimpleNamespace(**{k: self._to_namespace(v) for k, v in obj.items()})
        if isinstance(obj, list):
            return [self._to_namespace(v) for v in obj]
        return obj

    def __getattr__(self, name):
        return getattr(self._config, name)

    def __repr__(self):
        return repr(self._config)
