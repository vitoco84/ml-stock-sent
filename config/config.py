from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse

import yaml


def _is_url(s: str) -> bool:
    parsed = urlparse(s)
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)

def _looks_like_path(s: str) -> bool:
    return s.startswith(("~", ".", "/", "\\")) or ("/" in s) or ("\\" in s)

class Config:
    """Minimal YAML config loader."""

    def __init__(self, path: Path):
        path = Path(path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        self._config_file = path
        self._config_dir = path.parent

        with path.open("r", encoding="utf-8") as f:
            raw_cfg = yaml.safe_load(f) or {}

        self._config = self._to_namespace(raw_cfg)

    def _to_namespace(self, obj):
        if isinstance(obj, dict):
            return SimpleNamespace(**{k: self._to_namespace(v) for k, v in obj.items()})
        if isinstance(obj, list):
            return [self._to_namespace(v) for v in obj]
        if isinstance(obj, str):
            s = obj.strip()
            if _is_url(s):
                return s
            if _looks_like_path(s):
                return (self._config_dir / s).expanduser().resolve()
            return s
        return obj

    def __getattr__(self, name):
        return getattr(self._config, name)

    def __repr__(self):
        return f"Config({self._config_file})"
