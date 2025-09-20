from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlparse

import yaml


def _is_url(s: str) -> bool:
    """Check if a string looks like an HTTP(S) URL."""
    parsed = urlparse(s)
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)

def _looks_like_path(s: str) -> bool:
    """Heuristic: check if a string looks like a filesystem path."""
    return s.startswith(("~", ".", "/", "\\")) or any(ch in s for ch in ("/", "\\"))

class Config:
    """Minimal YAML configuration loader."""

    def __init__(self, path: Path | str) -> None:
        cfg_path = Path(path).expanduser().resolve()
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config file not found: {cfg_path}")

        self._config_file: Path = cfg_path
        self._config_dir: Path = cfg_path.parent

        with cfg_path.open("r", encoding="utf-8") as f:
            raw_cfg: dict[str, Any] = yaml.safe_load(f) or {}

        self._config: SimpleNamespace = self._to_namespace(raw_cfg)

    def _to_namespace(self, obj: Any) -> Any:
        """Recursively convert dicts/lists into namespaces and resolve paths/URLs."""
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

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the internal namespace."""
        return getattr(self._config, name)

    def __repr__(self) -> str:
        env = getattr(self._config, "env", None)
        return f"Config(file={self._config_file}, env={env!r})"

    @classmethod
    def load(cls, path: Path | str | None = None) -> Config:
        """Load a config file, resolving automatically in any environment."""
        if path is None:
            env_path = os.getenv("CONFIG_PATH")
            if env_path:
                path = Path(env_path)
            else:
                # Walk upwards until we find config/config.yaml
                cur = Path(__file__).resolve()
                for parent in cur.parents:
                    candidate = parent / "config" / "config.yaml"
                    if candidate.exists():
                        path = candidate
                        break
                else:
                    raise FileNotFoundError("Could not find config/config.yaml")

        return cls(path)
