import re
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse

import yaml


HF_ID = re.compile(r'^[\w.-]+/[\w.-]+$')

class Config:
    """Configuration Class: Loads configuration from YAML file."""

    def __init__(self, path: Path):
        path = Path(path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        self._root = path.parent

        with path.open("r", encoding="utf-8") as f:
            cfg_dict = yaml.safe_load(f)

        self._config = self._to_namespace(cfg_dict)

    def _to_namespace(self, obj):
        if isinstance(obj, dict):
            return SimpleNamespace(**{k: self._to_namespace(v) for k, v in obj.items()})
        elif isinstance(obj, list):
            return [self._to_namespace(v) for v in obj]
        elif isinstance(obj, str):
            s = obj.strip()
            if self._is_url(s) or self._is_hf_repo_id(s):
                return s
            if self._looks_like_path(s):
                p = Path(s)
                if not p.is_absolute():
                    p = (self._root / p).resolve()
                return p
            return s
        else:
            return obj

    def _is_url(self, s: str) -> bool:
        parsed = urlparse(s)
        return parsed.scheme in ("http", "https")

    def _is_hf_repo_id(self, s: str) -> bool:
        return bool(HF_ID.fullmatch(s))

    def _looks_like_path(self, s: str) -> bool:
        return s.startswith((".", "..", "/", "\\")) or ("/" in s or "\\" in s)

    def __getattr__(self, name):
        return getattr(self._config, name)

    def __repr__(self):
        return repr(self._config)
