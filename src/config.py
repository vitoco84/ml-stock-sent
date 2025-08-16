from types import SimpleNamespace

import yaml


class Config:
    """Configuration Class: Loads configuration from YAML file."""

    def __init__(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            cfg_dict = yaml.safe_load(f)
        self._config = self._dict_to_namespace(cfg_dict)

    def _dict_to_namespace(self, d):
        if isinstance(d, dict):
            return SimpleNamespace(**{k: self._dict_to_namespace(v) for k, v in d.items()})
        return d

    def __getattr__(self, name):
        return getattr(self._config, name)

    def __repr__(self):
        return repr(self._config)
