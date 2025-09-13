from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Final, Union

import numpy as np
import pandas as pd

from src.logger import get_logger


logger = get_logger(__name__)

_THREAD_ENV_VARS: Final[dict[str, str]] = {
    "PYTHONHASHSEED": None,
    "TF_DETERMINISTIC_OPS": "1",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "OMP_NUM_THREADS": "8",
    "OPENBLAS_NUM_THREADS": "8",
    "MKL_NUM_THREADS": "8",
    "VECLIB_MAXIMUM_THREADS": "8",
    "NUMEXPR_NUM_THREADS": "8"
}

def set_seed(seed: int = 42) -> np.random.Generator:
    """Set global random seed for reproducibility across multiple libraries."""

    # Environment variables
    for k, v in _THREAD_ENV_VARS.items():
        os.environ[k] = str(seed) if v is None else v

    # Python & NumPy
    random.seed(seed)
    np.random.seed(seed)

    # TensorFlow
    try:
        import tensorflow as tf  # type: ignore
        tf.random.set_seed(seed)
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"Could not fully set seed for TensorFlow: {e}")

    # PyTorch
    try:
        import torch  # type: ignore
        torch.manual_seed(seed)
        if hasattr(torch, "use_deterministic_algorithms"):
            try:
                torch.use_deterministic_algorithms(True)
            except Exception as e:
                logger.warning(
                    f"torch deterministic algorithms not fully supported: {e}"
                )
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"Could not fully set seed for PyTorch: {e}")

    logger.info(f"Global random seed set to {seed}")
    return np.random.default_rng(seed)

def results_to_df(results, key: Union[str, list[str]]) -> pd.DataFrame:
    if isinstance(key, str):
        key = [key]

    dfs = []
    for res in results:
        data = res
        for k in key:
            data = data[k]

        if isinstance(data, dict) and all(isinstance(v, dict) for v in data.values()):
            df = pd.DataFrame.from_dict(data, orient="index").reset_index().rename(
                columns={"index": key[-1]}
            )
            df.insert(0, "model", res["kind"])
        else:
            df = pd.DataFrame([data])
            df.insert(0, "model", res["kind"])

        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)

def _to_1d_float(arr) -> np.ndarray:
    if arr is None:
        return np.array([], dtype=float)
    a = np.asarray(arr).ravel()
    a = pd.to_numeric(pd.Series(a), errors="coerce").to_numpy(dtype=float)
    return a[np.isfinite(a)]

def load_results_from_dir(out_dir: Path, load_arrays: bool = True) -> list[dict]:
    results = []
    for json_path in sorted(out_dir.glob("*_result.json")):
        with open(json_path) as f:
            res = json.load(f)

        if load_arrays:
            npz_path = Path(res["paths"]["preds_npz"])
            if npz_path.exists():
                with np.load(npz_path, allow_pickle=True) as data:
                    res["y_pred_val"] = _to_1d_float(data.get("y_pred_val"))
                    res["y_pred_test"] = _to_1d_float(data.get("y_pred_test"))
                    res["y_pred_last"] = _to_1d_float(data.get("y_pred_last"))

            test_idx_path = Path(res["paths"]["test_index_npy"])
            if test_idx_path.exists():
                res["test_index"] = np.load(test_idx_path, allow_pickle=True)

        results.append(res)

    return results
