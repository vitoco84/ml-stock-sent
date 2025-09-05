from __future__ import annotations

import os
import random
from typing import Final

import numpy as np

from src.logger import get_logger


logger = get_logger(__name__)

_THREAD_ENV_VARS: Final[dict[str, str]] = {
    "PYTHONHASHSEED": None,
    "TF_DETERMINISTIC_OPS": "1",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1"
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
