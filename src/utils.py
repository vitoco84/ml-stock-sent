import os
import random
from pathlib import Path

import numpy as np
import pandas as pd

from src.logger import get_logger


logger = get_logger(__name__)

def set_seed(seed: int = 42) -> np.random.Generator:
    """Set Random Seed globally."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"Could not fully set deterministic behavior for torch: {e}")

    logger.info(f"Global random seed set to {seed}")
    return np.random.default_rng(seed)

def is_cuda_available() -> bool:
    """Check if GPU is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False

def load_csv(path: Path) -> pd.DataFrame:
    """ Load a CSV file."""
    return pd.read_csv(path)

def save_csv(path: Path, df: pd.DataFrame) -> None:
    """Save to a CSV file."""
    df.to_csv(path, index=False)
