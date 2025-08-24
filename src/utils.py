import json
import os
import random
from pathlib import Path
from typing import Dict

import numpy as np

from src.logger import get_logger


logger = get_logger(__name__)

def set_seed(seed: int = 42) -> np.random.Generator:
    """Set random seed globally across numpy, random, torch, tensorflow if available."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    # TensorFlow
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass

    # PyTorch
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
