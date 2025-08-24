import os
import random

import numpy as np

from src.logger import get_logger


logger = get_logger(__name__)

def set_seed(seed: int = 42) -> np.random.Generator:
    """Set random seed globally across numpy, random, torch, tensorflow if available."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

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
        if hasattr(torch, "use_deterministic_algorithms"):
            try:
                torch.use_deterministic_algorithms(True)
            except Exception as e:
                logger.warning(f"torch deterministic algorithms not fully supported: {e}")
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"Could not fully set deterministic behavior for torch: {e}")

    logger.info(f"Global random seed set to {seed}")
    return np.random.default_rng(seed)
