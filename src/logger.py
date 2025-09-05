from __future__ import annotations

import logging
from pathlib import Path


def get_logger(
        name: str = __name__,
        level: int | str = logging.INFO,
        log_to_file: bool = False,
        file_path: str | Path = "app.log",
) -> logging.Logger:
    """
    Create and return a configured logger.

    Args:
        name: Logger name, typically __name__.
        level: Log level (int or str, e.g. logging.DEBUG or "INFO").
        log_to_file: If True, also log to a file.
        file_path: Path to the log file (only if log_to_file=True).

    Example:
        logger = get_logger(__name__, log_to_file=True)
        logger.info("Started process.")
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Optional file handler
    if log_to_file:
        file_handler = logging.FileHandler(Path(file_path))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
