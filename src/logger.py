import logging
from pathlib import Path


def get_logger(name: str = __name__, level: int = logging.INFO, log_to_file: bool = False) -> logging.Logger:
    """
    Returns a logger with stream and optional file logging.

    Args:
        name: Logger name, typically __name__
        level: Log level (e.g., logging.DEBUG, logging.INFO)
        log_to_file: Whether to also log to a file (logs/app.log)

    Usage:
        logger = get_logger(__name__, log_to_file=True)
        logger.info("Started process.")
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")

    if not logger.handlers:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        if log_to_file:
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            file_handler = logging.FileHandler(log_dir / "app.log")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger
