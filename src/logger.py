import logging


def get_logger(name: str = __name__, level: int = logging.INFO) -> logging.Logger:
    """
    Simple Console logger.
    Args:
        name: Logger name, typically __name__
        level: Log level (e.g., logging.DEBUG, logging.INFO)

    Usage:
        logger = get_logger(__name__, log_to_file=True)
        logger.info("Started process.")
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
