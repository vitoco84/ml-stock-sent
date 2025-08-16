import logging


def get_logger(name: str = __name__, level: int = logging.INFO) -> logging.Logger:
    """
    Return a logger with the given name.
    How to use:
    logging.info("This is an INFO message")

    Log Levels: INFO, DEBUGm WARNING, ERROR, CRITICAL
    """

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger
