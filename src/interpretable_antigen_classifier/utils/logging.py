"""Logging utilities for consistent formatting across the pipeline."""
import logging
from typing import Optional

DEFAULT_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"


def get_logger(name: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """Return a logger with a standard format.

    Parameters
    ----------
    name:
        Name for the logger; defaults to root when None.
    level:
        Logging level to set.
    """
    logging.basicConfig(level=level, format=DEFAULT_LOG_FORMAT)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger
