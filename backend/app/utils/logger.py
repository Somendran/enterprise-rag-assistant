"""
logger.py
─────────
Provides a single, consistently-formatted logger for the whole application.

Usage anywhere in the project:
    from app.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Something happened")
"""

import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """
    Returns a logger with:
    - ISO-8601 timestamps
    - Module-level name for easy tracing
    - stdout output (works well with container log aggregators)
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if the logger is fetched multiple times
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Prevent log records from propagating to the root logger
    # (avoids duplicate output when uvicorn also sets up root logging)
    logger.propagate = False

    return logger
