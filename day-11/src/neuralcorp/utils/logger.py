"""Structured logging setup for NeuralCorp.

Why structured logging (not bare print)?
- Log messages carry context: timestamp, level, module name.
- Log levels let you tune verbosity without code changes.
- Production systems parse structured logs for alerting/dashboards.
"""

import logging
import sys

from neuralcorp.config import settings


def get_logger_name(name: str) -> logging.Logger:
    """Create and configure a named logger.

    Args:
        name: Logger name — use __name__ in every module for
              automatic hierarchy (neuralcorp.fetcher.client, etc.)

    Returns:
        A configured logging.Logger instance.

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Starting fetch for 5 URLs")
    """
    logger = logging.getLogger(name)
    # Only add handler if this logger has none — prevents duplicate logs
    # when get_logger() is called multiple times in the same process.
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)

    # Set level from config — e.g., "INFO" → logging.INFO (20)
    logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))
    return logger
