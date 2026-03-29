"""Logging configuration with structured JSON output."""
import logging
import logging.handlers
from pathlib import Path
from pythonjsonlogger import jsonlogger
from src.config import settings
from src.exceptions import LoggingConfigError


def setup_logging() -> logging.Logger:
    """
    Configure structured JSON logging for the pipeline.

    Creates:
    - JSON file handler for structured logs (machine-readable)
    - Console handler for human-readable output

    Returns:
        logging.Logger: Configured root logger.

    Raises:
        LoggingConfigError: If log directory cannot be created.
    """
    try:
        # Create log directory
        log_path = settings.log_file_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise LoggingConfigError(f"Cannot create log directory: {e}") from e

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(settings.log_level)

    # Remove existing handlers (idempotent - safe to call multiple times)
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # JSON file handler (structured logs for machines)
    json_handler = logging.handlers.RotatingFileHandler(
        str(log_path),
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=3
    )
    json_formatter = jsonlogger.JsonFormatter(
        fmt="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    json_handler.setFormatter(json_formatter)
    root_logger.addHandler(json_handler)

    # Console handler (human-readable)
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a named logger (best practice for modules).

    Args:
        name: Logger name, typically __name__.

    Returns:
        logging.Logger: Named logger instance.
    """
    return logging.getLogger(name)