"""Custom exceptions for Day 7 pipeline."""


class VisualisationError(Exception):
    """Raised when plot generation fails."""
    pass


class DataGenerationError(Exception):
    """Raised when data cannot be generated."""
    pass


class LoggingConfigError(Exception):
    """Raised when logging configuration fails."""
    pass