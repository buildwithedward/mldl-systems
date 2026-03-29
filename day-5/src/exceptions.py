"""Custom exceptions for the Day 5 file analyzer project."""


class FileAnalyzerError(Exception):
    """Base exception for file analyzer errors."""

    pass


class PathError(FileAnalyzerError):
    """Raised when a path is invalid or inaccessible."""

    pass


class ConfigError(FileAnalyzerError):
    """Raised when configuration is invalid."""

    pass


class AnalysisError(FileAnalyzerError):
    """Raised when file analysis fails."""

    pass