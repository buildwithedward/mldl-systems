"""Custom exceptions for the NeuralCorp Scaffold project."""


class NeuralCorpError(Exception):
    """Base exception class for NeuralCorp Scaffold errors."""

    pass


class DataValidationError(NeuralCorpError):
    """Exception raised for errors in data validation."""

    pass


class ModelNotFoundError(NeuralCorpError):
    """Exception raised when a specified model is not found."""

    pass


class InferenceError(NeuralCorpError):
    """Exception raised for errors during model inference."""

    pass


class ConfigurationError(NeuralCorpError):
    """Exception raised for errors in configuration settings."""

    pass
