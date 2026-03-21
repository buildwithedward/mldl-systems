# Production rule: never raise bare Exception() — always a descriptive custom class
# This makes error handling precise and logs easy to filter by error type

class NeuralCorpBaseException(Exception):
    """Base exception for NeuralCorp ML Template."""
    pass


class ConfigurationError(NeuralCorpBaseException):
    """Raised when there is a configuration error or environment variable issue."""
    pass


class DataValidationError(NeuralCorpBaseException):
    """Raised when there is a data validation error."""
    pass


class ModelLoadingError(NeuralCorpBaseException):
    """Raised when there is an error loading the model or cannot find the model file."""
    pass
