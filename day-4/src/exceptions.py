"""Custom exceptions for the day 4 project."""


class NC004Error(Exception):
    """Base class for all NC004 exceptions."""
    pass

class InvalidBatchSizeError(NC004Error):
    """Raised when the batch size is invalid."""
    pass


class ModelNotInitializedError(NC004Error):
    """Raised when the model is not initialized before training."""
    pass

class ModelNotFittedError(NC004Error):
    """Raised when trying to access model properties before fitting."""
    pass