"""Custom exception classes for the data pipeline."""


class DataPipelineError(Exception):
    """Base exception for data pipeline errors."""
    pass


class DataGenerationError(DataPipelineError):
    """Raised when data generation fails."""
    pass


class DataValidationError(DataPipelineError):
    """Raised when data validation fails."""
    pass


class AnalysisError(DataPipelineError):
    """Raised when analysis operations fail."""
    pass