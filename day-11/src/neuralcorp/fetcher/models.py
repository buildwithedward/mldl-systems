"""Pydantic models for async fetch results.

Why Pydantic for data models?
- Validates shape and types automatically.
- Gives clear error messages when data doesn't match expectations.
- Serialises to/from JSON for free.
"""

from pydantic import BaseModel


class FetchResult(BaseModel):
    """Represents the result of fetching a single URL."""

    url: str
    status_code: int
    content_length: int
    elapsed_ms: float
    success: bool

    @classmethod
    def from_error(cls, url: str, elapsed_ms: float) -> "FetchResult":
        """Create a failed FetchResult when an exception occurred.

        Args:
            url: The URL that failed.
            elapsed_ms: Time elapsed before the error.

        Returns:
            A FetchResult representing the failure.
        """
        return cls(
            url=url,
            status_code=0,
            content_length=0,
            elapsed_ms=elapsed_ms,
            success=False,
        )
