"""Sample module 3."""

from typing import Optional
from dataclasses import dataclass


@dataclass
class Config:
    """Configuration object."""

    name: str
    timeout: int = 30


def validate(value: Optional[str]) -> bool:
    """Validate input."""
    return value is not None and len(value) > 0
