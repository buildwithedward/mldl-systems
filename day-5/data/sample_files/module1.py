"""Sample module 1."""

import os
import sys
from pathlib import Path


def hello(name: str) -> str:
    """Greet someone."""
    return f"Hello, {name}!"


def process_list(items: list[int]) -> list[int]:
    """Process list of integers."""
    return [x * 2 for x in items if x > 0]

