"""Dataclasses: auto-generated methods, validation, immutability."""

import logging
from dataclasses import dataclass, field
from typing import Any, List

logger = logging.getLogger(__name__)


@dataclass
class Point:
    """2D point with x and y coordinates.

    Attributes:
        x: X coordinate.
        y: Y coordinate.
    """

    x: float
    y: float

    def __post_init__(self) -> None:
        """Validate coordinates after initialization."""
        if not isinstance(self.x, (int, float)) or not isinstance(
            self.y, (int, float)
        ):
            logger.error(f"Invalid coordinates: x={self.x}, y={self.y}")
            raise TypeError("x and y must be numeric")
        logger.debug(f"Point created: ({self.x}, {self.y})")

    def distance_from_origin(self) -> float:
        """Calculate distance from origin (0, 0).

        Returns:
            Euclidean distance from origin.
        """
        logger.debug(f"Calculating distance for {self}")
        dist = (self.x**2 + self.y**2) ** 0.5
        logger.debug(f"Distance: {dist}")
        return dist


@dataclass
class Rectangle:
    """Rectangle defined by top-left point and dimensions.

    Attributes:
        top_left: Top-left corner as Point.
        width: Width in units.
        height: Height in units.
    """

    top_left: Point
    width: float
    height: float

    def __post_init__(self) -> None:
        """Validate dimensions after initialization."""
        if self.width <= 0 or self.height <= 0:
            logger.error(f"Invalid dimensions: {self.width}x{self.height}")
            raise ValueError("Width and height must be positive")
        logger.debug(f"Rectangle created: {self.width}x{self.height}")

    def area(self) -> float:
        """Calculate area of rectangle.

        Returns:
            Area as width * height.
        """
        logger.debug(f"Calculating area for {self}")
        result = self.width * self.height
        logger.debug(f"Area: {result}")
        return result


@dataclass(frozen=True)
class Config:
    """Immutable configuration object.

    Attributes:
        name: Configuration name.
        timeout: Timeout in seconds.
    """

    name: str
    timeout: int

    def __post_init__(self) -> None:
        """Validate config after initialization."""
        if self.timeout <= 0:
            logger.error(f"Invalid timeout: {self.timeout}")
            raise ValueError("timeout must be positive")
        logger.debug(f"Immutable Config created: {self.name}")


@dataclass
class FileStats:
    """Statistics collected from analyzing a file.

    Attributes:
        filepath: Path to file (string).
        line_count: Number of lines.
        char_count: Number of characters.
        word_count: Number of words.
        imports: List of imported module names.
    """

    filepath: str
    line_count: int
    char_count: int
    word_count: int
    imports: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate stats after initialization."""
        if self.line_count < 0 or self.char_count < 0 or self.word_count < 0:
            logger.error(f"Negative counts in {self}")
            raise ValueError("Counts cannot be negative")
        logger.debug(f"FileStats created for {self.filepath}")

    def summary(self) -> str:
        """Generate summary string.

        Returns:
            Human-readable summary of file statistics.
        """
        logger.debug(f"Generating summary for {self.filepath}")
        return (
            f"{self.filepath}: {self.line_count} lines, "
            f"{self.char_count} chars, {self.word_count} words, "
            f"{len(self.imports)} imports"
        )