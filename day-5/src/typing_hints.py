"""Type hints: annotations, generics, protocols, and type checking."""

import logging
from typing import Callable, Optional, Protocol, TypeVar, Union

logger = logging.getLogger(__name__)

T = TypeVar("T")
U = TypeVar("U")


class Drawable(Protocol):
    """Protocol for objects that can be drawn."""

    def draw(self) -> str:
        """Return drawing representation."""
        ...


def greet(name: str) -> str:
    """Greet a person by name.

    Args:
        name: Person's name.

    Returns:
        Greeting string.

    Raises:
        TypeError: If name is not a string.
    """
    if not isinstance(name, str):
        logger.error(f"Expected str, got {type(name)}")
        raise TypeError(f"name must be str, got {type(name).__name__}")
    logger.info(f"Greeting {name}")
    return f"Hello, {name}!"


def add_numbers(a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
    """Add two numbers of any numeric type.

    Args:
        a: First number.
        b: Second number.

    Returns:
        Sum of a and b.
    """
    logger.debug(f"Adding {a} + {b}")
    result = a + b
    logger.debug(f"Result: {result}")
    return result


def find_in_list(
    items: list[str], target: str
) -> Optional[int]:
    """Find index of target in list, or None if not found.

    Args:
        items: List of strings.
        target: String to find.

    Returns:
        Index of target, or None if not found.
    """
    logger.debug(f"Searching for '{target}' in {len(items)} items")
    try:
        index = items.index(target)
        logger.info(f"Found '{target}' at index {index}")
        return index
    except ValueError:
        logger.info(f"'{target}' not found")
        return None


def apply_function(func: Callable[[int], int], value: int) -> int:
    """Apply a function to a value.

    Args:
        func: Function taking int and returning int.
        value: Input value.

    Returns:
        Result of func(value).
    """
    logger.debug(f"Applying function to {value}")
    result = func(value)
    logger.debug(f"Function returned {result}")
    return result


def identity(x: T) -> T:
    """Return input unchanged (identity function for type variables).

    Args:
        x: Input value.

    Returns:
        Same value unchanged.
    """
    logger.debug(f"Identity returning {x!r}")
    return x


def compose(f: Callable[[U], T], g: Callable[[int], U]) -> Callable[[int], T]:
    """Compose two functions: f(g(x)).

    Args:
        f: Outer function.
        g: Inner function.

    Returns:
        Composed function.
    """
    logger.debug("Creating composed function")

    def composed(x: int) -> T:
        """Execute composition."""
        logger.debug(f"Executing composition on {x}")
        return f(g(x))

    return composed


def draw_shape(shape: Drawable) -> str:
    """Draw a shape using its draw() method (protocol example).

    Args:
        shape: Any object implementing Drawable protocol.

    Returns:
        Drawing string from shape.draw().
    """
    logger.debug(f"Drawing {shape}")
    return shape.draw()