"""Tests for typing hint functions."""

import pytest

from src.typing_hints import (
    add_numbers,
    apply_function,
    compose,
    find_in_list,
    greet,
    identity,
)


class TestBasicTyping:
    """Test basic typed functions."""

    def test_greet_happy_path(self) -> None:
        """Test greeting function."""
        result = greet("Alice")
        assert result == "Hello, Alice!"

    def test_greet_invalid_type(self) -> None:
        """Test error on wrong type."""
        with pytest.raises(TypeError):
            greet(123)  # type: ignore


class TestUnionTypes:
    """Test union type handling."""

    def test_add_numbers_integers(self) -> None:
        """Test adding integers."""
        result = add_numbers(2, 3)
        assert result == 5

    def test_add_numbers_floats(self) -> None:
        """Test adding floats."""
        result = add_numbers(2.5, 3.5)
        assert result == pytest.approx(6.0)

    def test_add_numbers_mixed(self) -> None:
        """Test adding int and float."""
        result = add_numbers(2, 3.5)
        assert result == pytest.approx(5.5)


class TestOptionalTypes:
    """Test optional type handling."""

    def test_find_in_list_found(self) -> None:
        """Test finding element."""
        result = find_in_list(["a", "b", "c"], "b")
        assert result == 1

    def test_find_in_list_not_found(self) -> None:
        """Test element not found returns None."""
        result = find_in_list(["a", "b", "c"], "x")
        assert result is None


class TestCallableTypes:
    """Test callable type handling."""

    def test_apply_function_happy_path(self) -> None:
        """Test applying function."""
        def double(x: int) -> int:
            """Double the input."""
            return x * 2

        result = apply_function(double, 5)
        assert result == 10


class TestGenericTypes:
    """Test generic/TypeVar functions."""

    def test_identity_int(self) -> None:
        """Test identity with int."""
        result = identity(42)
        assert result == 42

    def test_identity_string(self) -> None:
        """Test identity with string."""
        result = identity("hello")
        assert result == "hello"


class TestComposition:
    """Test function composition."""

    def test_compose_happy_path(self) -> None:
        """Test composing two functions."""
        def double(x: int) -> int:
            """Double input."""
            return x * 2

        def add_one(x: int) -> int:
            """Add one to input."""
            return x + 1

        composed = compose(double, add_one)
        result = composed(5)
        assert result == 12