"""Tests for decorator functions."""

import time
from unittest.mock import Mock

import pytest

from src.decorators import log_calls_decorator, retry_decorator, timer_decorator


class TestTimerDecorator:
    """Test timing decorator."""

    def test_timer_decorator_executes_function(self) -> None:
        """Test that decorated function still executes."""

        @timer_decorator
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        result = add(2, 3)
        assert result == 5

    def test_timer_decorator_preserves_metadata(self) -> None:
        """Test that functools.wraps preserves function name."""

        @timer_decorator
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        assert add.__name__ == "add"
        assert "Add two" in add.__doc__


class TestRetryDecorator:
    """Test retry decorator."""

    def test_retry_decorator_succeeds_first_attempt(self) -> None:
        """Test succeeds without retrying."""

        @retry_decorator(max_attempts=3, delay=0.01)
        def always_works() -> str:
            """Return success."""
            return "success"

        result = always_works()
        assert result == "success"

    def test_retry_decorator_succeeds_second_attempt(self) -> None:
        """Test succeeds on retry."""
        attempt = [0]

        @retry_decorator(max_attempts=3, delay=0.01)
        def fails_once() -> str:
            """Fail once then succeed."""
            attempt[0] += 1
            if attempt[0] == 1:
                raise ValueError("First attempt fails")
            return "success"

        result = fails_once()
        assert result == "success"
        assert attempt[0] == 2

    def test_retry_decorator_exhausts_attempts(self) -> None:
        """Test fails after max attempts."""

        @retry_decorator(max_attempts=2, delay=0.01)
        def always_fails() -> None:
            """Always fail."""
            raise ValueError("Always fails")

        with pytest.raises(ValueError):
            always_fails()


class TestLogCallsDecorator:
    """Test logging decorator."""

    def test_log_calls_decorator_executes_function(self) -> None:
        """Test that function executes normally."""

        @log_calls_decorator
        def multiply(a: int, b: int) -> int:
            """Multiply two numbers."""
            return a * b

        result = multiply(3, 4)
        assert result == 12