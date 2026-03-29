"""Decorators for timing, logging, validation, and retry logic."""

import functools
import logging
import time
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def timer_decorator(func: F) -> F:
    """Decorator that times function execution.

    Args:
        func: Function to decorate.

    Returns:
        Wrapped function that logs execution time.
    
    Example:
        @timer_decorator
        def slow_function():
            time.sleep(2)
            return "done"
        
        # Logs: "slow_function took 2.0012s"
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        """Execute function and log elapsed time."""
        start = time.time()
        logger.debug(f"Starting {func.__name__}")
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger.info(f"{func.__name__} took {elapsed:.4f}s")
        return result

    return wrapper  # type: ignore


def validate_types_decorator(**type_checks: type) -> Callable[[F], F]:
    """Decorator that validates argument types at runtime.

    Args:
        **type_checks: Mapping of argument name to expected type.

    Returns:
        Decorator function.
    
    Example:
        @validate_types_decorator(name=str, age=int)
        def create_user(name, age):
            return f"User: {name}, Age: {age}"
        
        create_user(name="John", age=25)  # Works fine
        create_user(name="John", age="25")  # Raises TypeError
    """

    def decorator(func: F) -> F:
        """Inner decorator."""

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            """Validate types and execute function."""
            logger.debug(f"Validating types for {func.__name__}")
            for arg_name, expected_type in type_checks.items():
                if arg_name in kwargs:
                    if not isinstance(kwargs[arg_name], expected_type):
                        logger.error(
                            f"{arg_name} expected {expected_type}, "
                            f"got {type(kwargs[arg_name])}"
                        )
                        raise TypeError(
                            f"{arg_name} must be {expected_type.__name__}, "
                        )
            logger.debug(f"Type validation passed for {func.__name__}")
            return func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


def retry_decorator(
    max_attempts: int = 3, delay: float = 1.0
) -> Callable[[F], F]:
    """Decorator that retries function on exception.

    Args:
        max_attempts: Maximum number of attempts.
        delay: Delay in seconds between retries.

    Returns:
        Decorator function.
    
    Example:
        @retry_decorator(max_attempts=3, delay=0.5)
        def unreliable_api_call():
            import random
            if random.random() < 0.7:  # 70% chance of failure
                raise ConnectionError("API unavailable")
            return {"status": "success"}
        
        # Will retry up to 3 times with 0.5s delay between attempts
    """

    def decorator(func: F) -> F:
        """Inner decorator."""

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            """Retry logic."""
            for attempt in range(1, max_attempts + 1):
                try:
                    logger.debug(f"Attempt {attempt}/{max_attempts} for {func.__name__}")
                    result = func(*args, **kwargs)
                    logger.info(f"{func.__name__} succeeded on attempt {attempt}")
                    return result
                except Exception as e:
                    if attempt == max_attempts:
                        logger.error(f"{func.__name__} failed after {max_attempts} attempts")
                        raise
                    logger.warning(
                        f"{func.__name__} attempt {attempt} failed: {e}. "
                        f"Retrying in {delay}s..."
                    )
                    time.sleep(delay)
            return None

        return wrapper  # type: ignore

    return decorator


def log_calls_decorator(func: F) -> F:
    """Decorator that logs function calls with arguments and return values.

    Args:
        func: Function to decorate.

    Returns:
        Wrapped function that logs calls.
    
    Example:
        @log_calls_decorator
        def calculate_sum(a, b, multiply_by=1):
            return (a + b) * multiply_by
        
        calculate_sum(5, 3, multiply_by=2)
        # Logs: "Calling calculate_sum(5, 3, multiply_by=2)"
        # Logs: "calculate_sum returned 16"
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        """Log call details and execute function."""
        args_repr = [repr(a) for a in args]
        kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
        signature = ", ".join(args_repr + kwargs_repr)
        logger.debug(f"Calling {func.__name__}({signature})")
        result = func(*args, **kwargs)
        logger.debug(f"{func.__name__} returned {result!r}")
        return result

    return wrapper  # type: ignore
