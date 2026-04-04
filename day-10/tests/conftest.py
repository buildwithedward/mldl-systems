from typing import List

import pytest


@pytest.fixture
def simple_numbers() -> List[float]:
    """Simple dataset for testing."""
    return [1.0, 2.0, 3.0, 4.0, 5.0]


@pytest.fixture
def large_numbers() -> List[float]:
    """Larger dataset for testing edge cases."""
    return list(range(1, 101))  # 1 to 100


@pytest.fixture
def negative_numbers() -> List[float]:
    """Numbers including negatives."""
    return [-5.0, -2.0, 0.0, 2.0, 5.0]


@pytest.fixture
def single_number() -> List[float]:
    """Single value (edge case)."""
    return [42.0]


@pytest.fixture
def duplicate_numbers() -> List[float]:
    """All values the same."""
    return [7.0, 7.0, 7.0, 7.0]
