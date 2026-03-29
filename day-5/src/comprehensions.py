"""List, set and dict comprehensions with examples."""

import logging
from typing import Dict, List, Set

logger = logging.getLogger(__name__)


def filter_even_numbers(numbers: List[int]) -> List[int]:
    """Filter even numbers from a list using a list comprehension."""
    logger.debug(f"Filtering {len(numbers)} numbers.")
    return [num for num in numbers if num % 2 == 0]

def square_numbers(numbers: List[int]) -> List[int]:
    logger.debug(f"Squaring {len(numbers)} numbers.")
    """Square numbers in a list using a list comprehension."""
    return [num ** 2 for num in numbers]

def dict_from_lists(keys: List[str], values: List[int]) -> Dict[str, int]:
    """Create a dictionary from two lists using a dict comprehension."""
    # This function takes two lists as input: keys (strings) and values (integers)
    # It returns a dictionary where each key from the keys list is paired with the corresponding value from the values list
    # For example: keys=['a', 'b', 'c'], values=[1, 2, 3] would return {'a': 1, 'b': 2, 'c': 3}
    # If the lists have different lengths, it raises a ValueError
    if len(keys) != len(values):
        raise ValueError("Keys and values lists must be of the same length.")
    logger.debug(f"Creating dictionary from {len(keys)} keys and values.")
    return dict(zip(keys, values))


def nested_list_comprehension(matrix: List[List[int]]) -> List[int]:
    """Flatten 2D matrix using nested comprehension."""
    logger.debug(f"Flattening {len(matrix)}×{len(matrix[0]) if matrix else 0} matrix")
    result = [item for row in matrix for item in row]
    logger.info(f"Flattened {len(matrix)} rows → {len(result)} items")
    return result

def set_unique_words(text: str) -> Set[str]:
    """Extract unique words from a string using a set comprehension."""
    logger.debug("Extracting unique words from text.")
    return {word.lower() for word in text.split() if word.isalpha()}

def dict_comprehension_with_condition(
    data: Dict[str, int], threshold: int
) -> Dict[str, int]:
    """Filter dict by value threshold using dict comprehension."""
    logger.debug(f"Filtering dict with {len(data)} items, threshold={threshold}")
    result = {k: v for k, v in data.items() if v >= threshold}
    logger.info(f"Filtered {len(data)} → {len(result)} items above threshold")
    return result
