import logging
from typing import List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def mean(numbers: List[float]) -> float:
    """
    Calculate the arithmetic mean of a list of numbers.

    Args:
        numbers: List of floats or ints

    Returns:
        The mean as a float

    Raises:
        ValueError: If the list is empty
        TypeError: If numbers contains non-numeric values

    Example:
        >>> mean([1, 2, 3])
        2.0
    """
    if not numbers:
        logger.error("Cannot calculate mean of empty list")
        raise ValueError("Cannot calculate mean of empty list")

    try:
        total = sum(numbers)
        result = total / len(numbers)
        logger.debug(f"Calculated mean of {len(numbers)} values: {result}")
        return result
    except TypeError as e:
        logger.error(f"Invalid data type in numbers list: {e}")
        raise TypeError("All elements must be numeric") from e


def variance(numbers: List[float]) -> float:
    """
    Calculate the variance (average squared deviation from mean).

    Args:
        numbers: List of floats or ints

    Returns:
        The variance as a float

    Raises:
        ValueError: If the list is empty or has only one element
        TypeError: If numbers contains non-numeric values

    Example:
        >>> variance([1, 2, 3])
        0.6666666666666666
    """
    if not numbers:
        logger.error("Cannot calculate variance of empty list")
        raise ValueError("Cannot calculate variance of empty list")

    if len(numbers) < 2:
        logger.error("Variance requires at least 2 values")
        raise ValueError(
            "Variance requires at least 2 values (population has no variance)"
        )

    try:
        m = mean(numbers)
        squared_diffs = [(x - m) ** 2 for x in numbers]
        result = sum(squared_diffs) / len(numbers)
        logger.debug(f"Calculated variance: {result}")
        return result
    except TypeError as e:
        logger.error(f"Invalid data type in numbers list: {e}")
        raise TypeError("All elements must be numeric") from e


def std_dev(numbers: List[float]) -> float:
    """
    Calculate the standard deviation (square root of variance).

    Args:
        numbers: List of floats or ints

    Returns:
        The standard deviation as a float

    Raises:
        ValueError: If the list is empty or has only one element
        TypeError: If numbers contains non-numeric values

    Example:
        >>> std_dev([1, 2, 3])
        0.8164965809004287
    """
    if not numbers:
        logger.error("Cannot calculate std dev of empty list")
        raise ValueError("Cannot calculate std dev of empty list")

    try:
        var = variance(numbers)
        result = var**0.5
        logger.debug(f"Calculated std dev: {result}")
        return result
    except (ValueError, TypeError) as e:
        raise


def is_significant(p_value: float, alpha: float = 0.05) -> bool:
    """
    Determine if a p-value is statistically significant.

    In plain English: Is this result unlikely to happen by chance?
    If p_value < alpha, we say "yes, this is surprising" (significant).

    Args:
        p_value: The p-value (must be between 0 and 1)
        alpha: The significance level (default 0.05 = 5%)

    Returns:
        True if p_value < alpha, False otherwise

    Raises:
        ValueError: If p_value or alpha are outside [0, 1]

    Example:
        >>> is_significant(0.03, 0.05)
        True
        >>> is_significant(0.08, 0.05)
        False
    """
    if not (0 <= p_value <= 1):
        logger.error(f"p_value {p_value} is outside [0, 1]")
        raise ValueError("p_value must be between 0 and 1")

    if not (0 <= alpha <= 1):
        logger.error(f"alpha {alpha} is outside [0, 1]")
        raise ValueError("alpha must be between 0 and 1")

    result = p_value < alpha
    logger.debug(f"p_value={p_value}, alpha={alpha} → significant={result}")
    return result


def bayes_update(
    prior: float, likelihood: float, likelihood_complement: Optional[float] = None
) -> float:
    """
    Update a prior probability using Bayes' Theorem.

    In plain English: You had a guess (prior). You saw new evidence (likelihood).
    What should your new guess be (posterior)?

    Bayes' Theorem: P(A|B) = P(B|A) * P(A) / P(B)

    Args:
        prior: Your initial guess (0 to 1)
        likelihood: Probability of evidence given your guess (0 to 1)
        likelihood_complement: Probability of evidence given NOT your guess.
                              If None, assumed to be 1 - likelihood

    Returns:
        The updated probability (posterior)

    Raises:
        ValueError: If any probability is outside [0, 1]

    Example:
        >>> bayes_update(0.5, 0.9)
        0.6428571428571429
    """
    # Validate inputs
    if not (0 <= prior <= 1):
        logger.error(f"prior {prior} is outside [0, 1]")
        raise ValueError("prior must be between 0 and 1")

    if not (0 <= likelihood <= 1):
        logger.error(f"likelihood {likelihood} is outside [0, 1]")
        raise ValueError("likelihood must be between 0 and 1")

    # Default likelihood_complement
    if likelihood_complement is None:
        likelihood_complement = 1 - likelihood

    if not (0 <= likelihood_complement <= 1):
        logger.error(f"likelihood_complement {likelihood_complement} is outside [0, 1]")
        raise ValueError("likelihood_complement must be between 0 and 1")

    # Bayes' Theorem
    posterior = (
        likelihood * prior / (likelihood * prior + likelihood_complement * (1 - prior))
    )

    logger.info(f"Bayes update: prior={prior} → posterior={posterior:.4f}")
    return posterior
