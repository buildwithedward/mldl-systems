"""NumPy-based numerical operations on exam scores."""
import logging
from typing import Tuple, Dict
import numpy as np

from src.exceptions import AnalysisError

logger = logging.getLogger(__name__)


def compute_score_statistics(scores: np.ndarray) -> Dict[str, float]:
    """
    Compute basic statistics from score array.

    Args:
        scores: 1D numpy array of scores (float32 or float64).

    Returns:
        Dictionary with keys: mean, median, std, min, max, q25, q75.

    Raises:
        AnalysisError: If computation fails.
    """
    try:
        if not isinstance(scores, np.ndarray):
            scores = np.array(scores, dtype=np.float32)

        if scores.size == 0:
            raise ValueError("Empty score array")

        stats = {
            "mean": float(np.mean(scores)),
            "median": float(np.median(scores)),
            "std": float(np.std(scores)),
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
            "q25": float(np.percentile(scores, 25)),
            "q75": float(np.percentile(scores, 75)),
        }
        logger.debug(f"Computed stats: mean={stats['mean']:.2f}, std={stats['std']:.2f}")
        return stats

    except Exception as e:
        logger.error(f"Statistics computation failed: {e}")
        raise AnalysisError(f"Failed to compute statistics: {e}")


def normalize_scores(scores: np.ndarray, min_val: float = 0, max_val: float = 100) -> np.ndarray:
    """
    Normalize scores to [0, 1] or custom range using min-max scaling.

    Args:
        scores: 1D numpy array of scores.
        min_val: Minimum value in original range.
        max_val: Maximum value in original range.

    Returns:
        Normalized scores in [0, 1].

    Raises:
        AnalysisError: If normalization fails.
    """
    try:
        if not isinstance(scores, np.ndarray):
            scores = np.array(scores, dtype=np.float32)

        normalized = (scores - min_val) / (max_val - min_val)
        normalized = np.clip(normalized, 0, 1)
        logger.debug(f"Normalized {len(scores)} scores")
        return normalized

    except Exception as e:
        logger.error(f"Normalization failed: {e}")
        raise AnalysisError(f"Failed to normalize scores: {e}")


def compute_grade_distribution(scores: np.ndarray) -> Dict[str, int]:
    """
    Compute histogram of grades from scores.

    Args:
        scores: 1D numpy array of scores.

    Returns:
        Dictionary with grade counts: A, B, C, D, F.

    Raises:
        AnalysisError: If computation fails.
    """
    try:
        if not isinstance(scores, np.ndarray):
            scores = np.array(scores, dtype=np.float32)

        a_count = np.sum(scores >= 90)
        b_count = np.sum((scores >= 80) & (scores < 90))
        c_count = np.sum((scores >= 70) & (scores < 80))
        d_count = np.sum((scores >= 60) & (scores < 70))
        f_count = np.sum(scores < 60)

        distribution = {
            "A": int(a_count),
            "B": int(b_count),
            "C": int(c_count),
            "D": int(d_count),
            "F": int(f_count),
        }
        logger.debug(f"Grade distribution: {distribution}")
        return distribution

    except Exception as e:
        logger.error(f"Grade distribution computation failed: {e}")
        raise AnalysisError(f"Failed to compute grade distribution: {e}")


def compute_percentiles(scores: np.ndarray, percentiles: list) -> Dict[int, float]:
    """
    Compute percentiles for scores.

    Args:
        scores: 1D numpy array of scores.
        percentiles: List of percentile values (0-100).

    Returns:
        Dictionary mapping percentile -> value.

    Raises:
        AnalysisError: If computation fails.
    """
    try:
        if not isinstance(scores, np.ndarray):
            scores = np.array(scores, dtype=np.float32)

        result = {}
        for p in percentiles:
            result[p] = float(np.percentile(scores, p))

        logger.debug(f"Computed {len(percentiles)} percentiles")
        return result

    except Exception as e:
        logger.error(f"Percentile computation failed: {e}")
        raise AnalysisError(f"Failed to compute percentiles: {e}")


def compare_arrays(arr1: np.ndarray, arr2: np.ndarray) -> Dict[str, float]:
    """
    Compare two arrays element-wise and compute correlation.

    Args:
        arr1: First 1D numpy array.
        arr2: Second 1D numpy array (same length as arr1).

    Returns:
        Dictionary with correlation and mean difference.

    Raises:
        AnalysisError: If comparison fails.
    """
    try:
        if not isinstance(arr1, np.ndarray):
            arr1 = np.array(arr1, dtype=np.float32)
        if not isinstance(arr2, np.ndarray):
            arr2 = np.array(arr2, dtype=np.float32)

        if arr1.shape != arr2.shape:
            raise ValueError(f"Array shapes don't match: {arr1.shape} vs {arr2.shape}")

        correlation = float(np.corrcoef(arr1, arr2)[0, 1])
        mean_diff = float(np.mean(np.abs(arr1 - arr2)))

        result = {"correlation": correlation, "mean_difference": mean_diff}
        logger.debug(f"Compared arrays: correlation={correlation:.3f}")
        return result

    except Exception as e:
        logger.error(f"Array comparison failed: {e}")
        raise AnalysisError(f"Failed to compare arrays: {e}")