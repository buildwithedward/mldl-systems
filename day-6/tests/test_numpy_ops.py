"""Tests for numpy operations."""
import pytest
import numpy as np
from src.numpy_ops import (
    compute_score_statistics,
    normalize_scores,
    compute_grade_distribution,
    compute_percentiles,
    compare_arrays,
)
from src.exceptions import AnalysisError


class TestComputeScoreStatistics:
    """Test compute_score_statistics function."""

    def test_basic_statistics(self):
        """Test with simple array."""
        scores = np.array([50, 60, 70, 80, 90], dtype=np.float32)
        stats = compute_score_statistics(scores)

        assert stats["mean"] == 70.0
        assert stats["min"] == 50.0
        assert stats["max"] == 90.0

    def test_with_list_input(self):
        """Test with Python list input."""
        scores = [60, 70, 80]
        stats = compute_score_statistics(scores)
        assert stats["mean"] == 70.0

    def test_empty_array_raises_error(self):
        """Test that empty array raises error."""
        with pytest.raises(AnalysisError):
            compute_score_statistics(np.array([], dtype=np.float32))


class TestNormalizeScores:
    """Test normalize_scores function."""

    def test_normalize_to_unit_range(self):
        """Test normalization to [0, 1]."""
        scores = np.array([0, 50, 100], dtype=np.float32)
        normalized = normalize_scores(scores, 0, 100)

        assert normalized[0] == 0.0
        assert normalized[1] == 0.5
        assert normalized[2] == 1.0

    def test_normalized_clipped_to_bounds(self):
        """Test that normalized scores are clipped to [0, 1]."""
        scores = np.array([-10, 110], dtype=np.float32)
        normalized = normalize_scores(scores, 0, 100)

        assert np.all(normalized >= 0)
        assert np.all(normalized <= 1)


class TestComputeGradeDistribution:
    """Test compute_grade_distribution function."""

    def test_grade_boundaries(self):
        """Test grade boundaries."""
        scores = np.array([95, 85, 75, 65, 55], dtype=np.float32)
        dist = compute_grade_distribution(scores)

        assert dist["A"] == 1  # 95
        assert dist["B"] == 1  # 85
        assert dist["C"] == 1  # 75
        assert dist["D"] == 1  # 65
        assert dist["F"] == 1  # 55

    def test_all_same_grade(self):
        """Test when all scores fall in same grade."""
        scores = np.array([92, 94, 96], dtype=np.float32)
        dist = compute_grade_distribution(scores)

        assert dist["A"] == 3
        assert sum(dist.values()) == 3


class TestComputePercentiles:
    """Test compute_percentiles function."""

    def test_basic_percentiles(self):
        """Test percentile computation."""
        scores = np.arange(1, 101, dtype=np.float32)  # 1 to 100
        percentiles = compute_percentiles(scores, [25, 50, 75])

        assert percentiles[25] == pytest.approx(25.75, abs=1)
        assert percentiles[50] == pytest.approx(50.5, abs=1)
        assert percentiles[75] == pytest.approx(75.25, abs=1)


class TestCompareArrays:
    """Test compare_arrays function."""

    def test_perfect_correlation(self):
        """Test perfect correlation."""
        arr1 = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        arr2 = arr1.copy()

        result = compare_arrays(arr1, arr2)
        assert result["correlation"] == pytest.approx(1.0, abs=0.01)

    def test_shape_mismatch_raises_error(self):
        """Test that mismatched shapes raise error."""
        arr1 = np.array([1, 2, 3], dtype=np.float32)
        arr2 = np.array([1, 2], dtype=np.float32)

        with pytest.raises(AnalysisError):
            compare_arrays(arr1, arr2)