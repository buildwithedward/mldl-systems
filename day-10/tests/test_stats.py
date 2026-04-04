import logging
from typing import List

import pytest
from stats import bayes_update, is_significant, mean, std_dev, variance

# Get logger for test output
logger = logging.getLogger(__name__)


# ============================================================================
# TESTS FOR mean()
# ============================================================================


class TestMean:
    """Test the mean() function."""

    def test_mean_happy_path(self, simple_numbers):
        """Test normal case: positive integers."""
        result = mean(simple_numbers)
        logger.info(f"mean(simple_numbers) = {result}")
        assert result == 3.0

    def test_mean_single_number(self, single_number):
        """Edge case: single value."""
        result = mean(single_number)
        logger.info(f"mean(single_number) = {result}")
        assert result == 42.0

    def test_mean_all_same(self, duplicate_numbers):
        """Edge case: all values identical."""
        result = mean(duplicate_numbers)
        logger.info(f"mean(duplicate_numbers) = {result}")
        assert result == 7.0

    def test_mean_with_negatives(self, negative_numbers):
        """Include negative numbers."""
        result = mean(negative_numbers)
        logger.info(f"mean(negative_numbers) = {result}")
        assert result == pytest.approx(0.0)

    def test_mean_empty_list_raises(self):
        """Empty list should raise ValueError."""
        logger.info("Testing mean([]) raises ValueError")
        with pytest.raises(ValueError, match="Cannot calculate mean"):
            mean([])

    def test_mean_with_floats(self):
        """Floats should work correctly."""
        result = mean([1.5, 2.5, 3.5])
        logger.info(f"mean([1.5, 2.5, 3.5]) = {result}")
        assert result == pytest.approx(2.5)

    @pytest.mark.parametrize(
        "numbers,expected",
        [
            ([1, 2, 3], 2.0),
            ([10, 20], 15.0),
            ([5], 5.0),
            ([-1, -2, -3], -2.0),
            ([0, 0, 0], 0.0),
            ([1.1, 2.2, 3.3], 2.2),
        ],
    )
    def test_mean_parametrised(self, numbers, expected):
        """Parametrised: test many cases at once."""
        result = mean(numbers)
        logger.info(f"mean({numbers}) = {result}, expected = {expected}")
        assert result == pytest.approx(expected)

    def test_mean_non_numeric_raises(self):
        """Non-numeric values should raise TypeError."""
        logger.info("Testing mean() with non-numeric values")
        with pytest.raises(TypeError):
            mean([1, 2, "three"])


# ============================================================================
# TESTS FOR variance()
# ============================================================================


class TestVariance:
    """Test the variance() function."""

    def test_variance_happy_path(self, simple_numbers):
        """Test normal case."""
        result = variance(simple_numbers)
        expected = 2.0
        logger.info(f"variance(simple_numbers) = {result}, expected = {expected}")
        assert result == pytest.approx(expected)

    def test_variance_two_elements(self):
        """Edge case: minimum valid input."""
        result = variance([1.0, 3.0])
        expected = 1.0
        logger.info(f"variance([1.0, 3.0]) = {result}, expected = {expected}")
        assert result == pytest.approx(expected)

    def test_variance_all_same(self, duplicate_numbers):
        """Edge case: variance of identical values is zero."""
        result = variance(duplicate_numbers)
        logger.info(f"variance(duplicate_numbers) = {result}, expected = 0.0")
        assert result == pytest.approx(0.0)

    def test_variance_empty_list_raises(self):
        """Empty list should raise ValueError."""
        logger.info("Testing variance([]) raises ValueError")
        with pytest.raises(ValueError, match="Cannot calculate variance"):
            variance([])

    def test_variance_single_element_raises(self):
        """Single element is not enough for variance."""
        logger.info("Testing variance([42.0]) raises ValueError")
        with pytest.raises(ValueError, match="at least 2 values"):
            variance([42.0])

    def test_variance_negative_numbers(self, negative_numbers):
        """Variance of numbers including negatives."""
        result = variance(negative_numbers)
        expected = 10.4
        logger.info(f"variance(negative_numbers) = {result}, expected = {expected}")
        assert result == pytest.approx(expected)

    @pytest.mark.parametrize(
        "numbers,expected",
        [
            ([1, 2, 3], 2.0 / 3.0),
            ([1, 3], 1.0),
            ([0, 0], 0.0),
            ([2, 2, 2, 2], 0.0),
        ],
    )
    def test_variance_parametrised(self, numbers, expected):
        """Parametrised variance tests."""
        result = variance(numbers)
        logger.info(f"variance({numbers}) = {result}, expected = {expected}")
        assert result == pytest.approx(expected)


# ============================================================================
# TESTS FOR std_dev()
# ============================================================================


class TestStdDev:
    """Test the std_dev() function."""

    def test_std_dev_happy_path(self, simple_numbers):
        """Test normal case."""
        result = std_dev(simple_numbers)
        expected = 2.0**0.5
        logger.info(f"std_dev(simple_numbers) = {result}, expected = {expected}")
        assert result == pytest.approx(expected)

    def test_std_dev_all_same(self, duplicate_numbers):
        """Edge case: std dev of identical values is zero."""
        result = std_dev(duplicate_numbers)
        logger.info(f"std_dev(duplicate_numbers) = {result}, expected = 0.0")
        assert result == pytest.approx(0.0)

    def test_std_dev_empty_list_raises(self):
        """Empty list should raise ValueError."""
        logger.info("Testing std_dev([]) raises ValueError")
        with pytest.raises(ValueError):
            std_dev([])

    def test_std_dev_single_element_raises(self):
        """Single element is not enough."""
        logger.info("Testing std_dev([42.0]) raises ValueError")
        with pytest.raises(ValueError):
            std_dev([42.0])


# ============================================================================
# TESTS FOR is_significant()
# ============================================================================


class TestIsSignificant:
    """Test the is_significant() function."""

    def test_significant_below_threshold(self):
        """p-value < alpha → True."""
        result = is_significant(0.03, 0.05)
        logger.info(f"is_significant(0.03, 0.05) = {result}, expected = True")
        assert result is True

    def test_not_significant_above_threshold(self):
        """p-value > alpha → False."""
        result = is_significant(0.08, 0.05)
        logger.info(f"is_significant(0.08, 0.05) = {result}, expected = False")
        assert result is False

    def test_boundary_equals_alpha(self):
        """p-value == alpha → False (not strictly less)."""
        result = is_significant(0.05, 0.05)
        logger.info(f"is_significant(0.05, 0.05) = {result}, expected = False")
        assert result is False

    def test_boundary_just_below_alpha(self):
        """Just below alpha threshold."""
        result = is_significant(0.049999, 0.05)
        logger.info(f"is_significant(0.049999, 0.05) = {result}, expected = True")
        assert result is True

    def test_custom_alpha(self):
        """Different significance level."""
        result = is_significant(0.08, alpha=0.10)
        logger.info(f"is_significant(0.08, alpha=0.10) = {result}, expected = True")
        assert result is True

    def test_p_value_zero(self):
        """p-value = 0 is always significant."""
        result = is_significant(0.0, 0.05)
        logger.info(f"is_significant(0.0, 0.05) = {result}, expected = True")
        assert result is True

    def test_p_value_one(self):
        """p-value = 1 is never significant."""
        result = is_significant(1.0, 0.05)
        logger.info(f"is_significant(1.0, 0.05) = {result}, expected = False")
        assert result is False

    def test_invalid_p_value_raises(self):
        """p-value outside [0, 1] raises ValueError."""
        logger.info("Testing is_significant with invalid p_value")
        with pytest.raises(ValueError, match="p_value must be between"):
            is_significant(-0.05, 0.05)

        with pytest.raises(ValueError, match="p_value must be between"):
            is_significant(1.5, 0.05)

    def test_invalid_alpha_raises(self):
        """alpha outside [0, 1] raises ValueError."""
        logger.info("Testing is_significant with invalid alpha")
        with pytest.raises(ValueError, match="alpha must be between"):
            is_significant(0.05, -0.05)

    @pytest.mark.parametrize(
        "p_value,alpha,expected",
        [
            (0.01, 0.05, True),
            (0.05, 0.05, False),
            (0.10, 0.05, False),
            (0.001, 0.01, True),
            (0.0, 0.05, True),
            (1.0, 0.05, False),
        ],
    )
    def test_is_significant_parametrised(self, p_value, alpha, expected):
        """Parametrised tests for various thresholds."""
        result = is_significant(p_value, alpha)
        logger.info(
            f"is_significant({p_value}, {alpha}) = {result}, expected = {expected}"
        )
        assert result is expected


# ============================================================================
# TESTS FOR bayes_update()
# ============================================================================


class TestBayesUpdate:
    """Test the bayes_update() function."""

    def test_bayes_update_basic(self):
        """Test standard Bayes' Theorem calculation."""
        result = bayes_update(0.5, 0.9)
        expected = 0.5 * 0.9 / (0.5 * 0.9 + 0.5 * 0.1)
        logger.info(f"bayes_update(0.5, 0.9) = {result}, expected ≈ {expected}")
        assert result == pytest.approx(expected)
        assert result == pytest.approx(0.9)

    def test_bayes_update_weak_prior(self):
        """Low prior gets updated strongly by evidence."""
        result = bayes_update(prior=0.1, likelihood=0.8)
        expected = 0.8 * 0.1 / (0.8 * 0.1 + 0.2 * 0.9)
        logger.info(f"bayes_update(0.1, 0.8) = {result}, expected ≈ {expected}")
        assert result == pytest.approx(expected)
        assert result > 0.1  # Posterior > prior

    def test_bayes_update_strong_prior(self):
        """High prior stays high unless evidence is strong."""
        result = bayes_update(prior=0.9, likelihood=0.6)
        logger.info(f"bayes_update(0.9, 0.6) = {result}, expected > 0.9")
        assert result > 0.9  # Still high

    def test_bayes_update_explicit_complement(self):
        """Can specify likelihood_complement explicitly."""
        result1 = bayes_update(0.5, 0.8, likelihood_complement=0.3)
        result2 = bayes_update(0.5, 0.8)  # Default: 1 - 0.8 = 0.2
        logger.info(f"explicit complement: {result1}, default complement: {result2}")
        assert result1 != result2

    def test_bayes_update_zero_prior(self):
        """Posterior is zero if prior is zero."""
        result = bayes_update(0.0, 0.9)
        logger.info(f"bayes_update(0.0, 0.9) = {result}, expected = 0.0")
        assert result == 0.0

    def test_bayes_update_one_prior(self):
        """Posterior stays high if prior is very high."""
        result = bayes_update(1.0, 0.5)
        logger.info(f"bayes_update(1.0, 0.5) = {result}, expected = 1.0")
        assert result == 1.0

    def test_bayes_update_invalid_prior_raises(self):
        """Prior outside [0, 1] raises ValueError."""
        logger.info("Testing bayes_update with invalid prior")
        with pytest.raises(ValueError, match="prior must be between"):
            bayes_update(-0.1, 0.5)

        with pytest.raises(ValueError, match="prior must be between"):
            bayes_update(1.5, 0.5)

    def test_bayes_update_invalid_likelihood_raises(self):
        """Likelihood outside [0, 1] raises ValueError."""
        logger.info("Testing bayes_update with invalid likelihood")
        with pytest.raises(ValueError, match="likelihood must be between"):
            bayes_update(0.5, 1.5)

    def test_bayes_update_invalid_complement_raises(self):
        """Complement outside [0, 1] raises ValueError."""
        logger.info("Testing bayes_update with invalid complement")
        with pytest.raises(ValueError, match="likelihood_complement must be between"):
            bayes_update(0.5, 0.8, likelihood_complement=1.5)

    @pytest.mark.parametrize(
        "prior,likelihood,expected_comparison",
        [
            (0.5, 0.9, "greater"),  # Strong evidence increases posterior
            (0.5, 0.5, "equal"),  # Equal evidence keeps it steady
            (0.5, 0.1, "less"),  # Weak evidence decreases posterior
        ],
    )
    def test_bayes_update_parametrised(self, prior, likelihood, expected_comparison):
        """Parametrised Bayes update tests."""
        result = bayes_update(prior, likelihood)
        logger.info(f"bayes_update({prior}, {likelihood}) = {result}")

        if expected_comparison == "greater":
            assert result > prior or result == pytest.approx(prior)
        elif expected_comparison == "equal":
            assert result == pytest.approx(prior)
        elif expected_comparison == "less":
            assert result < prior or result == pytest.approx(prior)
