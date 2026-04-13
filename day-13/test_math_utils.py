# test_maths_utils.py — save this file
import numpy as np
import pytest
from math_utils import normalise, rmse

# ── normalise() tests ───────────────────────────────────────────
def test_normalise_mean_is_zero():
    """After normalising, mean should be ~0."""
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = normalise(data)
    assert abs(result.mean()) < 1e-10          # mean very close to 0

def test_normalise_std_is_one():
    """After normalising, std should be ~1."""
    data = np.array([10.0, 20.0, 30.0])
    result = normalise(data)
    assert abs(result.std() - 1.0) < 1e-10    # std very close to 1

def test_normalise_preserves_length():
    """Output should have the same number of elements as input."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal(50)
    assert len(normalise(data)) == len(data)

# ── rmse() tests ─────────────────────────────────────────────────
def test_rmse_perfect_predictions():
    """RMSE should be 0 when predictions match exactly."""
    y = np.array([1.0, 2.0, 3.0])
    assert rmse(y, y) == pytest.approx(0.0)    # use approx for floats

def test_rmse_known_value():
    """RMSE of [0,0] vs [3,4] = sqrt((9+16)/2) = sqrt(12.5) ≈ 3.536"""
    y_true = np.array([0.0, 0.0])
    y_pred = np.array([3.0, 4.0])
    assert rmse(y_true, y_pred) == pytest.approx(3.5355, rel=1e-3)

def test_rmse_is_always_positive():
    """RMSE can never be negative."""
    rng = np.random.default_rng(42)
    y_true = rng.standard_normal(100)
    y_pred = rng.standard_normal(100)
    assert rmse(y_true, y_pred) >= 0