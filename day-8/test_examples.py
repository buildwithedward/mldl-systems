import logging

from gradient_descent import gradient_descent_1d, visualize_gradient_descent
from math_toolkit import (
    cosine_similarity,
    dot_product,
    matrix_transpose,
    matrix_vector_multiply,
    numerical_gradient,
    scalar_multiply,
    vector_add,
    vector_magnitude,
    vector_subtract,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def test_linear_algebra() -> None:
    """Test linear algebra functions."""
    logger.info("=== Testing Linear Algebra ===\n")

    # Test dot product
    a = [1, 2, 3]
    b = [4, 5, 6]
    expected_dot = 1 * 4 + 2 * 5 + 3 * 6  # 32
    actual_dot = dot_product(a, b)
    assert actual_dot == expected_dot, (
        f"Dot product failed: {actual_dot} != {expected_dot}"
    )
    logger.info(f"✓ Dot product: {a} · {b} = {actual_dot}")

    # Test vector magnitude
    v = [3, 4]
    expected_mag = 5.0
    actual_mag = vector_magnitude(v)
    assert abs(actual_mag - expected_mag) < 1e-6, (
        f"Magnitude failed: {actual_mag} != {expected_mag}"
    )
    logger.info(f"✓ Magnitude: |{v}| = {actual_mag}")

    # Test cosine similarity (same vector)
    sim = cosine_similarity(a, a)
    assert abs(sim - 1.0) < 1e-6, f"Cosine similarity failed: {sim} != 1.0"
    logger.info(f"✓ Cosine similarity (same vector): {sim}")

    # Test cosine similarity (perpendicular vectors)
    perp1 = [1, 0]
    perp2 = [0, 1]
    sim_perp = cosine_similarity(perp1, perp2)
    assert abs(sim_perp) < 1e-6, (
        f"Cosine similarity (perpendicular) failed: {sim_perp} != 0"
    )
    logger.info(f"✓ Cosine similarity (perpendicular): {sim_perp}")

    # Test matrix-vector multiplication
    A = [[1, 2], [3, 4], [5, 6]]  # 3×2
    v = [1, 2]
    result = matrix_vector_multiply(A, v)
    expected = [1 * 1 + 2 * 2, 3 * 1 + 4 * 2, 5 * 1 + 6 * 2]  # [5, 11, 17]
    assert result == expected, f"Matrix-vector multiply failed: {result} != {expected}"
    logger.info(f"✓ Matrix-vector multiply: {result}")


def test_calculus() -> None:
    """Test calculus functions."""
    logger.info("=== Testing Calculus ===\n")

    # Test numerical derivative of x²
    f = lambda x: x**2
    x = 3.0
    deriv = numerical_gradient(f, x)
    expected = 2 * x  # d/dx(x²) = 2x
    assert abs(deriv - expected) < 1e-3, f"Derivative failed: {deriv} != {expected}"
    logger.info(f"✓ d/dx(x²) at x={x}: {deriv:.4f} (expected: {expected})")

    # Test numerical derivative of x³
    g = lambda x: x**3
    deriv_g = numerical_gradient(g, x)
    expected_g = 3 * x**2  # d/dx(x³) = 3x²
    assert abs(deriv_g - expected_g) < 1e-2, (
        f"Derivative failed: {deriv_g} != {expected_g}"
    )
    logger.info(f"✓ d/dx(x³) at x={x}: {deriv_g:.4f} (expected: {expected_g})")

    logger.info("")


def test_gradient_descent() -> None:
    """Test gradient descent."""
    logger.info("=== Testing Gradient Descent ===\n")

    # Minimize f(x) = (x-3)²
    def f(x):
        return (x - 3) ** 2

    x_final, x_hist, loss_hist, _ = gradient_descent_1d(
        f, x_init=0.0, learning_rate=0.1, max_iterations=100
    )

    # Should converge to x ≈ 3
    assert abs(x_final - 3.0) < 0.01, f"Gradient descent failed: {x_final} != 3.0"
    logger.info(f"✓ Minimized (x-3)²: x = {x_final:.6f} (expected: 3.0)")
    logger.info(f"✓ Loss decreased from {loss_hist[0]:.6f} to {loss_hist[-1]:.6f}")


if __name__ == "__main__":
    test_linear_algebra()
    test_calculus()
    test_gradient_descent()

    logger.info("=== All Tests Passed! ===\n")
