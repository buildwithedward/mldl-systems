import logging
from typing import List, Union

import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def vector_add(a: List[float], b: List[float]) -> List[float]:
    """Add two vectors element-wise."""
    if len(a) != len(b):
        raise ValueError(
            "Vectors must be of the same length: {} vs {}".format(len(a), len(b))
        )
    return [a[i] + b[i] for i in range(len(a))]


def vector_subtract(a: List[float], b: List[float]) -> List[float]:
    """Subtract two vectors element-wise."""
    if len(a) != len(b):
        raise ValueError(
            "Vectors must be of the same length: {} vs {}".format(len(a), len(b))
        )
    return [a[i] - b[i] for i in range(len(a))]


def scalar_multiply(scalar: float, b: List[float]) -> List[float]:
    """Multiply a scalar by a vector element-wise."""
    return [scalar * x for x in b]


def dot_product(
    a: Union[List[float], List[int]],
    b: Union[List[float], List[int]],
) -> float:
    """Compute the dot product of two vectors."""
    if len(a) != len(b):
        raise ValueError(
            "Vectors must be of the same length: {} vs {}".format(len(a), len(b))
        )
    return sum(a[i] * b[i] for i in range(len(a)))


def vector_magnitude(a: Union[List[float], List[int]]) -> float:
    """Compute the magnitude of a vector."""
    return np.sqrt(sum(x**2 for x in a))


def cosine_similarity(
    a: Union[List[float], List[int]], b: Union[List[float], List[int]]
) -> float:
    """Compute the cosine similarity of two vectors."""
    dot = dot_product(a, b)
    mag_a = vector_magnitude(a)
    mag_b = vector_magnitude(b)
    if mag_a == 0 or mag_b == 0:  # handle zero magnitude
        return 0.0
    return dot / (mag_a * mag_b)


def matrix_transpose(A: List[List[float]]) -> List[List[float]]:
    """Compute the transpose of a matrix."""
    rows = len(A)
    cols = len(A[0])
    return [[A[j][i] for j in range(rows)] for i in range(cols)]


def matrix_vector_multiply(A: List[List[float]], v: List[float]) -> List[float]:
    """Multiply a matrix by a vector."""
    rows = len(A)
    result = []
    for i in range(rows):
        dot = dot_product(A[i], v)
        result.append(dot)
    return result


def matrix_matrix_multiply(
    A: List[List[float]], B: List[List[float]]
) -> List[List[float]]:
    """Multiply two matrices: A (m x n) and B (n x p) -> (m x p)"""
    m = len(A)
    n = len(A[0])
    p = len(B[0])

    # Transpose B to make it n x p
    B_T = matrix_transpose(B)
    result = []
    if n != len(B):
        raise ValueError(
            "Matrix dimensions do not match for multiplication: {} x {} vs {} x {}".format(
                m, n, len(B), len(B[0])
            )
        )
    for i in range(m):
        row = []
        for j in range(p):
            dot = dot_product(A[i], B_T[j])
            row.append(dot)
        result.append(row)
    return result


def f2(x):
    return x**3


def f(x):
    return x**2


def numerical_gradient(f, x: float, h: float = 1e-5) -> float:
    """Compute the numerical gradient of a function at a point x."""
    return (f(x + h) - f(x)) / h


def print_vector(name: str, v: Union[List[float], List[int]]) -> None:
    """Helper: log a vector in readable format."""
    logger.info(f"{name} = {[round(x, 4) for x in v]}")


def print_matrix(name: str, M: Union[List[List[float]], List[List[int]]]) -> None:
    """Helper: log a matrix in readable format."""
    logger.info(f"{name} =")
    for row in M:
        logger.info(f"  {[round(x, 4) for x in row]}")


if __name__ == "__main__":
    logger.info("=== Linear Algebra Toolkit ===")

    # Vectors
    a = [1, 2, 3]
    b = [4, 5, 6]

    logger.info(f"a = {a}, b = {b}")
    logger.info(f"a + b = {vector_add(a, b)}")
    logger.info(f"a - b = {vector_subtract(a, b)}")
    logger.info(f"3 × a = {scalar_multiply(3, a)}")
    logger.info(f"a · b = {dot_product(a, b)}")
    logger.info(f"|a| = {vector_magnitude(a):.4f}")
    logger.info(f"cosine_similarity(a, b) = {cosine_similarity(a, b):.4f}")

    # Matrices
    logger.info("\n=== Matrix Operations ===")
    A = [[1, 2, 3], [4, 5, 6]]
    logger.info(f"A shape: {len(A)}×{len(A[0])}")
    print_matrix("A", A)
    print_matrix("A^T", matrix_transpose(A))

    v = [1, 2, 3]
    print_vector("v", v)
    print_vector("A × v", matrix_vector_multiply(A, v))

    # Calculus
    logger.info("\n=== Numerical Derivatives ===")

    def f(x):
        return x**2

    x = 2.0
    derivative = numerical_gradient(f, x)
    logger.info(f"f(x) = x², at x={x}: f'(x) ≈ {derivative:.4f} (true: 4.0)")

    derivative2 = numerical_gradient(f2, x)
    logger.info(f"f(x) = x³, at x={x}: f'(x) ≈ {derivative2:.4f} (true: 12.0)")
