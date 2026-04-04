import logging
from typing import Callable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from math_toolkit import f, numerical_gradient

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def gradient_descent_1d(
    f: Callable[[float], float],
    x_init: float,
    learning_rate: float,
    max_iterations: int,
    tolerance: float = 1e-6,
) -> Tuple[float, List[float], List[float], List[float]]:
    """
    Perform 1D gradient descent.

    Returns:
        final_x: The x value at the minimum
        x_history: List of x values at each iteration
        loss_history: List of loss values at each iteration
        grad_history: List of gradient magnitudes at each iteration
    """
    x = x_init
    x_history = [x]
    loss_history = [f(x)]
    grad_history = []

    for iteration in range(max_iterations):
        grad = numerical_gradient(f, x)
        grad_history.append(abs(grad))
        x = x - learning_rate * grad
        x_history.append(x)
        loss_history.append(f(x))

        if abs(grad) < tolerance:
            logger.info(f"Converged at iteration {iteration}, gradient: {grad:.2e}")
            break

        if (iteration + 1) % 10 == 0:
            logger.info(
                f"Iteration {iteration + 1}, x: {x:.6f}, loss: {f(x):.6f}, gradient: {grad:.2e}"
            )

    return x, x_history, loss_history, grad_history


def visualize_gradient_descent(
    f: Callable[[float], float],
    x_history: List[float],
    loss_history: List[float],
    x_range: Tuple[float, float] = (-1, 5),
    filename: str = "gradient_descent.png",
) -> None:
    """
    Visualize the gradient descent process.

    Creates a 2-panel plot:
    - Left: Function curve with descent steps marked
    - Right: Loss vs iteration
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left panel: function curve + descent path
    x_vals = np.linspace(x_range[0], x_range[1], 300)
    y_vals = [f(x) for x in x_vals]

    ax1.plot(x_vals, y_vals, "b-", linewidth=2, label="f(x)")

    # Plot descent steps
    for i in range(len(x_history) - 1):
        x_curr = x_history[i]
        y_curr = loss_history[i]
        x_next = x_history[i + 1]
        y_next = loss_history[i + 1]

        # Draw arrow
        ax1.annotate(
            "",
            xy=(x_next, y_next),
            xytext=(x_curr, y_curr),
            arrowprops=dict(arrowstyle="->", color="red", alpha=0.6, lw=1.5),
        )

    # Highlight start and end
    ax1.plot(x_history[0], loss_history[0], "go", markersize=10, label="Start")
    ax1.plot(
        x_history[-1], loss_history[-1], "r*", markersize=15, label="End (minimum)"
    )

    ax1.set_xlabel("x", fontsize=12)
    ax1.set_ylabel("f(x)", fontsize=12)
    ax1.set_title(
        "Gradient Descent: Function & Descent Path", fontsize=14, fontweight="bold"
    )
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Right panel: loss over iterations
    iterations = range(len(loss_history))
    ax2.plot(iterations, loss_history, "b-", linewidth=2, marker="o", markersize=4)
    ax2.set_xlabel("Iteration", fontsize=12)
    ax2.set_ylabel("Loss f(x)", fontsize=12)
    ax2.set_title("Loss Convergence", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    logger.info(f"Plot saved to {filename}")
    plt.close()


# Demonstration
if __name__ == "__main__":
    logger.info("=== Gradient Descent: 1D Example ===\n")

    # Define function: f(x) = x² - 4x + 6
    # Minimum at x = 2, value = 2
    def f(x):
        return x**2 - 4 * x + 6

    logger.info("Objective: minimize f(x) = x² - 4x + 6")
    logger.info("True minimum: x = 2, f(2) = 2\n")

    # Run gradient descent
    x_init = 0.0
    learning_rate = 0.1
    max_iterations = 100

    logger.info(f"Starting from x = {x_init}")
    logger.info(f"Learning rate = {learning_rate}\n")

    x_final, x_history, loss_history, grad_history = gradient_descent_1d(
        f, x_init, learning_rate, max_iterations
    )

    logger.info(f"\nFinal x: {x_final:.6f}")
    logger.info(f"Final loss: {f(x_final):.6f}")
    logger.info(f"Iterations: {len(x_history) - 1}")

    # Visualize
    visualize_gradient_descent(
        f, x_history, loss_history, filename="gradient_descent.png"
    )

    # Try different learning rates
    logger.info("\n=== Testing Different Learning Rates ===\n")

    learning_rates = [0.01, 0.05, 0.1, 0.2]
    for lr in learning_rates:
        x_final, _, loss_hist, _ = gradient_descent_1d(f, 0.0, lr, 100)
        logger.info(
            f"LR={lr}: x={x_final:.6f}, f(x)={f(x_final):.6f}, iterations={len(loss_hist) - 1}"
        )
