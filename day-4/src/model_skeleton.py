"""OOP: building a complete model class with gradient descent.

This mirrors the PyTorch pattern you'll use from Day 33:
    class MyNet(nn.Module):
        def __init__(self): ...   ← set up layers
        def forward(self, x): ... ← define computation

We use pure Python so every line is visible and testable
without installing PyTorch.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any

from src.exceptions import ModelNotFittedError

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Abstract base class — the contract all models must follow.

    Attributes:
        name: Human-readable model name.
        _is_fitted: Guards against predict() before training.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._is_fitted = False

    @abstractmethod
    def predict(self, x: Any) -> Any:
        """Subclasses must implement a forward pass.

        Args:
            x: Input (type depends on subclass).

        Returns:
            Model prediction (type depends on subclass).
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', fitted={self._is_fitted})"


class LinearRegressionModel(BaseModel):
    """y = w·x + b — trained with manual gradient descent.

    Gradient descent, step by step:
      1. Forward pass: compute ŷ = w·x + b for all samples
      2. Compute MSE loss = mean((ŷ - y)²)
      3. Compute gradients:  ∂L/∂w = (2/n)·Σ(ŷᵢ-yᵢ)·xᵢ
                             ∂L/∂b = (2/n)·Σ(ŷᵢ-yᵢ)
      4. Update:  w ← w - lr·∂L/∂w
                  b ← b - lr·∂L/∂b

    Attributes:
        weight: Learned slope of the line.
        bias:   Learned y-intercept.
    """

    def __init__(self) -> None:
        super().__init__(name="LinearRegressionModel")
        self.weight: float = 0.0
        self.bias:   float = 0.0
        self._history: list[dict[str, Any]] = []

    def predict(self, x: float) -> float:
        """Compute ŷ = w·x + b.

        Args:
            x: Single input scalar.

        Returns:
            Predicted output scalar.

        Raises:
            ModelNotFittedError: If called before fit().
        """
        if not self._is_fitted:
            raise ModelNotFittedError(
                f"Call fit() before predict() on '{self.name}'."
            )
        return self.weight * x + self.bias

    def fit(
        self,
        x_data: list[float],
        y_data: list[float],
        learning_rate: float = 0.01,
        num_steps: int = 200,
    ) -> None:
        """Train via batch gradient descent.

        Args:
            x_data: Input features.
            y_data: Target values (same length as x_data).
            learning_rate: Parameter update step size.
            num_steps: Number of gradient descent iterations.
        """
        n = len(x_data)
        logger.info(
            "Starting fit: n=%d lr=%.4f steps=%d", n, learning_rate, num_steps
        )

        for step in range(num_steps):
            # 1. Forward pass
            preds  = [self.weight * x + self.bias for x in x_data]
            # 2. Loss
            errors = [p - t for p, t in zip(preds, y_data)]
            loss   = sum(e**2 for e in errors) / n
            # 3. Gradients
            grad_w = (2 / n) * sum(e * x for e, x in zip(errors, x_data))
            grad_b = (2 / n) * sum(errors)
            # 4. Update
            self.weight -= learning_rate * grad_w
            self.bias   -= learning_rate * grad_b

            self._history.append({
                "step":   step,
                "loss":   round(loss, 8),
                "weight": round(self.weight, 6),
                "bias":   round(self.bias, 6),
            })

            if step % 50 == 0 or step == num_steps - 1:
                logger.info(
                    "Step %4d | loss=%.6f | w=%.4f | b=%.4f",
                    step, loss, self.weight, self.bias,
                )

        self._is_fitted = True
        logger.info("Fit complete: w=%.4f b=%.4f", self.weight, self.bias)

    @property
    def loss_history(self) -> list[float]:
        """List of MSE values across all training steps."""
        return [h["loss"] for h in self._history]

    @property
    def did_converge(self) -> bool:
        """True if last 10 losses changed by < 1e-6."""
        if len(self._history) < 10:
            return False
        recent = [h["loss"] for h in self._history[-10:]]
        return max(recent) - min(recent) < 1e-6

    @classmethod
    def from_checkpoint(cls, weight: float, bias: float) -> "LinearRegressionModel":
        """Load a pre-trained model from saved parameters.

        Args:
            weight: Saved weight value.
            bias: Saved bias value.

        Returns:
            A fitted LinearRegressionModel with the loaded params.
        """
        model = cls()
        model.weight     = weight
        model.bias       = bias
        model._is_fitted = True
        logger.info("Checkpoint loaded: w=%.4f b=%.4f", weight, bias)
        return model

    @staticmethod
    def compute_mse(predictions: list[float], targets: list[float]) -> float:
        """Compute mean squared error between two lists.

        Args:
            predictions: Model output values.
            targets: Ground-truth values.

        Returns:
            MSE scalar.

        Raises:
            ValueError: If lists have different lengths.
        """
        if len(predictions) != len(targets):
            raise ValueError(
                f"Length mismatch: {len(predictions)} vs {len(targets)}"
            )
        return sum((p - t)**2 for p, t in zip(predictions, targets)) / len(predictions)

    def __repr__(self) -> str:
        return (
            f"LinearRegressionModel("
            f"w={self.weight:.4f}, "
            f"b={self.bias:.4f}, "
            f"converged={self.did_converge})"
        )