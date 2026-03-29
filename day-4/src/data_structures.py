"""Lists and dicts in an ML context.

Real use case: every ML dataset is a list of sample dicts.
Example: [{"image": "cat.jpg", "label": 0, "split": "train"}, ...]
This module practises building, mutating, and querying those structures.
"""

import logging
from typing import Any

from src.exceptions import InvalidBatchSizeError

logger = logging.getLogger(__name__)


def create_epoch_log(num_epochs: int) -> list[dict[str, Any]]:
    """Pre-allocate a training log: one dict per epoch.

    This is the structure you'd fill during training and
    later export to MLflow or W&B.

    Args:
        num_epochs: Total number of training epochs.

    Returns:
        List of dicts with sentinel values (metrics set to None).

    Raises:
        ValueError: If num_epochs < 1.
    """
    if num_epochs < 1:
        raise ValueError(f"num_epochs must be >= 1, got {num_epochs}")

    # List comprehension: concise + readable + fast
    log: list[dict[str, Any]] = [
        {
            "epoch":      e,
            "train_loss": None,
            "val_loss":   None,
            "val_acc":    None,
        }
        for e in range(1, num_epochs + 1)
    ]
    logger.debug("Epoch log created: %d entries", len(log))
    return log


def update_epoch(
    log: list[dict[str, Any]],
    epoch: int,
    train_loss: float,
    val_loss: float,
    val_acc: float,
) -> None:
    """Fill in one epoch's metrics in-place.

    Lists are passed by reference — mutating the dict inside the
    list changes the original. The caller sees the update.

    Args:
        log: The epoch log from create_epoch_log().
        epoch: 1-based epoch number to update.
        train_loss: Training loss for this epoch.
        val_loss: Validation loss for this epoch.
        val_acc: Validation accuracy (0.0 – 1.0).
    """
    entry = log[epoch - 1]   # epochs are 1-indexed; list is 0-indexed
    entry["train_loss"] = round(train_loss, 6)
    entry["val_loss"]   = round(val_loss, 6)
    entry["val_acc"]    = round(val_acc, 4)
    logger.info(
        "Epoch %d/%d | train_loss=%.6f val_loss=%.6f val_acc=%.4f",
        epoch, len(log), train_loss, val_loss, val_acc,
    )


def build_hyperparameter_grid(
    learning_rates: list[float],
    batch_sizes: list[int],
) -> list[dict[str, float | int]]:
    """Build all (lr, bs) combinations for a grid search.

    Uses a nested list comprehension — the cartesian product.
    This is how GridSearchCV works internally.

    Args:
        learning_rates: LR values to try.
        batch_sizes: Batch size values to try.

    Returns:
        List of all (lr, bs) combination dicts.

    Raises:
        InvalidBatchSizeError: If any batch size is <= 0.
    """
    for bs in batch_sizes:
        if bs <= 0:
            raise InvalidBatchSizeError(f"Batch size must be > 0, got {bs}")

    grid = [
        {"lr": lr, "bs": bs}
        for lr in learning_rates   # outer loop
        for bs in batch_sizes      # inner loop
    ]
    logger.info("Grid built: %d combinations", len(grid))
    return grid


def summarise_log(log: list[dict[str, Any]]) -> dict[str, float]:
    """Find the best epoch across all completed runs.

    Args:
        log: The epoch log (may be partially filled).

    Returns:
        Dict with best_val_acc, best_val_loss, best_epoch.
        Empty dict if no epochs have been completed.
    """
    completed = [e for e in log if e["val_acc"] is not None]
    if not completed:
        logger.warning("No completed epochs to summarise.")
        return {}

    best = max(completed, key=lambda e: e["val_acc"])
    summary: dict[str, float] = {
        "best_val_acc":  best["val_acc"],
        "best_val_loss": best["val_loss"],
        "best_epoch":    float(best["epoch"]),
    }
    logger.info("Best epoch: %s", summary)
    return summary