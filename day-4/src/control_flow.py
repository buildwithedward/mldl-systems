"""Loops and functions — the skeleton of every training loop.

The canonical PyTorch training loop:
    for epoch in range(num_epochs):
        for batch in dataloader:
            output = model(batch)
            loss   = criterion(output, target)
            loss.backward()
            optimiser.step()

This file builds that exact pattern in pure Python so you see
every piece before the framework abstracts it away.
"""

import logging
import math
from collections.abc import Callable, Iterable

logger = logging.getLogger(__name__)


def simulate_dataloader(
    dataset_size: int,
    batch_size: int,
) -> Iterable[list[int]]:
    """Yield sequential mini-batches of sample indices.

    A real PyTorch DataLoader does the same thing — yields
    tensors loaded from disk. This version uses index lists
    so the mechanics are visible without torch installed.

    Args:
        dataset_size: Total number of samples.
        batch_size: Samples per batch.

    Yields:
        Lists of sample indices, length <= batch_size.
    """
    indices = list(range(dataset_size))

    # range(start, stop, step) — the classic batch slicer
    for start in range(0, dataset_size, batch_size):
        batch = indices[start : start + batch_size]
        logger.debug("Batch: indices %d–%d", start, start + len(batch) - 1)
        yield batch


def fake_forward_pass(batch: list[int]) -> float:
    """Simulate a model forward pass → loss.

    In reality: loss = criterion(model(inputs), targets)
    Here we return a deterministic value for reproducible tests.

    Args:
        batch: Sample indices in this mini-batch.

    Returns:
        A fake loss that decreases as batch index increases.
    """
    mean_idx = sum(batch) / len(batch)
    return round(1.0 / (1.0 + math.log1p(mean_idx)), 6)


def run_training_loop(
    num_epochs: int,
    dataset_size: int,
    batch_size: int,
    on_epoch_end: Callable[[int, float], None] | None = None,
) -> list[float]:
    """Simulate a multi-epoch training loop.

    This IS the PyTorch training loop pattern — just swap
    fake_forward_pass with real model + criterion calls.

    Args:
        num_epochs: Full passes over the dataset.
        dataset_size: Total samples in the dataset.
        batch_size: Mini-batch size.
        on_epoch_end: Optional callback called after each epoch.
            Signature: (epoch_number: int, avg_loss: float) -> None.
            Use this for early stopping, W&B logging, checkpointing.

    Returns:
        Per-epoch average losses (length == num_epochs).
    """
    epoch_losses: list[float] = []

    # ── Outer loop: epochs ─────────────────────────────────────────────
    for epoch in range(1, num_epochs + 1):
        batch_losses: list[float] = []

        # ── Inner loop: mini-batches ───────────────────────────────────
        for batch in simulate_dataloader(dataset_size, batch_size):
            loss = fake_forward_pass(batch)
            batch_losses.append(loss)

        # Average loss across all batches this epoch
        avg_loss = sum(batch_losses) / len(batch_losses)
        epoch_losses.append(avg_loss)

        logger.info(
            "Epoch %d/%d complete | avg_loss=%.6f | batches=%d",
            epoch, num_epochs, avg_loss, len(batch_losses),
        )

        # Fire callback if provided (e.g. W&B logger, early stopper)
        if on_epoch_end is not None:
            on_epoch_end(epoch, avg_loss)

    return epoch_losses


def find_best_lr(
    lr_candidates: list[float],
    loss_fn: Callable[[float], float],
) -> tuple[float, float]:
    """Brute-force learning rate search.

    Real LR finding uses Optuna (Day 21) or a LR range test,
    but the loop pattern is identical.

    Args:
        lr_candidates: LR values to evaluate.
        loss_fn: Takes a learning rate, returns a validation loss.

    Returns:
        Tuple of (best_lr, best_loss). Lower loss is better.
    """
    best_lr   = lr_candidates[0]
    best_loss = float("inf")

    # while loop — intentional, to show the pattern
    idx = 0
    while idx < len(lr_candidates):
        lr   = lr_candidates[idx]
        loss = loss_fn(lr)
        logger.debug("LR=%.6f → loss=%.6f", lr, loss)

        if loss < best_loss:
            best_loss = loss
            best_lr   = lr

        idx += 1   # always increment — infinite loop risk if you forget this

    logger.info("Best LR=%.6f | loss=%.6f", best_lr, best_loss)
    return best_lr, best_loss