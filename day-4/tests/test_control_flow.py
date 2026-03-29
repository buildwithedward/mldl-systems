"""Tests for src/control_flow.py."""

import pytest
from src.control_flow import find_best_lr, run_training_loop, simulate_dataloader


class TestSimulateDataloader:
    def test_correct_batch_count(self) -> None:
        # 100 samples / 32 per batch = ceil(100/32) = 4 batches
        assert len(list(simulate_dataloader(100, 32))) == 4

    def test_last_batch_is_partial(self) -> None:
        batches = list(simulate_dataloader(100, 32))
        assert len(batches[-1]) == 4  # 100 - 96 = 4 remaining


class TestRunTrainingLoop:
    def test_one_loss_per_epoch(self) -> None:
        assert len(run_training_loop(3, 50, 10)) == 3

    def test_all_losses_positive(self) -> None:
        assert all(l > 0 for l in run_training_loop(5, 100, 20))

    def test_callback_fires_each_epoch(self) -> None:
        calls: list[tuple[int, float]] = []
        run_training_loop(3, 30, 10, on_epoch_end=lambda e, l: calls.append((e, l)))
        assert len(calls) == 3
        assert calls[0][0] == 1


class TestFindBestLR:
    def test_picks_lr_with_lowest_loss(self) -> None:
        best_lr, best_loss = find_best_lr([0.1, 0.01, 0.001], lambda lr: abs(lr - 0.01))
        assert best_lr == pytest.approx(0.01)
        assert best_loss == pytest.approx(0.0)