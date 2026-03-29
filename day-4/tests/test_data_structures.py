"""Tests for src/data_structures.py."""

import pytest
from src.data_structures import (
    build_hyperparameter_grid,
    create_epoch_log,
    summarise_log,
    update_epoch,
)
from src.exceptions import InvalidBatchSizeError


class TestCreateEpochLog:
    def test_correct_length(self) -> None:
        assert len(create_epoch_log(5)) == 5

    def test_one_indexed_epochs(self) -> None:
        log = create_epoch_log(3)
        assert log[0]["epoch"] == 1
        assert log[2]["epoch"] == 3

    def test_metrics_initially_none(self) -> None:
        for entry in create_epoch_log(3):
            assert entry["train_loss"] is None

    def test_raises_on_zero(self) -> None:
        with pytest.raises(ValueError, match="num_epochs must be >= 1"):
            create_epoch_log(0)


class TestUpdateEpoch:
    def test_updates_correct_row(self) -> None:
        log = create_epoch_log(3)
        update_epoch(log, 2, 0.5, 0.6, 0.85)
        assert log[1]["train_loss"] == 0.5
        assert log[0]["train_loss"] is None  # other rows untouched


class TestBuildHyperparameterGrid:
    def test_cartesian_product_size(self) -> None:
        grid = build_hyperparameter_grid([0.01, 0.001], [32, 64, 128])
        assert len(grid) == 6  # 2 × 3

    def test_raises_on_zero_batch_size(self) -> None:
        with pytest.raises(InvalidBatchSizeError):
            build_hyperparameter_grid([0.01], [0])


class TestSummariseLog:
    def test_finds_best_epoch(self) -> None:
        log = create_epoch_log(3)
        update_epoch(log, 1, 1.0, 0.9, 0.70)
        update_epoch(log, 2, 0.5, 0.4, 0.90)
        update_epoch(log, 3, 0.6, 0.5, 0.85)
        assert summarise_log(log)["best_epoch"] == 2.0

    def test_empty_if_no_completed_epochs(self) -> None:
        assert summarise_log(create_epoch_log(3)) == {}