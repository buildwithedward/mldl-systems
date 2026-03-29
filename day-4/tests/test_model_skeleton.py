"""Tests for src/model_skeleton.py."""

import pytest
from src.exceptions import ModelNotFittedError
from src.model_skeleton import LinearRegressionModel


class TestInit:
    def test_starts_at_zero(self) -> None:
        m = LinearRegressionModel()
        assert m.weight == 0.0 and m.bias == 0.0

    def test_repr_shows_class_name(self) -> None:
        assert "LinearRegressionModel" in repr(LinearRegressionModel())


class TestPredict:
    def test_raises_before_fit(self) -> None:
        with pytest.raises(ModelNotFittedError):
            LinearRegressionModel().predict(1.0)

    def test_checkpoint_load_enables_predict(self) -> None:
        m = LinearRegressionModel.from_checkpoint(weight=2.0, bias=1.0)
        assert m.predict(3.0) == pytest.approx(7.0)  # 2*3 + 1 = 7


class TestFit:
    def test_loss_decreases(self) -> None:
        m = LinearRegressionModel()
        m.fit([1.0, 2.0, 3.0, 4.0, 5.0], [2.0, 4.0, 6.0, 8.0, 10.0], num_steps=200)
        history = m.loss_history
        assert history[-1] < history[0]

    def test_learns_y_equals_2x(self) -> None:
        m = LinearRegressionModel()
        m.fit([1.0, 2.0, 3.0, 4.0, 5.0], [2.0, 4.0, 6.0, 8.0, 10.0],
              learning_rate=0.01, num_steps=500)
        assert m.weight == pytest.approx(2.0, abs=0.05)
        assert m.bias   == pytest.approx(0.0, abs=0.05)

    def test_learns_y_equals_2x_plus_1(self) -> None:
        m = LinearRegressionModel()
        m.fit([1.0, 2.0, 3.0, 4.0, 5.0], [3.0, 5.0, 7.0, 9.0, 11.0],
              learning_rate=0.01, num_steps=500)
        assert m.weight == pytest.approx(2.0, abs=0.1)
        assert m.bias   == pytest.approx(1.0, abs=0.1)


class TestComputeMSE:
    def test_perfect_predictions(self) -> None:
        assert LinearRegressionModel.compute_mse([1.0, 2.0], [1.0, 2.0]) == pytest.approx(0.0)

    def test_known_value(self) -> None:
        assert LinearRegressionModel.compute_mse([2.0, 3.0, 4.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)

    def test_raises_on_length_mismatch(self) -> None:
        with pytest.raises(ValueError):
            LinearRegressionModel.compute_mse([1.0], [1.0, 2.0])