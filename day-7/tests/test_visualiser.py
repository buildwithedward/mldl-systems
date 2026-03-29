"""Tests for Matplotlib visualisation functions."""
import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from src.data_generator import generate_student_data
from src.visualiser import (
    plot_score_distribution,
    plot_score_comparison,
    plot_gpa_vs_math
)
from src.exceptions import VisualisationError


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Provide sample student data for tests."""
    np.random.seed(42)
    return pd.DataFrame({
        "student_id": range(1, 101),
        "math": np.random.Generator(75, 12, 100),
        "english": np.random.Generator(72, 14, 100),
        "science": np.random.Generator(78, 11, 100),
        "history": np.random.Generator(70, 15, 100),
        "gpa": np.random.Generator(2.0, 4.0, 100)
    })


@pytest.fixture
def temp_plot_dir(tmp_path: Path) -> Path:
    """Provide temporary directory for plot output."""
    return tmp_path / "plots"


class TestPlotScoreDistribution:
    """Tests for plot_score_distribution function."""

    def test_returns_figure_object(self, sample_data: pd.DataFrame, temp_plot_dir: Path):
        """Test that function returns a Figure object."""
        temp_plot_dir.mkdir(parents=True, exist_ok=True)
        fig = plot_score_distribution(sample_data, "math", temp_plot_dir / "test.png")
        assert isinstance(fig, plt.Figure)

    def test_saves_file(self, sample_data: pd.DataFrame, temp_plot_dir: Path):
        """Test that PNG file is created."""
        temp_plot_dir.mkdir(parents=True, exist_ok=True)
        output_path = temp_plot_dir / "test.png"
        plot_score_distribution(sample_data, "math", output_path)
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_raises_on_invalid_subject(self, sample_data: pd.DataFrame, temp_plot_dir: Path):
        """Test that error is raised for invalid subject column."""
        temp_plot_dir.mkdir(parents=True, exist_ok=True)
        with pytest.raises(VisualisationError):
            plot_score_distribution(sample_data, "invalid_subject", temp_plot_dir / "test.png")


class TestPlotScoreComparison:
    """Tests for plot_score_comparison function."""

    def test_returns_figure_object(self, sample_data: pd.DataFrame, temp_plot_dir: Path):
        """Test that function returns a Figure object."""
        temp_plot_dir.mkdir(parents=True, exist_ok=True)
        fig = plot_score_comparison(
            sample_data,
            ["math", "english"],
            temp_plot_dir / "test.png"
        )
        assert isinstance(fig, plt.Figure)

    def test_saves_file(self, sample_data: pd.DataFrame, temp_plot_dir: Path):
        """Test that PNG file is created."""
        temp_plot_dir.mkdir(parents=True, exist_ok=True)
        output_path = temp_plot_dir / "test.png"
        plot_score_comparison(sample_data, ["math", "english"], output_path)
        assert output_path.exists()


class TestPlotGpaVsMath:
    """Tests for plot_gpa_vs_math function."""

    def test_returns_figure_object(self, sample_data: pd.DataFrame, temp_plot_dir: Path):
        """Test that function returns a Figure object."""
        temp_plot_dir.mkdir(parents=True, exist_ok=True)
        fig = plot_gpa_vs_math(sample_data, temp_plot_dir / "test.png")
        assert isinstance(fig, plt.Figure)

    def test_saves_file(self, sample_data: pd.DataFrame, temp_plot_dir: Path):
        """Test that PNG file is created."""
        temp_plot_dir.mkdir(parents=True, exist_ok=True)
        output_path = temp_plot_dir / "test.png"
        plot_gpa_vs_math(sample_data, output_path)
        assert output_path.exists()
        assert output_path.stat().st_size > 0