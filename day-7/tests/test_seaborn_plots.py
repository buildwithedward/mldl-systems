"""Tests for Seaborn visualisation functions."""
import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from src.seaborn_plots import (
    plot_correlation_heatmap,
    plot_subject_distributions,
    plot_violin_comparison
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


class TestPlotCorrelationHeatmap:
    """Tests for plot_correlation_heatmap function."""

    def test_returns_figure_object(self, sample_data: pd.DataFrame, temp_plot_dir: Path):
        """Test that function returns a Figure object."""
        temp_plot_dir.mkdir(parents=True, exist_ok=True)
        fig = plot_correlation_heatmap(sample_data, temp_plot_dir / "test.png")
        assert isinstance(fig, plt.Figure)

    def test_saves_file(self, sample_data: pd.DataFrame, temp_plot_dir: Path):
        """Test that PNG file is created."""
        temp_plot_dir.mkdir(parents=True, exist_ok=True)
        output_path = temp_plot_dir / "test.png"
        plot_correlation_heatmap(sample_data, output_path)
        assert output_path.exists()


class TestPlotSubjectDistributions:
    """Tests for plot_subject_distributions function."""

    def test_returns_figure_object(self, sample_data: pd.DataFrame, temp_plot_dir: Path):
        """Test that function returns a Figure object."""
        temp_plot_dir.mkdir(parents=True, exist_ok=True)
        fig = plot_subject_distributions(sample_data, temp_plot_dir / "test.png")
        assert isinstance(fig, plt.Figure)

    def test_saves_file(self, sample_data: pd.DataFrame, temp_plot_dir: Path):
        """Test that PNG file is created."""
        temp_plot_dir.mkdir(parents=True, exist_ok=True)
        output_path = temp_plot_dir / "test.png"
        plot_subject_distributions(sample_data, output_path)
        assert output_path.exists()
        assert output_path.stat().st_size > 0


class TestPlotViolinComparison:
    """Tests for plot_violin_comparison function."""

    def test_returns_figure_object(self, sample_data: pd.DataFrame, temp_plot_dir: Path):
        """Test that function returns a Figure object."""
        temp_plot_dir.mkdir(parents=True, exist_ok=True)
        fig = plot_violin_comparison(sample_data, temp_plot_dir / "test.png")
        assert isinstance(fig, plt.Figure)

    def test_saves_file(self, sample_data: pd.DataFrame, temp_plot_dir: Path):
        """Test that PNG file is created."""
        temp_plot_dir.mkdir(parents=True, exist_ok=True)
        output_path = temp_plot_dir / "test.png"
        plot_violin_comparison(sample_data, output_path)
        assert output_path.exists()