"""Tests for pipeline integration."""
import pytest
import pandas as pd
from pathlib import Path
from src.pipeline import run_pipeline
from src.config import settings


class TestPipelineIntegration:
    """Test full pipeline execution."""

    def test_pipeline_runs_successfully(self, tmp_path):
        """Test that pipeline completes without error."""
        # Temporarily change settings for test
        original_data_dir = settings.data_dir
        original_log_dir = settings.log_dir

        settings.data_dir = str(tmp_path / "data")
        settings.log_dir = str(tmp_path / "logs")

        try:
            result = run_pipeline()
            assert result is not None
            assert result["dataset_rows"] > 0
            assert "numpy_stats" in result
            assert "summary" in result
        finally:
            settings.data_dir = original_data_dir
            settings.log_dir = original_log_dir

    def test_pipeline_output_keys(self, tmp_path):
        """Test that pipeline output has required keys."""
        settings.data_dir = str(tmp_path / "data")
        settings.log_dir = str(tmp_path / "logs")

        try:
            result = run_pipeline()
            required_keys = {
                "dataset_rows",
                "dataset_path",
                "numpy_stats",
                "grade_distribution",
                "subject_stats",
                "top_students",
                "summary",
                "report_path",
            }
            assert required_keys.issubset(result.keys())
        finally:
            settings.data_dir = "data"
            settings.log_dir = "logs"