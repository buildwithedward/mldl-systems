"""Tests for pandas operations."""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from src.pandas_ops import (
    validate_dataframe,
    group_by_subject,
    aggregate_scores_by_subject,
    compute_student_average,
    filter_by_grade,
    get_summary_statistics,
)
from src.exceptions import DataValidationError, AnalysisError


@pytest.fixture
def sample_df():
    """Create sample exam score dataframe."""
    return pd.DataFrame({
        "student_id": [1, 1, 2, 2, 3, 3],
        "name": ["Alice", "Alice", "Bob", "Bob", "Charlie", "Charlie"],
        "subject": ["Math", "Science", "Math", "Science", "Math", "Science"],
        "score": [90, 85, 75, 80, 88, 92],
        "grade": ["A", "B", "C", "B", "B", "A"],
        "exam_date": [datetime(2024, 1, 1)] * 6,
    })


class TestValidateDataframe:
    """Test validate_dataframe function."""

    def test_valid_dataframe(self, sample_df):
        """Test validation of valid dataframe."""
        validate_dataframe(sample_df)  # Should not raise

    def test_missing_columns_raises_error(self):
        """Test that missing columns raise error."""
        df = pd.DataFrame({"col1": [1, 2]})
        with pytest.raises(DataValidationError):
            validate_dataframe(df)

    def test_empty_dataframe_raises_error(self):
        """Test that empty dataframe raises error."""
        df = pd.DataFrame(columns=["student_id", "name", "subject", "score", "grade", "exam_date"])
        with pytest.raises(DataValidationError):
            validate_dataframe(df)

    def test_null_scores_raise_error(self):
        """Test that null scores raise error."""
        df = pd.DataFrame({
            "student_id": [1, 2],
            "name": ["A", "B"],
            "subject": ["Math", "Math"],
            "score": [90, np.nan],
            "grade": ["A", "B"],
            "exam_date": [datetime(2024, 1, 1)] * 2,
        })
        with pytest.raises(DataValidationError):
            validate_dataframe(df)

    def test_scores_out_of_range_raise_error(self):
        """Test that out-of-range scores raise error."""
        df = pd.DataFrame({
            "student_id": [1, 2],
            "name": ["A", "B"],
            "subject": ["Math", "Math"],
            "score": [90, 150],  # 150 > 100
            "grade": ["A", "F"],
            "exam_date": [datetime(2024, 1, 1)] * 2,
        })
        with pytest.raises(DataValidationError):
            validate_dataframe(df)


class TestGroupBySubject:
    """Test group_by_subject function."""

    def test_groupby_count(self, sample_df):
        """Test that groupby returns correct number of groups."""
        grouped = group_by_subject(sample_df)
        assert len(grouped) == 2  # Math and Science

    def test_groupby_contains_data(self, sample_df):
        """Test that grouped data is correct."""
        grouped = group_by_subject(sample_df)
        assert len(grouped["Math"]) == 3
        assert len(grouped["Science"]) == 3


class TestAggregateScoresBySubject:
    """Test aggregate_scores_by_subject function."""

    def test_aggregation_shape(self, sample_df):
        """Test aggregation output shape."""
        agg = aggregate_scores_by_subject(sample_df)
        assert len(agg) == 2  # 2 subjects
        assert "mean_score" in agg.columns

    def test_aggregation_values(self, sample_df):
        """Test aggregation values."""
        agg = aggregate_scores_by_subject(sample_df)
        math_row = agg[agg["subject"] == "Math"].iloc[0]

        assert math_row["mean_score"] == pytest.approx((90 + 75 + 88) / 3, abs=0.01)
        assert math_row["count"] == 3


class TestComputeStudentAverage:
    """Test compute_student_average function."""

    def test_student_average_shape(self, sample_df):
        """Test average computation."""
        avg = compute_student_average(sample_df)
        assert len(avg) == 3  # 3 students

    def test_student_average_values(self, sample_df):
        """Test average values."""
        avg = compute_student_average(sample_df)
        alice = avg[avg["student_id"] == 1].iloc[0]

        assert alice["average_score"] == pytest.approx((90 + 85) / 2, abs=0.01)


class TestFilterByGrade:
    """Test filter_by_grade function."""

    def test_filter_grade_a(self, sample_df):
        """Test filtering by grade A."""
        filtered = filter_by_grade(sample_df, "A")
        assert len(filtered) == 2
        assert all(filtered["grade"] == "A")

    def test_invalid_grade_raises_error(self, sample_df):
        """Test that invalid grade raises error."""
        with pytest.raises(AnalysisError):
            filter_by_grade(sample_df, "Z")


class TestGetSummaryStatistics:
    """Test get_summary_statistics function."""

    def test_summary_keys(self, sample_df):
        """Test that summary has required keys."""
        summary = get_summary_statistics(sample_df)

        assert "total_records" in summary
        assert "unique_students" in summary
        assert "score_mean" in summary

    def test_summary_values(self, sample_df):
        """Test summary values."""
        summary = get_summary_statistics(sample_df)
        assert summary["total_records"] == 6
        assert summary["unique_students"] == 3
        assert summary["unique_subjects"] == 2