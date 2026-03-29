"""Pandas-based data operations on exam scores."""
import logging
from typing import Dict, List
import pandas as pd
import numpy as np

from src.exceptions import AnalysisError, DataValidationError

logger = logging.getLogger(__name__)


def validate_dataframe(df: pd.DataFrame) -> None:
    """
    Validate exam score DataFrame structure.

    Args:
        df: DataFrame to validate.

    Raises:
        DataValidationError: If validation fails.
    """
    required_cols = {"student_id", "name", "subject", "score", "grade", "exam_date"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise DataValidationError(f"Missing columns: {missing}")

    if df.empty:
        raise DataValidationError("DataFrame is empty")

    if df["score"].isna().any():
        raise DataValidationError("Found null scores")

    if (df["score"] < 0).any() or (df["score"] > 100).any():
        raise DataValidationError("Scores outside [0, 100] range")

    logger.info(f"Validated DataFrame: {len(df)} rows, {len(df.columns)} columns")


def group_by_subject(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Group DataFrame by subject.

    Args:
        df: Input DataFrame.

    Returns:
        Dictionary mapping subject -> subset DataFrame.

    Raises:
        AnalysisError: If grouping fails.
    """
    try:
        validate_dataframe(df)
        grouped = {subject: group_df for subject, group_df in df.groupby("subject")}
        logger.info(f"Grouped {len(df)} rows by {len(grouped)} subjects")
        return grouped

    except Exception as e:
        logger.error(f"Groupby failed: {e}")
        raise AnalysisError(f"Failed to group by subject: {e}")


def aggregate_scores_by_subject(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate scores by subject using groupby.agg().

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with columns: subject, mean_score, std_score, min_score, max_score, count.

    Raises:
        AnalysisError: If aggregation fails.
    """
    try:
        validate_dataframe(df)

        agg_result = df.groupby("subject")["score"].agg(
            mean_score="mean",
            std_score="std",
            min_score="min",
            max_score="max",
            count="count"
        ).reset_index()

        logger.info(f"Aggregated scores for {len(agg_result)} subjects")
        return agg_result

    except Exception as e:
        logger.error(f"Aggregation failed: {e}")
        raise AnalysisError(f"Failed to aggregate scores: {e}")


def compute_student_average(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute average score per student across all subjects and exams.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with columns: student_id, name, average_score.

    Raises:
        AnalysisError: If computation fails.
    """
    try:
        validate_dataframe(df)

        student_avg = df.groupby(["student_id", "name"])["score"].mean().reset_index()
        student_avg.columns = ["student_id", "name", "average_score"]
        student_avg = student_avg.sort_values("average_score", ascending=False)

        logger.info(f"Computed averages for {len(student_avg)} students")
        return student_avg

    except Exception as e:
        logger.error(f"Student average computation failed: {e}")
        raise AnalysisError(f"Failed to compute student averages: {e}")


def filter_by_grade(df: pd.DataFrame, grade: str) -> pd.DataFrame:
    """
    Filter DataFrame by grade.

    Args:
        df: Input DataFrame.
        grade: Target grade (A, B, C, D, F).

    Returns:
        Filtered DataFrame.

    Raises:
        AnalysisError: If filtering fails.
    """
    try:
        validate_dataframe(df)

        valid_grades = {"A", "B", "C", "D", "F"}
        if grade not in valid_grades:
            raise ValueError(f"Invalid grade: {grade}")

        filtered = df[df["grade"] == grade]
        logger.info(f"Filtered {len(filtered)} rows with grade {grade}")
        return filtered

    except Exception as e:
        logger.error(f"Grade filtering failed: {e}")
        raise AnalysisError(f"Failed to filter by grade: {e}")


def merge_student_and_subject_stats(
    df: pd.DataFrame,
    student_avg: pd.DataFrame,
    subject_agg: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge student averages with subject aggregates using left join.

    Args:
        df: Original DataFrame.
        student_avg: Student average DataFrame.
        subject_agg: Subject aggregate DataFrame.

    Returns:
        Merged DataFrame with student and subject info.

    Raises:
        AnalysisError: If merge fails.
    """
    try:
        # Add student average to original df
        merged = df.merge(student_avg[["student_id", "average_score"]], on="student_id", how="left")

        # Add subject stats to merged df
        merged = merged.merge(
            subject_agg[["subject", "mean_score", "std_score"]],
            on="subject",
            how="left",
            suffixes=("_student", "_subject")
        )

        logger.info(f"Merged datasets: {len(merged)} rows")
        return merged

    except Exception as e:
        logger.error(f"Merge failed: {e}")
        raise AnalysisError(f"Failed to merge dataframes: {e}")


def apply_score_transformation(df: pd.DataFrame, transform_func) -> pd.DataFrame:
    """
    Apply transformation function to scores using apply().

    Args:
        df: Input DataFrame.
        transform_func: Function that takes a score and returns transformed score.

    Returns:
        DataFrame with new column "score_transformed".

    Raises:
        AnalysisError: If apply fails.
    """
    try:
        validate_dataframe(df)

        df_copy = df.copy()
        df_copy["score_transformed"] = df_copy["score"].apply(transform_func)

        logger.info(f"Applied transformation to {len(df_copy)} scores")
        return df_copy

    except Exception as e:
        logger.error(f"Transform apply failed: {e}")
        raise AnalysisError(f"Failed to apply transformation: {e}")


def get_summary_statistics(df: pd.DataFrame) -> Dict[str, any]:
    """
    Get comprehensive summary statistics using describe().

    Args:
        df: Input DataFrame.

    Returns:
        Dictionary with overall statistics.

    Raises:
        AnalysisError: If summary fails.
    """
    try:
        validate_dataframe(df)

        summary = {
            "total_records": len(df),
            "unique_students": df["student_id"].nunique(),
            "unique_subjects": df["subject"].nunique(),
            "score_mean": float(df["score"].mean()),
            "score_median": float(df["score"].median()),
            "score_std": float(df["score"].std()),
            "score_min": float(df["score"].min()),
            "score_max": float(df["score"].max()),
        }

        logger.info(f"Summary: {summary['total_records']} records, {summary['unique_students']} students")
        return summary

    except Exception as e:
        logger.error(f"Summary computation failed: {e}")
        raise AnalysisError(f"Failed to compute summary: {e}")