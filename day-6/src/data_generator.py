"""Generate synthetic exam score dataset."""
import logging
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.config import settings
from src.exceptions import DataGenerationError

logger = logging.getLogger(__name__)


def generate_exam_dataset(
    num_students: int = 100,
    num_subjects: int = 5,
    num_exams: int = 4,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic exam score dataset.

    Args:
        num_students: Number of unique students.
        num_subjects: Number of subjects.
        num_exams: Number of exams per subject.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with columns: student_id, name, subject, score, grade, exam_date.

    Raises:
        DataGenerationError: If dataset generation fails.
    """
    try:
        rng = np.random.default_rng(seed)
        logger.info(f"Generating dataset: {num_students} students, {num_subjects} subjects, {num_exams} exams")

        subjects = ["Math", "Science", "English", "History", "Art"][:num_subjects]
        names = [f"Student_{i}" for i in range(1, num_students + 1)]

        records = []
        base_date = datetime(2024, 1, 1)

        for student_idx in range(num_students):
            for subject in subjects:
                for exam_idx in range(num_exams):
                    # Generate score: normally distributed around 75, clipped to [0, 100]
                    score = np.clip(rng.normal(75, 12), 0, 100)

                    # Grade based on score
                    if score >= 90:
                        grade = "A"
                    elif score >= 80:
                        grade = "B"
                    elif score >= 70:
                        grade = "C"
                    elif score >= 60:
                        grade = "D"
                    else:
                        grade = "F"

                    exam_date = base_date + timedelta(days=exam_idx * 30 + rng.integers(0, 5))

                    records.append({
                        "student_id": student_idx + 1,
                        "name": names[student_idx],
                        "subject": subject,
                        "score": round(score, 2),
                        "grade": grade,
                        "exam_date": exam_date,
                    })

        df = pd.DataFrame(records)
        logger.info(f"Generated {len(df)} exam records")
        return df

    except Exception as e:
        logger.error(f"Dataset generation failed: {e}")
        raise DataGenerationError(f"Failed to generate dataset: {e}")


def save_dataset(df: pd.DataFrame, filepath: str) -> None:
    """
    Save dataset to CSV.

    Args:
        df: DataFrame to save.
        filepath: Path to CSV file.

    Raises:
        DataGenerationError: If save fails.
    """
    try:
        df.to_csv(filepath, index=False)
        logger.info(f"Dataset saved to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save dataset: {e}")
        raise DataGenerationError(f"Failed to save dataset: {e}")


def load_dataset(filepath: str) -> pd.DataFrame:
    """
    Load dataset from CSV.

    Args:
        filepath: Path to CSV file.

    Returns:
        Loaded DataFrame.

    Raises:
        DataGenerationError: If load fails.
    """
    try:
        df = pd.read_csv(filepath)
        df["exam_date"] = pd.to_datetime(df["exam_date"])
        logger.info(f"Dataset loaded from {filepath}: {len(df)} rows")
        return df
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise DataGenerationError(f"Failed to load dataset: {e}")