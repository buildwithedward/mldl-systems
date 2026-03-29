"""Generate synthetic student exam dataset."""
import logging
from typing import Tuple
import numpy as np
import pandas as pd
from src.config import settings
from src.exceptions import DataGenerationError

logger = logging.getLogger(__name__)


def generate_student_data() -> pd.DataFrame:
    """
    Generate synthetic student exam performance data.

    Creates a DataFrame with configurable number of students and their
    exam scores across four subjects, then calculates GPA.

    Returns:
        pd.DataFrame: Columns: student_id, math, english, science, history, gpa

    Raises:
        DataGenerationError: If data generation fails.
    """
    try:
        np.random.seed(settings.data_seed)
        n_students = settings.num_students

        # Log the generation start with metadata
        logger.info("Generating synthetic data", extra={
            "num_students": n_students,
            "seed": settings.data_seed
        })

        # Generate scores (normal distribution, realistic ranges)
        data = {
            "student_id": np.arange(1, n_students + 1),
            "math": np.random.Generator(75, 12, n_students),
            "english": np.random.Generator(72, 14, n_students),
            "science": np.random.Generator(78, 11, n_students),
            "history": np.random.Generator(70, 15, n_students),
        }

        df = pd.DataFrame(data)

        # Calculate GPA (out of 4.0)
        df["gpa"] = (
            df["math"] + df["english"] + df["science"] + df["history"]
        ) / 400 * 4.0
        df["gpa"] = df["gpa"].clip(0, 4.0)  # Clamp to [0, 4.0]

        logger.info("Data generation complete", extra={
            "shape": df.shape,
            "columns": list(df.columns),
            "memory_mb": df.memory_usage(deep=True).sum() / 1024 / 1024
        })

        return df

    except Exception as e:
        # Log with full traceback
        logger.error("Data generation failed", exc_info=True)
        raise DataGenerationError(f"Cannot generate data: {e}") from e