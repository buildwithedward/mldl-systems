"""Main data processing pipeline."""
import logging
from pathlib import Path
from typing import Dict, Any
import pandas as pd

from src.config import settings
from src.data_generator import generate_exam_dataset, save_dataset, load_dataset
from src.numpy_ops import compute_score_statistics, compute_grade_distribution, normalize_scores
from src.pandas_ops import (
    validate_dataframe,
    aggregate_scores_by_subject,
    compute_student_average,
    merge_student_and_subject_stats,
    get_summary_statistics,
)
from src.exceptions import DataPipelineError

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    """Configure logging for the pipeline."""
    log_file = Path(settings.log_dir) / settings.log_file

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ]
    )
    logger.info("Logging initialized")


def run_pipeline() -> Dict[str, Any]:
    """
    Execute full data analysis pipeline.

    Returns:
        Dictionary with analysis results.

    Raises:
        DataPipelineError: If any pipeline step fails.
    """
    try:
        setup_logging()
        logger.info("=" * 60)
        logger.info("STARTING DATA ANALYSIS PIPELINE")
        logger.info("=" * 60)

        # Step 1: Generate dataset
        logger.info("Step 1: Generating dataset...")
        df = generate_exam_dataset(
            num_students=settings.num_students,
            num_subjects=settings.num_subjects,
            num_exams=settings.num_exams,
            seed=settings.seed,
        )

        # Step 2: Save dataset
        csv_path = Path(settings.data_dir) / "exam_scores.csv"
        save_dataset(df, str(csv_path))

        # Step 3: Validate data
        logger.info("Step 2: Validating data...")
        validate_dataframe(df)

        # Step 4: NumPy operations
        logger.info("Step 3: Computing NumPy statistics...")
        scores_array = df["score"].values
        numpy_stats = compute_score_statistics(scores_array)
        grade_dist = compute_grade_distribution(scores_array)
        logger.info(f"Score mean: {numpy_stats['mean']:.2f}, std: {numpy_stats['std']:.2f}")
        logger.info(f"Grade distribution: {grade_dist}")

        # Step 5: Pandas aggregations
        logger.info("Step 4: Computing Pandas aggregations...")
        subject_stats = aggregate_scores_by_subject(df)
        student_avg = compute_student_average(df)
        logger.info(f"\nTop 5 students:\n{student_avg.head()}")

        # Step 6: Merge operations
        logger.info("Step 5: Merging dataframes...")
        merged_df = merge_student_and_subject_stats(df, student_avg, subject_stats)

        # Step 7: Overall summary
        logger.info("Step 6: Computing summary statistics...")
        summary = get_summary_statistics(df)
        logger.info(f"Summary: {summary}")

        # Step 8: Save report
        logger.info("Step 7: Saving report...")
        report_path = Path(settings.data_dir) / "summary_report.csv"
        subject_stats.to_csv(report_path, index=False)
        logger.info(f"Report saved to {report_path}")

        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)

        return {
            "dataset_rows": len(df),
            "dataset_path": str(csv_path),
            "numpy_stats": numpy_stats,
            "grade_distribution": grade_dist,
            "subject_stats": subject_stats.to_dict(orient="records"),
            "top_students": student_avg.head(5).to_dict(orient="records"),
            "summary": summary,
            "report_path": str(report_path),
        }

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise DataPipelineError(f"Pipeline execution failed: {e}")


if __name__ == "__main__":
    result = run_pipeline()
    print("\n" + "=" * 60)
    print("PIPELINE RESULTS")
    print("=" * 60)
    print(f"Processed {result['dataset_rows']} records")
    print(f"Dataset saved to: {result['dataset_path']}")
    print(f"Report saved to: {result['report_path']}")