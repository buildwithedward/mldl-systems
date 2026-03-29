"""Main pipeline: generate data, create plots, log everything."""
import logging
import sys
from pathlib import Path
from src.config import settings
from src.logger_setup import setup_logging, get_logger
from src.data_generator import generate_student_data
from src.visualiser import (
    plot_score_distribution,
    plot_score_comparison,
    plot_gpa_vs_math
)
from src.seaborn_plots import (
    plot_correlation_heatmap,
    plot_subject_distributions,
    plot_violin_comparison
)

# Setup logging FIRST (before any logging calls)
setup_logging()
logger = get_logger(__name__)


def main() -> None:
    """
    Execute the full data visualisation pipeline.

    Flow:
    1. Generate synthetic student data (500 students, 4 subjects, GPA)
    2. Create Matplotlib plots (distributions, comparisons, scatter)
    3. Create Seaborn plots (heatmap, KDE, violin)
    4. Log all operations to JSON file
    5. Save all plots to output/plots/

    Raises:
        SystemExit: If any step fails.
    """
    try:
        logger.info("Pipeline starting", extra={"phase": "initialization"})

        # Create output directories
        plot_dir = settings.plot_dir_path
        plot_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Output directories created", extra={
            "plot_dir": str(plot_dir)
        })

        # Generate data
        df = generate_student_data()
        logger.info("Data loaded successfully", extra={
            "num_rows": len(df),
            "num_cols": len(df.columns),
            "columns": list(df.columns)
        })

        # Matplotlib plots
        logger.info("Creating Matplotlib plots", extra={"expected_count": 5})

        for subject in ["math", "english", "science"]:
            plot_score_distribution(
                df,
                subject,
                plot_dir / f"01_{subject}_distribution.png"
            )

        plot_score_comparison(
            df,
            ["math", "english", "science", "history"],
            plot_dir / "04_subject_comparison.png"
        )
        plot_gpa_vs_math(df, plot_dir / "05_gpa_vs_math.png")

        logger.info("Matplotlib plots created", extra={"count": 5})

        # Seaborn plots
        logger.info("Creating Seaborn plots", extra={"expected_count": 3})

        plot_correlation_heatmap(df, plot_dir / "06_correlation_heatmap.png")
        plot_subject_distributions(df, plot_dir / "07_kde_distributions.png")
        plot_violin_comparison(df, plot_dir / "08_violin_comparison.png")

        logger.info("Seaborn plots created", extra={"count": 3})

        # Summary
        logger.info("Pipeline completed successfully", extra={
            "total_plots": 8,
            "output_directory": str(plot_dir),
            "log_file": str(settings.log_file_path)
        })

    except Exception as e:
        logger.critical("Pipeline failed", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()