"""Matplotlib-based visualisation functions."""
import logging
from typing import Tuple
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from src.config import settings
from src.exceptions import VisualisationError

logger = logging.getLogger(__name__)


def plot_score_distribution(
    df: pd.DataFrame,
    subject: str,
    output_path: Path
) -> plt.Figure:
    """
    Create a histogram of exam scores for a subject.

    Args:
        df: Student DataFrame with score columns.
        subject: Column name (e.g., 'math', 'english').
        output_path: Path where PNG will be saved.

    Returns:
        plt.Figure: The created figure object.

    Raises:
        VisualisationError: If plot creation fails.
    """
    try:
        logger.info("Creating distribution plot", extra={
            "subject": subject,
            "output_file": str(output_path),
            "num_students": len(df)
        })

        # Create figure and axes
        fig, ax = plt.subplots(figsize=(10, 6))

        # Histogram
        ax.hist(
            df[subject],
            bins=30,
            color="#58a6ff",
            alpha=0.8,
            edgecolor="black"
        )

        # Labels and title
        ax.set_xlabel(f"{subject.title()} Score")
        ax.set_ylabel("Number of Students")
        ax.set_title(f"Distribution of {subject.title()} Scores")
        ax.grid(axis="y", alpha=0.3)

        # Save
        fig.tight_layout()
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
        logger.info("Distribution plot saved", extra={
            "path": str(output_path),
            "subject": subject
        })

        return fig

    except Exception as e:
        logger.error("Plot creation failed", exc_info=True, extra={
            "subject": subject
        })
        raise VisualisationError(f"Cannot create plot: {e}") from e
    finally:
        plt.close(fig)


def plot_score_comparison(
    df: pd.DataFrame,
    subjects: list,
    output_path: Path
) -> plt.Figure:
    """
    Create box plots comparing scores across subjects.

    Args:
        df: Student DataFrame.
        subjects: List of subject columns (e.g., ['math', 'english']).
        output_path: Path where PNG will be saved.

    Returns:
        plt.Figure: The created figure object.

    Raises:
        VisualisationError: If plot creation fails.
    """
    try:
        logger.info("Creating comparison plot", extra={
            "subjects": subjects,
            "output_file": str(output_path)
        })

        fig, ax = plt.subplots(figsize=(12, 6))

        # Prepare data for boxplot
        box_data = [df[s].values for s in subjects]
        bp = ax.boxplot(
            box_data,
            labels=[s.title() for s in subjects],
            patch_artist=True
        )

        # Color the boxes
        for patch in bp["boxes"]:
            patch.set_facecolor("#58a6ff")

        ax.set_ylabel("Score")
        ax.set_title("Score Comparison Across Subjects")
        ax.grid(axis="y", alpha=0.3)

        fig.tight_layout()
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
        logger.info("Comparison plot saved", extra={
            "path": str(output_path),
            "num_subjects": len(subjects)
        })

        return fig

    except Exception as e:
        logger.error("Comparison plot creation failed", exc_info=True)
        raise VisualisationError(f"Cannot create comparison plot: {e}") from e
    finally:
        plt.close(fig)


def plot_gpa_vs_math(
    df: pd.DataFrame,
    output_path: Path
) -> plt.Figure:
    """
    Create scatter plot of GPA vs Math score with color mapping.

    Args:
        df: Student DataFrame.
        output_path: Path where PNG will be saved.

    Returns:
        plt.Figure: The created figure object.

    Raises:
        VisualisationError: If plot creation fails.
    """
    try:
        logger.info("Creating scatter plot", extra={
            "x_axis": "math",
            "y_axis": "gpa",
            "output_file": str(output_path)
        })

        fig, ax = plt.subplots(figsize=(10, 7))

        # Scatter plot with color mapping
        scatter = ax.scatter(
            df["math"],
            df["gpa"],
            alpha=0.6,
            s=100,
            c=df["math"],           # Color by math score
            cmap="viridis",         # Color palette
            edgecolors="black",
            linewidth=0.5
        )

        ax.set_xlabel("Math Score")
        ax.set_ylabel("GPA")
        ax.set_title("Relationship: Math Score vs GPA")
        ax.grid(alpha=0.3)

        # Add colorbar
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label("Math Score")

        fig.tight_layout()
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
        logger.info("Scatter plot saved", extra={
            "path": str(output_path),
            "correlation": float(df[["math", "gpa"]].corr().iloc[0, 1])
        })

        return fig

    except Exception as e:
        logger.error("Scatter plot creation failed", exc_info=True)
        raise VisualisationError(f"Cannot create scatter plot: {e}") from e
    finally:
        plt.close(fig)