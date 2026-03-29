"""Seaborn-based statistical visualisation functions."""
import logging
from typing import Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from src.config import settings
from src.exceptions import VisualisationError

logger = logging.getLogger(__name__)


def plot_correlation_heatmap(
    df: pd.DataFrame,
    output_path: Path
) -> plt.Figure:
    """
    Create correlation heatmap for numerical columns.

    Shows how each subject correlates with others (e.g., if students
    who do well in math also do well in science).

    Args:
        df: Student DataFrame.
        output_path: Path where PNG will be saved.

    Returns:
        plt.Figure: The created figure object.

    Raises:
        VisualisationError: If plot creation fails.
    """
    try:
        logger.info("Creating correlation heatmap", extra={
            "output_file": str(output_path)
        })

        numeric_cols = ["math", "english", "science", "history", "gpa"]
        corr_matrix = df[numeric_cols].corr()

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            corr_matrix,
            annot=True,             # Show correlation values
            fmt=".2f",              # 2 decimal places
            cmap="coolwarm",        # Blue (negative) to red (positive)
            center=0,               # 0 is white
            square=True,
            linewidths=1,
            cbar_kws={"shrink": 0.8},
            ax=ax
        )

        ax.set_title("Correlation Matrix: All Subjects and GPA")
        fig.tight_layout()
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
        logger.info("Heatmap saved", extra={
            "path": str(output_path),
            "num_vars": len(numeric_cols)
        })

        return fig

    except Exception as e:
        logger.error("Heatmap creation failed", exc_info=True)
        raise VisualisationError(f"Cannot create heatmap: {e}") from e
    finally:
        plt.close(fig)


def plot_subject_distributions(
    df: pd.DataFrame,
    output_path: Path
) -> plt.Figure:
    """
    Create KDE (kernel density estimation) plots for all subjects.

    KDE plots are like smooth histograms. Good for seeing the overall
    shape of the distribution.

    Args:
        df: Student DataFrame.
        output_path: Path where PNG will be saved.

    Returns:
        plt.Figure: The created figure object.

    Raises:
        VisualisationError: If plot creation fails.
    """
    try:
        logger.info("Creating KDE plots", extra={
            "output_file": str(output_path)
        })

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        subjects = ["math", "english", "science", "history"]
        colors = ["#58a6ff", "#79c0ff", "#3fb950", "#ffa657"]

        for idx, (ax, subject, color) in enumerate(
            zip(axes.flat, subjects, colors)
        ):
            sns.kdeplot(
                data=df,
                x=subject,
                fill=True,
                color=color,
                ax=ax,
                alpha=0.7
            )
            ax.set_title(f"{subject.title()} Score Distribution")
            ax.set_xlabel(f"{subject.title()} Score")
            ax.set_ylabel("Density")

        fig.suptitle("Score Distributions (KDE)", fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
        logger.info("KDE plots saved", extra={
            "path": str(output_path),
            "num_subjects": len(subjects)
        })

        return fig

    except Exception as e:
        logger.error("KDE plot creation failed", exc_info=True)
        raise VisualisationError(f"Cannot create KDE plots: {e}") from e
    finally:
        plt.close(fig)


def plot_violin_comparison(
    df: pd.DataFrame,
    output_path: Path
) -> plt.Figure:
    """
    Create violin plots for subject score distributions.

    Violin plots show the full distribution and quartiles. Wide = more
    students with that score.

    Args:
        df: Student DataFrame.
        output_path: Path where PNG will be saved.

    Returns:
        plt.Figure: The created figure object.

    Raises:
        VisualisationError: If plot creation fails.
    """
    try:
        logger.info("Creating violin plot", extra={
            "output_file": str(output_path)
        })

        # Reshape data for Seaborn (long format)
        subjects = ["math", "english", "science", "history"]
        data_long = []
        for subject in subjects:
            temp = df[[subject]].copy()
            temp.columns = ["score"]
            temp["subject"] = subject.title()
            data_long.append(temp)

        df_long = pd.concat(data_long, ignore_index=True)

        fig, ax = plt.subplots(figsize=(12, 7))
        sns.violinplot(
            data=df_long,
            x="subject",
            y="score",
            ax=ax,
            palette="Set2"
        )

        ax.set_title("Score Distributions by Subject (Violin Plot)")
        ax.set_xlabel("Subject")
        ax.set_ylabel("Score")
        fig.tight_layout()
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
        logger.info("Violin plot saved", extra={
            "path": str(output_path),
            "num_subjects": len(subjects)
        })

        return fig

    except Exception as e:
        logger.error("Violin plot creation failed", exc_info=True)
        raise VisualisationError(f"Cannot create violin plot: {e}") from e
    finally:
        plt.close(fig)