import logging
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ============================================================================
# DATA GENERATION
# ============================================================================


def generate_exam_scores() -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic exam scores for study vs no-study groups."""
    np.random.seed(42)

    # Students who studied: higher mean (78), lower variance
    study_group = np.random.normal(loc=78, scale=8, size=50)

    # Students who didn't study: lower mean (68), higher variance
    no_study_group = np.random.normal(loc=68, scale=10, size=50)

    logger.info(f"Generated {len(study_group)} study group scores")
    logger.info(f"Generated {len(no_study_group)} no-study group scores")

    return study_group, no_study_group


# ============================================================================
# DESCRIPTIVE STATISTICS
# ============================================================================


def compute_stats(data: np.ndarray) -> Dict[str, float]:
    """Compute 7-point summary statistics."""
    logger.debug(f"Computing stats for {len(data)} values")

    return {
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
        "std": float(np.std(data, ddof=1)),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "q25": float(np.percentile(data, 25)),
        "q75": float(np.percentile(data, 75)),
    }


def log_stats(name: str, stats_Dict: Dict[str, float]) -> None:
    """Log statistics nicely."""
    logger.info(f"{name} statistics:")
    for key, value in stats_Dict.items():
        logger.info(f"  {key:8s}: {value:7.2f}")


# ============================================================================
# HYPOTHESIS TESTING: t-test
# ============================================================================


def run_t_test(group1: np.ndarray, group2: np.ndarray) -> Tuple[float, float]:
    """Run independent samples t-test."""
    t_stat, p_value = stats.ttest_ind(group1, group2)

    logger.info(f"t-test results:")
    logger.info(f"  t-statistic: {t_stat:.4f}")
    logger.info(f"  p-value:     {p_value:.6f}")

    if p_value < 0.05:
        logger.info("  ✓ Statistically significant difference (p < 0.05)")
    else:
        logger.info("  ✗ No statistically significant difference (p >= 0.05)")

    return t_stat, p_value


# ============================================================================
# CONFIDENCE INTERVALS
# ============================================================================


def compute_ci(data: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
    """Compute confidence interval for the mean."""
    mean = np.mean(data)
    sem = stats.sem(data)
    df = len(data) - 1

    alpha = 1 - confidence
    t_crit = stats.t.ppf(1 - alpha / 2, df)
    margin = t_crit * sem

    logger.debug(f"CI: mean={mean:.2f}, sem={sem:.4f}, margin={margin:.2f}")

    return (mean - margin, mean + margin)


# ============================================================================
# VISUALIZATION
# ============================================================================


def plot_score_distributions(study: np.ndarray, no_study: np.ndarray) -> None:
    """Plot overlaid histograms of both groups."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Histograms
    ax.hist(
        study, bins=15, alpha=0.6, label="Study Group", color="green", edgecolor="black"
    )
    ax.hist(
        no_study,
        bins=15,
        alpha=0.6,
        label="No-Study Group",
        color="red",
        edgecolor="black",
    )

    ax.set_xlabel("Exam Score")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Exam Scores by Study Status")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("score_distribution.png", dpi=150)
    logger.info("Saved score distribution plot")
    plt.close()


def plot_ci_comparison(
    study: np.ndarray,
    no_study: np.ndarray,
    ci_study: Tuple[float, float],
    ci_no_study: Tuple[float, float],
) -> None:
    """Plot confidence intervals for both groups."""
    fig, ax = plt.subplots(figsize=(10, 6))

    groups = ["Study Group", "No-Study Group"]
    means = [np.mean(study), np.mean(no_study)]

    errors = [
        [means[0] - ci_study[0], means[1] - ci_no_study[0]],
        [ci_study[1] - means[0], ci_no_study[1] - means[1]],
    ]

    ax.errorbar(
        groups,
        means,
        yerr=errors,
        fmt="o",
        markersize=12,
        capsize=12,
        capthick=2,
        linewidth=2,
        color="blue",
    )

    ax.set_ylabel("Mean Score")
    ax.set_title("95% Confidence Intervals for Mean Exam Scores")
    ax.grid(axis="y", alpha=0.3)

    # Add value labels
    for i, (group, mean, ci) in enumerate(zip(groups, means, [ci_study, ci_no_study])):
        ax.text(
            i,
            mean + 2,
            f"{mean:.1f}\n[{ci[0]:.1f}, {ci[1]:.1f}]",
            ha="center",
            fontsize=10,
        )

    plt.tight_layout()
    plt.savefig("ci_comparison.png", dpi=150)
    logger.info("Saved confidence interval comparison plot")
    plt.close()


# ============================================================================
# MAIN ANALYSIS
# ============================================================================


def main() -> None:
    """Run complete statistical analysis."""
    logger.info("Starting exam score analysis...")

    # Generate data
    study_group, no_study_group = generate_exam_scores()

    # Descriptive statistics
    logger.info("")
    logger.info("=" * 60)
    logger.info("DESCRIPTIVE STATISTICS")
    logger.info("=" * 60)

    study_stats = compute_stats(study_group)
    no_study_stats = compute_stats(no_study_group)

    log_stats("Study Group", study_stats)
    logger.info("")
    log_stats("No-Study Group", no_study_stats)

    # Hypothesis test
    logger.info("")
    logger.info("=" * 60)
    logger.info("HYPOTHESIS TESTING (t-test)")
    logger.info("=" * 60)

    t_stat, p_value = run_t_test(study_group, no_study_group)

    # Confidence intervals
    logger.info("")
    logger.info("=" * 60)
    logger.info("CONFIDENCE INTERVALS (95%)")
    logger.info("=" * 60)

    ci_study = compute_ci(study_group)
    ci_no_study = compute_ci(no_study_group)

    logger.info(f"Study Group CI:     [{ci_study[0]:.2f}, {ci_study[1]:.2f}]")
    logger.info(f"No-Study Group CI:  [{ci_no_study[0]:.2f}, {ci_no_study[1]:.2f}]")

    # Visualization
    logger.info("")
    logger.info("=" * 60)
    logger.info("CREATING VISUALIZATIONS")
    logger.info("=" * 60)

    plot_score_distributions(study_group, no_study_group)
    plot_ci_comparison(study_group, no_study_group, ci_study, ci_no_study)

    logger.info("")
    logger.info("=" * 60)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
