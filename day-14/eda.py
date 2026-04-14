import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import logging


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


logger = logging.getLogger(__name__)


# Load and Summarize the Data
def load_and_summarize_data(file_path):
    logger.info(f"Loading data from {file_path}")
    df = pd.read_csv(file_path)
    logger.info(f"Data loaded successfully. Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    
    # Data types
    logger.info(f"Data types:\n{df.dtypes}")

    # print NULL Counts
    logger.info("Null value counts:")
    null_counts = df.isnull().sum()
    logger.info(f"\n{null_counts}")
    print(null_counts[null_counts > 0] if null_counts.sum() > 0 else "No missing values!")


    logger.info("Data summary:")
    logger.info(f"\n{df.describe()}")
    
    return df


# Plot Distributions
def plot_distributions(df, numeric_only=True):
    """Plot histograms for all numeric features.
    
    Histograms show you the shape of each feature's distribution.
    Is it roughly normal (bell-shaped)? Skewed? Multimodal?
    This reveals whether features need transformation (e.g., log scaling)."""
    logger.info("Plotting Distributions....")


    # Select numeric columns
    if numeric_only:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
    else:
        numeric_cols = df.columns  # Include all columns


    # create subplots
    num_cols = 3
    num_rows = (len(numeric_cols) + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 3*num_rows))
    axes = axes.flatten() # flatten the axes array for easy indexing


    # Plot each column
    for idx, col in enumerate(numeric_cols):
        axes[idx].hist(df[col].dropna(), bins=30, color='skyblue', edgecolor='black')
        axes[idx].set_title(f'Distribution of {col}')
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Frequency')
        axes[idx].grid(True, alpha=0.3)

    # Remove empty subplots
    for idx in range(len(numeric_cols), len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.show()
    logger.info(f"Distributions plotted successfully for {len(numeric_cols)} numeric features.")


# Plot Correlation Heatmap
def plot_correlation_heatmap(df):
    """Plot a heatmap of feature correlations.
    
    Correlations show which features move together. High correlation between
    two features (say 0.95) suggests they're redundant—you might only need one.
    We only compute correlations for numeric features."""
    logger.info("Plotting Correlation Heatmap....")
    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.shape[1] == 0:
        logger.warning("No numeric features found for correlation heatmap.")
        return
    
    # Compute correlation matrix
    corr_matrix = numeric_df.corr()

    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, square=True, cbar_kws={'label': 'Correlation'})
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.show()
    logger.info("Correlation heatmap plotted successfully.")


# Check Class Imbalance
def check_class_imbalance(df, target_col):
    """For classification datasets, check the distribution of the target variable.
    
    A balanced dataset has roughly equal examples per class (e.g., 50% class A, 50% class B).
    An imbalanced dataset might have 95% class A and 5% class B.
    If the minority class is <20%, the problem is severe and requires special handling."""

    if target_col not in df.columns:
        logger.error(f"Target column '{target_col}' not found in the dataset.")
        return
    
    logger.info(f"Checking class imbalance for target column: '{target_col}'")

    # Get class counts
    class_counts = df[target_col].value_counts().sort_values(ascending=False)
    total_samples = len(df)

    print(f"Class distribution for '{target_col.upper()}':")
    for class_name, count in class_counts.items():
        percentage = (count / total_samples) * 100
        print(f"Class '{class_name}': {count} samples ({percentage:.2f}%)")

    # Check for imbalance
    min_percentage = (class_counts.min() / total_samples) * 100
    if min_percentage < 20:
        logger.warning(f"Class imbalance detected! Minority class is only {min_percentage:.2f}% of the data.")
        print("\n WARNING: Class imbalance detected! Consider techniques like resampling, class weights, or stratification.")
    else:
        logger.info("No significant class imbalance detected.")

    
    # Plot class distribution
    plt.figure(figsize=(8, 5))
    class_counts.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title(f"Class Distribution for '{target_col.upper()}'")
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()

# Run Full EDA

def run_full_eda(file_path, target_col=None):
    """    Orchestrate a complete EDA pipeline.
    
    This is the main function you call to explore a new dataset.
    It calls load_and_summarize, plots distributions, plots correlations,
    and (if target_column is provided) checks class balance."""

    logger.info("Starting full EDA pipeline...")

    # step 1: Load and summarize data
    df = load_and_summarize_data(file_path)


    # step 2: Plot distributions
    plot_distributions(df, numeric_only=True)


    # step 3: Plot correlation heatmap
    plot_correlation_heatmap(df)


    # step 4: Check class imbalance (if target column provided)
    if target_col:
        check_class_imbalance(df, target_col)

    logger.info("EDA pipeline completed successfully.")

    return df