"""
Main script demonstrating the EDA pipeline.

This script shows how to use the eda module on the Iris dataset
(a classic classification dataset with flowers).
"""

import logging
from eda import run_full_eda

if __name__ == '__main__':
    print("=" * 70)
    print("EXPLORATORY DATA ANALYSIS PIPELINE")
    print("=" * 70)
    print()
    
    # For this example, we'll use the Iris dataset bundled with sklearn
    # In practice, you'd use your own CSV file
    print("Downloading and preparing sample data (Iris dataset)...")
    
    from sklearn.datasets import load_iris
    import pandas as pd
    
    iris = load_iris()
    iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
    iris_df['species'] = iris.target_names[iris.target]
    
    # Save to CSV
    filepath = 'iris_data.csv'
    iris_df.to_csv(filepath, index=False)
    print(f"Sample data saved to {filepath}")
    print()
    
    # Run full EDA pipeline
    # target_column='species' tells the pipeline to check class balance
    df = run_full_eda(filepath, target_col='species')
    
    print("\nEDA pipeline complete! Explore the plots above.")
    print("The dataframe is stored in the 'df' variable for further analysis.")