import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- Data Loading ---
def load_data(file_path: str) -> pd.DataFrame:
    """Load the CSV data into a DataFrame."""
    return pd.read_csv(file_path)

# --- Visualization Functions ---

def scatterplot_performance_vs_dimension(df: pd.DataFrame):
    """
    Scatterplot showing the relationship between ambient dimension (n)
    and subspace dimension (d) for different algorithm variations.
    Color-code by the algorithmic variation.
    """
    variations = {1: 'MeanSDF', 2: 'MedianSDF', 3: 'HuberRegression'}
    plt.figure(figsize=(8,6))
    for var, label in variations.items():
        subset = df[df['variation'] == var]
        plt.scatter(subset['n'], subset['d'], alpha=0.5, label=label)
    plt.xlabel('Ambient Dimension (n)')
    plt.ylabel('Subspace Dimension (d)')
    plt.title('Ambient vs. Subspace Dimensions by Algorithm Variation')
    plt.legend()
    plt.tight_layout()
    plt.savefig('scatter_n_vs_d.png')
    plt.show()

def boxplot_metrics_by_variation(df: pd.DataFrame):
    """
    Create boxplots for performance metrics (R² and MSE) across different algorithm variations.
    """
    metrics = ['r2_regression', 'r2_orthogonal', 'mse_regression', 'mse_orthogonal']
    variations = {1: 'MeanSDF', 2: 'MedianSDF', 3: 'HuberRegression'}
    
    for metric in metrics:
        plt.figure(figsize=(8,6))
        data = [df[df['variation'] == var][metric] for var in variations]
        plt.boxplot(data, labels=list(variations.values()))
        plt.ylabel(metric)
        plt.title(f'Boxplot of {metric} by Algorithm Variation')
        plt.tight_layout()
        plt.savefig(f'boxplot_{metric}.png')
        plt.show()

def scatterplot_runtime_vs_accuracy(df: pd.DataFrame):
    """
    Plot the relationship between computational runtime (timeMs)
    and accuracy metrics (R² Regression) to investigate any correlation.
    """
    plt.figure(figsize=(8,6))
    plt.scatter(df['timeMs'], df['r2_regression'], alpha=0.5, c='blue')
    plt.xlabel('Runtime (ms)')
    plt.ylabel('R² Regression')
    plt.title('Runtime vs. R² Regression Accuracy')
    plt.tight_layout()
    plt.savefig('scatter_runtime_vs_r2.png')
    plt.show()

def correlation_heatmap(df: pd.DataFrame):
    """
    Compute and visualize the correlation matrix for numeric columns.
    """
    # Select numeric columns relevant for performance and runtime analysis
    cols = ['n', 'd', 'noise', 'outlierRatio', 'r2_regression', 'r2_orthogonal', 
            'mse_regression', 'mse_orthogonal', 'timeMs']
    corr_matrix = df[cols].corr()
    
    plt.figure(figsize=(10,8))
    # Use imshow for a basic heatmap
    im = plt.imshow(corr_matrix, cmap='viridis', interpolation='none')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(cols)), cols, rotation=45, ha='right')
    plt.yticks(range(len(cols)), cols)
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png')
    plt.show()

def noise_outlier_robustness(df: pd.DataFrame):
    """
    Visualize the effect of noise and outlier ratios on a performance metric (e.g., R² Regression)
    using scatterplots. You can extend this to include regression lines or binning.
    """
    plt.figure(figsize=(8,6))
    scatter = plt.scatter(df['noise'], df['r2_regression'], c=df['outlierRatio'], cmap='plasma', alpha=0.5)
    plt.xlabel('Noise Level')
    plt.ylabel('R² Regression')
    plt.title('Effect of Noise and Outlier Ratio on R² Regression')
    plt.colorbar(scatter, label='Outlier Ratio')
    plt.tight_layout()
    plt.savefig('scatter_noise_vs_r2.png')
    plt.show()

# --- Additional Visualization Suggestions ---
def additional_visualizations(df: pd.DataFrame):
    """
    Suggestions for further visualizations:
    - Violin plots of performance metrics to understand distribution shapes per variation.
    - A multi-panel grid showing performance metrics vs. noise levels segmented by outlierRatio.
    - Parallel coordinate plots to compare multiple performance metrics simultaneously.
    - Trend plots over iterations if the dataset contains temporal progression.
    """
    print("Additional Visualization Suggestions:")
    print("1. Violin plots for performance metrics per algorithm variation.")
    print("2. Faceted scatter or line plots of performance metrics vs. noise, segmented by outlier ratios.")
    print("3. Parallel coordinates plots for a multi-dimensional comparison of performance metrics and parameters.")
    print("4. Time-series plots if iteration information is sequentially informative.")

# --- Main Function ---
def main():
    # Update the file path if necessary
    file_path = '../fbuild/ransac_evaluation_results.csv'
    df = load_data(file_path)
    
    # Generate the visualizations
    scatterplot_performance_vs_dimension(df)
    boxplot_metrics_by_variation(df)
    scatterplot_runtime_vs_accuracy(df)
    correlation_heatmap(df)
    noise_outlier_robustness(df)
    
    # Print additional visualization suggestions
    additional_visualizations(df)

if __name__ == "__main__":
    main()
