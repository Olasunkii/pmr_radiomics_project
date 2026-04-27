import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class EDAAnalyzer:
    """
    EDAAnalyzer performs exploratory data analysis and generates
    summary statistics and visualizations.
    """

    def __init__(self, output_path: Path, logger):
        self.output_path = output_path
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.logger = logger

    def basic_info(self, df: pd.DataFrame):
        """
        Return basic dataset information.
        """
        shape = df.shape
        self.logger.info(f"Dataset shape: {shape}")
        return shape

    def target_distribution(self, df: pd.DataFrame):
        """
        Compute and plot target distribution.
        """
        counts = df["target"].value_counts()
        percentages = df["target"].value_counts(normalize=True)

        self.logger.info(f"Target counts:\n{counts}")
        self.logger.info(f"Target percentages:\n{percentages}")

        # Plot
        plt.figure()
        counts.plot(kind="bar")
        plt.title("Target Distribution")
        plt.xlabel("Class")
        plt.ylabel("Count")

        file_path = self.output_path / "target_distribution.png"
        plt.savefig(file_path)
        plt.show()
        plt.close()

        return counts, percentages

    def summary_statistics(self, df: pd.DataFrame):
        """
        Compute summary statistics (transpose).
        """
        summary = df.describe().T
        self.logger.info("Summary statistics computed.")
        return summary

    def skewness(self, df: pd.DataFrame):
        """
        Compute feature skewness.
        """
        skew = df.skew().sort_values(ascending=False)
        self.logger.info("Skewness computed.")
        return skew

    def correlation_matrix(self, df: pd.DataFrame):
        """
        Compute and plot correlation matrix.
        """
        corr = df.corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, cmap="coolwarm", center=0)

        file_path = self.output_path / "correlation_matrix.png"
        plt.savefig(file_path)
        plt.show()
        plt.close()

        return corr

    def high_correlation(self, corr_matrix: pd.DataFrame, threshold=0.9):
        """
        Identify highly correlated feature pairs.
        """
        high_corr = (
            corr_matrix.where((corr_matrix > threshold) & (corr_matrix < 1.0))
            .stack()
            .sort_values(ascending=False)
        )

        self.logger.info("High correlation pairs identified.")
        return high_corr