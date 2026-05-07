import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path


class Visualizer:
    """
    Visualizer handles plotting and saving all result figures.
    """

    def __init__(self, output_path):
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)

    def plot_model_performance(self, results_df):
        """
        Plot AUC performance across datasets and models.
        """

        plt.figure(figsize=(10, 6))

        for dataset in results_df["dataset"].unique():

            subset = results_df[results_df["dataset"] == dataset]

            plt.plot(
                subset["model"],
                subset["AUC"],
                marker='o',
                label=dataset
            )

        plt.title("Model Performance Comparison")
        plt.xlabel("Model")
        plt.ylabel("AUC")
        plt.ylim(0, 1)
        plt.legend()

        file_path = self.output_path / "model_comparison_auc.png"

        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()

    print("Saved: model_comparison_auc.png")

    def plot_stability(self, stability_series, filename="stability_distribution.png"):
        """
        Plot distribution of stability scores.
        """
        plt.figure()
        stability_series.hist(bins=20)

        plt.title("Feature Stability Distribution")
        plt.xlabel("Stability Score")
        plt.ylabel("Frequency")

        plt.savefig(self.output_path / filename, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved: {filename}")


    def plot_grouped_auc(self, results_df):
        """
        Publication-style grouped AUC comparison.
        """

        plt.figure(figsize=(10, 6))

        pivot_df = results_df.pivot(
            index="dataset",
            columns="model",
            values="AUC"
        )

        ax = pivot_df.plot(kind="bar")

        for container in ax.containers:
            ax.bar_label(container, fmt="%.3f")

        plt.title("AUC Comparison Across Feature Sets")
        plt.ylabel("AUC")
        plt.xlabel("Feature Set")
        plt.ylim(0, 1)

        plt.tight_layout()

        plt.savefig(
            self.output_path / "grouped_auc_comparison.png",
            dpi=300,
            bbox_inches='tight'
        )

        plt.close()

        print("Saved: grouped_auc_comparison.png")


    def plot_auc_heatmap(self, results_df):
        """
        Heatmap of AUC scores.
        """

        pivot_df = results_df.pivot(
            index="dataset",
            columns="model",
            values="AUC"
        )

        plt.figure(figsize=(8, 5))

        sns.heatmap(
            pivot_df,
            annot=True,
            fmt=".3f",
            cmap="viridis"
        )

        plt.title("AUC Heatmap")

        plt.tight_layout()

        plt.savefig(
            self.output_path / "auc_heatmap.png",
            dpi=300,
            bbox_inches='tight'
        )

        plt.close()

        print("Saved: auc_heatmap.png")


    def plot_feature_counts(self, all_count, stable_count, sm_count):
        """
        Plot feature count reduction.
        """

        labels = ["All", "Stable", "Stable+Moderate"]
        counts = [all_count, stable_count, sm_count]

        plt.figure(figsize=(7, 5))

        bars = plt.bar(labels, counts)

        for bar in bars:
            height = bar.get_height()

            plt.text(
                bar.get_x() + bar.get_width()/2,
                height + 0.5,
                str(height),
                ha='center'
            )

        plt.title("Feature Count Comparison")
        plt.ylabel("Number of Features")

        plt.tight_layout()

        plt.savefig(
            self.output_path / "feature_count_comparison.png",
            dpi=300,
            bbox_inches='tight'
        )

        plt.close()

        print("Saved: feature_count_comparison.png")


    def plot_confusion_matrix(self, cm, title, filename):
        """
        Plot confusion matrix.
        """

        plt.figure(figsize=(5, 4))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues"
        )

        plt.title(title)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        plt.tight_layout()

        plt.savefig(
            self.output_path / filename,
            dpi=300,
            bbox_inches='tight'
        )

        plt.close()

        print(f"Saved: {filename}")