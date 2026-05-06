import matplotlib.pyplot as plt
from pathlib import Path


class Visualizer:
    """
    Visualizer handles plotting and saving all result figures.
    """

    def __init__(self, output_path):
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)

    def plot_model_performance(self, results, filename="model_comparison_auc.png"):
        """
        Plot AUC comparison across models.
        """
        model_names = list(results.keys())
        auc_scores = [results[m]["AUC"] for m in model_names]

        plt.figure()
        plt.bar(model_names, auc_scores)

        plt.title("Model AUC Comparison")
        plt.ylabel("AUC")
        plt.xlabel("Models")

        plt.savefig(self.output_path / filename, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved: {filename}")

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