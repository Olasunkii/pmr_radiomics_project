import shap
import matplotlib.pyplot as plt


class ShapAnalyzer:
    """
    SHAP analysis module for interpretability.
    Saves multiple types of SHAP visualizations.
    """

    def __init__(self, output_path):
        self.output_path = output_path
        self.output_path.mkdir(parents=True, exist_ok=True)

    def compute_shap(self, model, X):
        """
        Compute SHAP values.
        """
        explainer = shap.Explainer(model, X)
        shap_values = explainer(X)
        return shap_values

    # ======================
    # GLOBAL PLOTS
    # ======================

    def save_summary_plot(self, shap_values, X, filename):
        """
        Beeswarm summary plot (good for small/medium features).
        """
        shap.summary_plot(shap_values, X, show=False)
        plt.savefig(self.output_path / filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filename}")

    def save_bar_plot(self, shap_values, filename):
        """
        Global feature importance bar plot.
        """
        shap.plots.bar(shap_values, show=False)
        plt.savefig(self.output_path / filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filename}")

    # ======================
    # ADVANCED PLOTS
    # ======================

    def save_violin_plot(self, shap_values, X, filename):
        """
        Violin plot (better than beeswarm for dense features).
        """
        shap.summary_plot(shap_values, X, plot_type="violin", show=False)
        plt.savefig(self.output_path / filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filename}")

    def save_heatmap_plot(self, shap_values, filename):
        """
        Heatmap (great for many features).
        """
        shap.plots.heatmap(shap_values, show=False)
        plt.savefig(self.output_path / filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filename}")

    def save_decision_plot(self, shap_values, filename):
        """
        Decision plot (model behavior across samples).
        """
        shap.plots.decision(shap_values, show=False)
        plt.savefig(self.output_path / filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filename}")
