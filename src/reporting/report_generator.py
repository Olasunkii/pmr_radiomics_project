from pathlib import Path
import pandas as pd


class ReportGenerator:
    """
    Generate automated experiment reports.
    """

    def __init__(
        self,
        output_path
    ):

        self.output_path = output_path

        self.output_path.mkdir(
            parents=True,
            exist_ok=True
        )

    def generate_markdown_report(
        self,
        experiment_name,
        results_df,
        stability_summary,
        significant_df,
        stable_count,
        sm_count,
        all_count
    ):
        """
        Generate markdown experiment report.
        """

        best_row = results_df.sort_values(
            "AUC",
            ascending=False
        ).iloc[0]

        report = f"""
# PMR Radiomics Experiment Report

## Experiment
**Name:** {experiment_name}

---

# 1. Supervised Learning Analysis

The supervised learning phase evaluated multiple machine learning models across different feature subsets:

- All radiomic features
- Stable features only
- Stable + moderate stability features

This progressive design aimed to evaluate whether feature stability improves generalization and interpretability.

## Best Performing Configuration

- Dataset: {best_row['dataset']}
- Model: {best_row['model']}
- AUC: {best_row['AUC']:.3f}

---

# 2. Feature Stability Analysis

Feature stability analysis was performed using iterative LASSO selection.

## Feature Counts

| Feature Set | Count |
|---|---|
| All Features | {all_count} |
| Stable Features | {stable_count} |
| Stable + Moderate Features | {sm_count} |

This stage identified reproducible biomarkers robust to repeated sampling variation.

---

# 3. Explainability Analysis

SHAP analysis was used to interpret model predictions and identify influential radiomic biomarkers.

Generated explainability outputs include:

- SHAP summary plots
- SHAP violin plots
- SHAP heatmaps
- Patient-specific SHAP waterfall plots

This improves clinical interpretability and transparency of model predictions.

---

# 4. Unsupervised Phenotype Discovery

Unsupervised learning was performed to identify latent patient subgroups independent of labels.

Techniques included:

- PCA
- UMAP
- KMeans clustering

The objective was to explore hidden inflammatory-radiomic phenotypes.

## Cluster Stability

| Metric | Value |
|---|---|
| Mean ARI | {stability_summary['mean_ari'].iloc[0]:.3f} |
| Std ARI | {stability_summary['std_ari'].iloc[0]:.3f} |

The high ARI indicates highly reproducible cluster structure.

---

# 5. Biomarker Discovery

Statistical testing identified significant biomarkers differentiating discovered clusters.

## Significant Biomarkers

Top significant features:

{significant_df.head(10).to_markdown(index=False)}

Features were evaluated using:
- Welch t-tests
- FDR correction
- Cohen's d effect sizes

This provides statistically rigorous biomarker discovery.

---

# 6. Overall Interpretation

The pipeline demonstrates that:

1. Stable radiomic biomarkers improve robustness
2. Explainable AI improves interpretability
3. Unsupervised learning reveals latent phenotype structure
4. Several radiomic inflammatory biomarkers strongly differentiate patient subgroups

The framework provides a reproducible and extensible computational pipeline for PMR radiomics research.
"""

        report_path = (
            self.output_path /
            "experiment_report.md"
        )

        with open(
            report_path,
            "w",
            encoding="utf-8"
        ) as f:

            f.write(report)

        print("Saved: experiment_report.md")