import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests


class ClusterStatistics:
    """
    Statistical comparison between clusters.
    """

    def __init__(self, logger=None):

        self.logger = logger

    def compare_clusters(
        self,
        X,
        cluster_labels
    ):
        """
        Perform feature-wise statistical testing.
        """

        df = X.copy()

        df["Cluster"] = cluster_labels

        cluster0 = df[df["Cluster"] == 0]
        cluster1 = df[df["Cluster"] == 1]

        results = []

        features = X.columns

        for feature in features:

            stat, pvalue = ttest_ind(
                cluster0[feature],
                cluster1[feature],
                equal_var=False
            )

            results.append({
                "feature": feature,
                "cluster0_mean": cluster0[feature].mean(),
                "cluster1_mean": cluster1[feature].mean(),
                "p_value": pvalue
            })

        results_df = pd.DataFrame(results)

        results_df = results_df.sort_values(
            "p_value"
        )

        return results_df

    def significant_features(
        self,
        results_df,
        alpha=0.05
    ):
        """
        Extract statistically significant features.
        """

        significant = results_df[
            results_df["p_value"] < alpha
        ]

        return significant
    
    def apply_fdr_correction(
        self,
        results_df,
        alpha=0.05
        ):
        """
        Apply Benjamini-Hochberg FDR correction.
        """

        corrected = multipletests(
            results_df["p_value"],
            alpha=alpha,
            method="fdr_bh"
        )

        results_df["fdr_corrected_p"] = corrected[1]

        results_df["significant_fdr"] = corrected[0]

        return results_df
    

    def compute_effect_sizes(
        self,
        X,
        cluster_labels,
        results_df
        ):
        """
        Compute Cohen's d effect size for each feature.
        """

        df = X.copy()

        df["Cluster"] = cluster_labels

        cluster0 = df[df["Cluster"] == 0]
        cluster1 = df[df["Cluster"] == 1]

        effect_sizes = []

        for feature in X.columns:

            mean0 = cluster0[feature].mean()
            mean1 = cluster1[feature].mean()

            std0 = cluster0[feature].std()
            std1 = cluster1[feature].std()

            n0 = len(cluster0)
            n1 = len(cluster1)

            pooled_std = np.sqrt(
                (
                    ((n0 - 1) * std0**2) +
                    ((n1 - 1) * std1**2)
                ) / (n0 + n1 - 2)
            )

            if pooled_std == 0:
                d = 0
            else:
                d = (mean1 - mean0) / pooled_std

            effect_sizes.append(d)

        results_df["cohens_d"] = effect_sizes

        return results_df #