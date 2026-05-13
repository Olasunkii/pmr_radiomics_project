import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score


class ClusterStability:
    """
    Evaluate clustering stability across multiple runs.
    """

    def __init__(
        self,
        n_runs=20,
        n_clusters=2,
        random_state=42,
        logger=None
    ):

        self.n_runs = n_runs
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.logger = logger

    def evaluate(
        self,
        X
    ):
        """
        Run clustering multiple times and compare stability.
        """

        cluster_runs = []

        for seed in range(self.n_runs):

            model = KMeans(
                n_clusters=self.n_clusters,
                random_state=seed,
                n_init=10
            )

            labels = model.fit_predict(X)

            cluster_runs.append(labels)

        scores = []

        for i in range(len(cluster_runs)):
            for j in range(i + 1, len(cluster_runs)):

                ari = adjusted_rand_score(
                    cluster_runs[i],
                    cluster_runs[j]
                )

                scores.append(ari)

        stability_df = pd.DataFrame({
            "ARI": scores
        })

        return stability_df

    def summarize(
        self,
        stability_df
    ):
        """
        Compute summary statistics.
        """

        summary = {
            "mean_ari": stability_df["ARI"].mean(),
            "std_ari": stability_df["ARI"].std(),
            "min_ari": stability_df["ARI"].min(),
            "max_ari": stability_df["ARI"].max()
        }

        return pd.DataFrame([summary])