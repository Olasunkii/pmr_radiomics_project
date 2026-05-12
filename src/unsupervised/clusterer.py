import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


class Clusterer:
    """
    KMeans clustering module.
    """

    def __init__(self, random_state=42, logger=None):

        self.random_state = random_state
        self.logger = logger

    def run_kmeans(self, X, n_clusters=2):

        model = KMeans(
            n_clusters=n_clusters,
            random_state=self.random_state,
            n_init=10
        )

        labels = model.fit_predict(X)

        score = silhouette_score(X, labels)

        if self.logger:
            self.logger.info(
                f"KMeans ({n_clusters} clusters) silhouette score: {score:.3f}"
            )

        return model, labels, score
    
    def evaluate_k_range(self, X, k_range=range(2, 7)):
        """
        Evaluate clustering across multiple K values.
        """

        results = []

        for k in k_range:

            model = KMeans(
                n_clusters=k,
                random_state=self.random_state,
                n_init=10
            )

            labels = model.fit_predict(X)

            inertia = model.inertia_

            silhouette = silhouette_score(X, labels)

            results.append({
                "k": k,
                "inertia": inertia,
                "silhouette": silhouette
            })

            if self.logger:
                self.logger.info(
                    f"K={k} | inertia={inertia:.3f} | silhouette={silhouette:.3f}"
                )
        return pd.DataFrame(results)
    
    def compare_clusters_to_labels(
        self,
        cluster_labels,
        true_labels
        ):
        """
        Compare clusters against true PMR labels.
        """

        comparison_df = pd.DataFrame({
            "cluster": cluster_labels,
            "true_label": true_labels
        })

        summary = pd.crosstab(
            comparison_df["cluster"],
            comparison_df["true_label"]
        )

        return comparison_df, summary