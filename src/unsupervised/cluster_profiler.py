import pandas as pd
import matplotlib.pyplot as plt


class ClusterProfiler:
    """
    Analyze and profile unsupervised clusters.
    """

    def __init__(self, output_path, logger=None):

        self.output_path = output_path
        self.logger = logger

        self.output_path.mkdir(
            parents=True,
            exist_ok=True
        )

    def compute_cluster_means(
        self,
        X,
        cluster_labels
    ):
        """
        Compute mean feature values per cluster.
        """

        df = X.copy()

        df["Cluster"] = cluster_labels

        cluster_means = (
            df.groupby("Cluster")
            .mean()
            .T
        )

        cluster_means.to_csv(
            self.output_path / "cluster_feature_means.csv"
        )

        print("Saved: cluster_feature_means.csv")

        return cluster_means

    def compute_cluster_std(
        self,
        X,
        cluster_labels
    ):
        """
        Compute standard deviation per cluster.
        """

        df = X.copy()

        df["Cluster"] = cluster_labels

        cluster_std = (
            df.groupby("Cluster")
            .std()
            .T
        )

        cluster_std.to_csv(
            self.output_path / "cluster_feature_std.csv"
        )

        print("Saved: cluster_feature_std.csv")

        return cluster_std

    def plot_cluster_heatmap(
        self,
        cluster_means,
        top_n=25,
        filename="cluster_heatmap.png"
    ):
        """
        Plot cluster heatmap using top varying features.
        """

        # Select most variable features
        variability = (
            cluster_means.var(axis=1)
            .sort_values(ascending=False)
        )

        top_features = variability.head(top_n).index

        heatmap_data = cluster_means.loc[top_features]

        plt.figure(figsize=(8, 10))

        plt.imshow(
            heatmap_data,
            aspect="auto"
        )

        plt.colorbar(label="Mean Scaled Value")

        plt.yticks(
            range(len(top_features)),
            top_features
        )

        plt.xticks(
            range(len(cluster_means.columns)),
            [f"Cluster {c}" for c in cluster_means.columns]
        )

        plt.title("Cluster Feature Heatmap")

        plt.tight_layout()

        plt.savefig(
            self.output_path / filename,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print(f"Saved: {filename}")