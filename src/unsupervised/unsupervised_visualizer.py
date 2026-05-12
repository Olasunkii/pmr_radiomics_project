import matplotlib.pyplot as plt
import seaborn as sns


class UnsupervisedVisualizer:
    """
    Visualization utilities for unsupervised learning.
    """

    def __init__(self, output_path):

        self.output_path = output_path

    def plot_pca_clusters(
        self,
        pca_df,
        labels,
        filename="pca_clusters.png"
    ):
        """
        Plot PCA clusters.
        """

        plt.figure(figsize=(8, 6))

        scatter = plt.scatter(
            pca_df["PC1"],
            pca_df["PC2"],
            c=labels
        )

        plt.xlabel("PC1")
        plt.ylabel("PC2")

        plt.title("PCA Cluster Projection")

        plt.colorbar(scatter)

        plt.savefig(
            self.output_path / filename,
            dpi=300,
            bbox_inches='tight'
        )

        plt.close()

        print(f"Saved: {filename}")

    def plot_elbow_curve(
        self,
        results_df,
        filename="elbow_curve.png"
        ):
        """
        Plot elbow curve.
        """

        plt.figure(figsize=(8, 6))

        plt.plot(
            results_df["k"],
            results_df["inertia"],
            marker='o'
        )

        for i, row in results_df.iterrows():

            plt.text(
                row["k"],
                row["inertia"],
                f"{row['inertia']:.1f}"
            )

        plt.xlabel("Number of Clusters (K)")
        plt.ylabel("Inertia")

        plt.title("Elbow Curve")

        plt.savefig(
            self.output_path / filename,
            dpi=300,
            bbox_inches='tight'
        )

        plt.close()

        print(f"Saved: {filename}")

    def plot_silhouette_scores(
        self,
        results_df,
        filename="silhouette_scores.png"
        ):
        """
        Plot silhouette scores.
        """

        plt.figure(figsize=(8, 6))

        plt.plot(
            results_df["k"],
            results_df["silhouette"],
            marker='o'
        )

        for i, row in results_df.iterrows():

            plt.text(
                row["k"],
                row["silhouette"],
                f"{row['silhouette']:.3f}"
            )

        plt.xlabel("Number of Clusters (K)")
        plt.ylabel("Silhouette Score")

        plt.title("Silhouette Analysis")

        plt.savefig(
            self.output_path / filename,
            dpi=300,
            bbox_inches='tight'
        )

        plt.close()

        print(f"Saved: {filename}")

    def plot_cluster_label_heatmap(
        self,
        summary_df,
        filename="cluster_label_heatmap.png"
        ):
        """
        Plot cluster vs true label heatmap.
        """

        plt.figure(figsize=(6, 5))

        sns.heatmap(
            summary_df,
            annot=True,
            fmt='d'
        )

        plt.title("Cluster vs True Label")

        plt.savefig(
            self.output_path / filename,
            dpi=300,
            bbox_inches='tight'
        )

        plt.close()

        print(f"Saved: {filename}")

    def plot_umap_clusters(
        self,
        umap_df,
        labels,
        filename="umap_clusters.png"
         ):
        """
        Plot UMAP clusters.
        """

        plt.figure(figsize=(8, 6))

        scatter = plt.scatter(
            umap_df["UMAP1"],
            umap_df["UMAP2"],
            c=labels
        )

        plt.xlabel("UMAP1")
        plt.ylabel("UMAP2")

        plt.title("UMAP Cluster Projection")

        plt.colorbar(scatter)

        plt.savefig(
            self.output_path / filename,
            dpi=300,
            bbox_inches='tight'
        )

        plt.close()

        print(f"Saved: {filename}")

    def plot_umap_true_labels(
        self,
        umap_df,
        true_labels,
        filename="umap_true_labels.png"
        ):
        """
        Plot UMAP using true PMR labels.
        """

        plt.figure(figsize=(8, 6))

        scatter = plt.scatter(
            umap_df["UMAP1"],
            umap_df["UMAP2"],
            c=true_labels
        )

        plt.xlabel("UMAP1")
        plt.ylabel("UMAP2")

        plt.title("UMAP True Labels")

        plt.colorbar(scatter)

        plt.savefig(
            self.output_path / filename,
            dpi=300,
            bbox_inches='tight'
        )

        plt.close()

        print(f"Saved: {filename}")