from sklearn.decomposition import PCA
import pandas as pd
import umap.umap_ as umap

class DimensionalityReducer:
    """
    Dimensionality reduction utilities.
    """

    def __init__(self, logger=None):
        self.logger = logger

    def run_pca(self, X, n_components=2):
        """
        Run PCA projection.
        """

        pca = PCA(n_components=n_components)

        components = pca.fit_transform(X)

        explained = pca.explained_variance_ratio_

        pca_df = pd.DataFrame(
            components,
            columns=["PC1", "PC2"]
        )

        if self.logger:
            self.logger.info(
                f"PCA explained variance: {explained}"
            )

        return pca_df, explained
    
    def run_umap(
        self,
        X,
        n_components=2,
        random_state=42
        ):
        """
        Run UMAP projection.
        """

        reducer = umap.UMAP(
            n_components=n_components,
            random_state=random_state
        )

        embedding = reducer.fit_transform(X)

        umap_df = pd.DataFrame(
            embedding,
            columns=["UMAP1", "UMAP2"]
        )

        if self.logger:
            self.logger.info(
                "UMAP projection completed."
            )

        return umap_df