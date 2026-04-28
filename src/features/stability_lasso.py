import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


class StabilityLasso:
    """
    Stability-based feature selection using repeated
    L1-regularized logistic regression with stratified CV.

    This implementation estimates feature selection frequency
    across repeated resampling iterations.
    """

    def __init__(self, n_iterations=50, n_splits=5, random_state=42, logger=None):
        """
        Parameters
        ----------
        n_iterations : int
            Number of repeated stability runs.
        n_splits : int
            Number of folds in StratifiedKFold.
        random_state : int
            Reproducibility seed.
        logger : logging.Logger
            Optional logger.
        """
        self.n_iterations = n_iterations
        self.n_splits = n_splits
        self.random_state = random_state
        self.logger = logger

    def run(self, X, y):
        """
        Compute stability scores for all features.

        Returns
        -------
        pd.Series
            Feature stability scores (0–1).
        """

        if self.logger:
            self.logger.info("Running Stability LASSO...")

        feature_names = X.columns
        selection_counts = pd.Series(0, index=feature_names)

        skf = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state
        )

        total_runs = self.n_iterations * self.n_splits

        for i in range(self.n_iterations):

            for train_idx, _ in skf.split(X, y):

                X_train = X.iloc[train_idx]
                y_train = y.iloc[train_idx]

                # Standardization (critical in LASSO)
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)

                model = LogisticRegression(
                    penalty='l1',
                    solver='saga',
                    max_iter=10000,
                    C=1.0,
                    random_state=self.random_state + i
                )

                model.fit(X_train_scaled, y_train)

                coefs = model.coef_[0]

                selected = feature_names[np.abs(coefs) > 1e-6]

                selection_counts[selected] += 1

        stability = (selection_counts / total_runs).sort_values(ascending=False)

        return stability

    def categorize(self, stability_series, stable_thr=0.7, moderate_thr=0.3):
        """
        Categorize features into stability groups.
        """

        stable = stability_series[stability_series > stable_thr]
        moderate = stability_series[
            (stability_series <= stable_thr) &
            (stability_series > moderate_thr)
        ]
        unstable = stability_series[stability_series <= moderate_thr]

        return stable, moderate, unstable