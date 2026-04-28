from sklearn.linear_model import LogisticRegressionCV
import numpy as np


class LassoSelector:
    """
    Performs LASSO-based feature selection using LogisticRegressionCV.
    """

    def __init__(self, cv_folds: int, random_state: int, logger):
        """
        Initialize LassoSelector.

        Parameters
        ----------
        cv_folds : int
            Number of cross-validation folds
        random_state : int
            Random seed
        logger : logging.Logger
            Logger instance
        """
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.logger = logger
        self.model = None

    def fit(self, X_train, y_train):
        """
        Fit LASSO model with cross-validation.
        """
        self.logger.info("Running LASSO feature selection...")

        self.model = LogisticRegressionCV(
            penalty='l1',
            solver='saga',
            cv=self.cv_folds,
            random_state=self.random_state,
            max_iter=5000,
            n_jobs=-1,
            scoring='roc_auc'
        )

        self.model.fit(X_train, y_train)

    def get_selected_features(self, feature_names):
        """
        Return selected features (non-zero coefficients).
        """
        coef = self.model.coef_[0]
        selected = np.array(feature_names)[coef != 0]

        self.logger.info(f"Selected {len(selected)} features")

        return selected, coef