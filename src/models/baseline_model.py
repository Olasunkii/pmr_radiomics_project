from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix


class BaselineModel:
    """
    BaselineModel trains and evaluates a logistic regression model
    using all available features.
    """

    def __init__(self, logger):
        self.logger = logger
        self.model = LogisticRegression(max_iter=1000)

    def train(self, X_train, y_train):
        """
        Train logistic regression model.
        """
        self.logger.info("Training baseline Logistic Regression model...")
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance.
        """
        self.logger.info("Evaluating model...")

        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        self.logger.info(f"AUC: {auc}")
        self.logger.info(f"Confusion Matrix:\n{cm}")
        self.logger.info(f"Classification Report:\n{report}")

        return auc, cm, report