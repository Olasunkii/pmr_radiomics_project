import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


class ModelTrainer:
    """
    Train and evaluate multiple ML models for comparison.
    """

    def __init__(self, random_state=42, logger=None):
        self.random_state = random_state
        self.logger = logger

    def prepare_data(self, X_train, X_test):
        """
        Scale data for models that require normalization.
        """
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        return X_train_scaled, X_test_scaled

    def train_models(self, X_train, y_train):
        """
        Train all models.
        """
        models = {}

        # Logistic Regression
        models["LR"] = LogisticRegression(max_iter=5000)

        # Random Forest
        models["RF"] = RandomForestClassifier(
            n_estimators=100,
            random_state=self.random_state
        )

        # XGBoost
        models["XGB"] = XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=self.random_state
        )

        # Fit all models
        for name, model in models.items():
            if self.logger:
                self.logger.info(f"Training {name}...")
            model.fit(X_train, y_train)

        # Ensemble
        ensemble = VotingClassifier(
            estimators=[
                ("lr", models["LR"]),
                ("rf", models["RF"]),
                ("xgb", models["XGB"])
            ],
            voting='soft'
        )

        ensemble.fit(X_train, y_train)
        models["Ensemble"] = ensemble

        return models

    def evaluate_models(self, models, X_test, y_test):
        """
        Evaluate all models.
        """
        results = {}

        for name, model in models.items():

            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            auc = roc_auc_score(y_test, y_prob)

            results[name] = {
                "AUC": auc,
                "confusion_matrix": confusion_matrix(y_test, y_pred),
                "report": classification_report(y_test, y_pred, output_dict=True)
            }

            if self.logger:
                self.logger.info(f"{name} AUC: {auc:.3f}")

        return results