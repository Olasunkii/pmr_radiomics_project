import optuna
import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


class ModelTuner:
    """
    ModelTuner performs hyperparameter optimization using Optuna
    for multiple machine learning models.

    It optimizes models based on cross-validated ROC-AUC.
    """

    def __init__(self, n_trials=50, random_state=42, logger=None):
        self.n_trials = n_trials
        self.random_state = random_state
        self.logger = logger

    def tune_logistic(self, X, y):
        """Tune Logistic Regression using Optuna."""

        def objective(trial):
            C = trial.suggest_loguniform("C", 1e-4, 10)

            model = LogisticRegression(
                penalty='l1',
                solver='saga',
                C=C,
                max_iter=5000,
                random_state=self.random_state
            )

            scores = cross_val_score(
                model, X, y,
                cv=5,
                scoring='roc_auc'
            )

            return np.mean(scores)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.n_trials)

        return study.best_params, study.best_value


    def tune_rf(self, X, y):
        """Tune Random Forest."""

        def objective(trial):
            model = RandomForestClassifier(
                n_estimators=trial.suggest_int("n_estimators", 100, 500),
                max_depth=trial.suggest_int("max_depth", 3, 15),
                min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
                random_state=self.random_state
            )

            scores = cross_val_score(
                model, X, y,
                cv=5,
                scoring='roc_auc'
            )

            return np.mean(scores)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.n_trials)

        return study.best_params, study.best_value


    def tune_xgb(self, X, y):
        """Tune XGBoost."""

        def objective(trial):
            model = XGBClassifier(
                n_estimators=trial.suggest_int("n_estimators", 100, 500),
                max_depth=trial.suggest_int("max_depth", 3, 10),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3),
                subsample=trial.suggest_float("subsample", 0.6, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
                eval_metric='logloss',
                random_state=self.random_state
            )

            scores = cross_val_score(
                model, X, y,
                cv=5,
                scoring='roc_auc'
            )

            return np.mean(scores)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.n_trials)

        return study.best_params, study.best_value