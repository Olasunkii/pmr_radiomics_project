from pathlib import Path
import joblib
import pandas as pd


def load_models():
    """
    Load trained models and feature sets.
    """

    base_path = Path(
        "outputs/experiments/baseline_experiment/models"
    )

    log_path = Path(
        "outputs/experiments/baseline_experiment/logs"
    )

    # =========================
    # LOAD MODELS
    # =========================

    model_all = joblib.load(
        base_path / "best_model_all.pkl"
    )

    model_stable = joblib.load(
        base_path / "best_model_stable.pkl"
    )

    model_sm = joblib.load(
        base_path / "best_model_stable_moderate.pkl"
    )

    # =========================
    # LOAD FEATURES
    # =========================

    stability_scores = pd.read_csv(
        log_path / "stability_scores.csv",
        index_col=0
    )

    stable_features = list(
        stability_scores[stability_scores["0"] > 0.7].index
    )

    moderate_features = list(
        stability_scores[
            (stability_scores["0"] <= 0.7) &
            (stability_scores["0"] > 0.3)
        ].index
    )

    stable_moderate = list(
        set(stable_features + moderate_features)
    )

    all_features = list(stability_scores.index)

    # =========================
    # RETURN STRUCTURE
    # =========================

    models = {
        "all": {
            "model": model_all,
            "features": all_features
        },

        "stable": {
            "model": model_stable,
            "features": stable_features
        },

        "stable_moderate": {
            "model": model_sm,
            "features": stable_moderate
        }
    }

    return models