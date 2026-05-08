from pathlib import Path
import joblib
import pandas as pd


# =========================
# BASE PATH
# =========================

BASE_PATH = Path(
    "outputs/experiments/baseline_experiment"
)

MODEL_PATH = BASE_PATH / "models"
LOG_PATH = BASE_PATH / "logs"


# =========================
# LOAD MODELS
# =========================

def load_models():
    """
    Load trained supervised models.
    """

    models = {
        "All Features": joblib.load(
            MODEL_PATH / "best_model_all.pkl"
        ),

        "Stable Features": joblib.load(
            MODEL_PATH / "best_model_stable.pkl"
        ),

        "Stable + Moderate": joblib.load(
            MODEL_PATH / "best_model_stable_moderate.pkl"
        )
    }

    return models


# =========================
# LOAD FEATURE LISTS
# =========================

def load_feature_sets():
    """
    Load feature subsets used by models.
    """

    stable = pd.read_csv(
        LOG_PATH / "stable_features.csv"
    )["0"].tolist()

    stable_moderate = pd.read_csv(
        LOG_PATH / "stable_moderate_features.csv"
    )["0"].tolist()

    all_features = pd.read_csv(
        LOG_PATH / "all_features.csv"
    )["0"].tolist()

    return {
        "All Features": all_features,
        "Stable Features": stable,
        "Stable + Moderate": stable_moderate
    }