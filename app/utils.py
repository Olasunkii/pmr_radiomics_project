import pandas as pd
import numpy as np


# =========================
# BUILD INPUT DATAFRAME
# =========================

def build_input_dataframe(feature_names, input_values):
    """
    Convert user inputs into model-ready DataFrame.

    Parameters
    ----------
    feature_names : list
        Ordered feature names.

    input_values : dict
        User-provided values.

    Returns
    -------
    pd.DataFrame
    """

    row = []

    for feature in feature_names:
        row.append(input_values[feature])

    df = pd.DataFrame(
        [row],
        columns=feature_names
    )

    return df


# =========================
# PREDICT
# =========================

def predict_model(model, input_df):
    """
    Generate prediction and probability.
    """

    prediction = model.predict(input_df)[0]

    probability = model.predict_proba(input_df)[0][1]

    return prediction, probability


# =========================
# FORMAT OUTPUT
# =========================

def format_prediction(prediction):
    """
    Convert numeric prediction to readable label.
    """

    if prediction == 1:
        return "PMR Likely"

    return "PMR Unlikely"