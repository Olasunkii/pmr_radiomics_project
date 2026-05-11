import streamlit as st
import pandas as pd

from loaders import load_models
from utils import (
    build_input_dataframe,
    predict_model,
    format_prediction
)


# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="PMR Prediction System",
    layout="wide"
)

st.title("PMR Radiomics Prediction System")

st.markdown("""
This application predicts PMR likelihood using trained supervised machine learning models.
""")


# =========================
# LOAD MODELS
# =========================

models = load_models()

model_choice = st.sidebar.selectbox(
    "Select Feature Set",
    list(models.keys())
)

selected = models[model_choice]

model = selected["model"]
scaler = selected["scaler"]
features = selected["features"]


# =========================
# USER INPUTS
# =========================

st.header("Patient Feature Inputs")

input_values = {}

# Create responsive columns
cols = st.columns(3)

for idx, feature in enumerate(features):

    col = cols[idx % 3]

    with col:
        value = st.number_input(
            feature,
            value=0.0,
            format="%.4f"
        )

        input_values[feature] = value


# =========================
# PREDICTION
# =========================

if st.button("Predict PMR"):

    input_df = build_input_dataframe(
        features,
        input_values
    )

    # Scale input
    # =========================
    # ALIGN FEATURES WITH TRAINING SPACE
    # =========================

    full_feature_names = scaler.feature_names_in_

    # create full zero vector
    aligned_input = pd.DataFrame(
        0,
        index=[0],
        columns=full_feature_names
    )

    # fill only selected features
    for f in features:
        aligned_input[f] = input_df[f]

    # scale correctly
    input_scaled = scaler.transform(aligned_input)

    input_scaled_df = pd.DataFrame(
        input_scaled,
        columns=full_feature_names
    )
    # Keep only features expected by selected model
    model_input_df = input_scaled_df[features]

    prediction, probability = predict_model(
        model,
        model_input_df
    )
    label = format_prediction(prediction)

    st.subheader("Prediction Result")

    st.success(label)

    st.metric(
        label="PMR Probability",
        value=f"{probability:.3f}"
    )

    st.dataframe(input_df)