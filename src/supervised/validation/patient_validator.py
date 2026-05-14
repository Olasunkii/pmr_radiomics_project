import pandas as pd
import joblib


class PatientValidator:
    """
    Validate trained models on individual patients.

    Supports:
    - patient-level predictions
    - probability comparison
    - multi-model validation
    """

    def __init__(self, logger=None):
        self.logger = logger

    def load_model(self, model_path):
        """
        Load trained model.
        """
        return joblib.load(model_path)

    def predict_patient(
        self,
        model,
        scaler,
        patient_df,
        feature_subset
    ):
        """
        Predict a single patient.
        """

        # Scale full feature space
        full_features = scaler.feature_names_in_

        aligned = pd.DataFrame(
            0,
            index=[0],
            columns=full_features
        )

        for col in feature_subset:
            aligned[col] = patient_df[col].values[0]

        scaled = scaler.transform(aligned)

        scaled_df = pd.DataFrame(
            scaled,
            columns=full_features
        )

        model_input = scaled_df[feature_subset]

        prediction = model.predict(model_input)[0]
        probability = model.predict_proba(model_input)[0][1]

        return prediction, probability

    def validate_patient(
        self,
        patient_index,
        X,
        y,
        models_dict,
        scaler
    ):
        """
        Run validation across all models
        for one patient.
        """

        patient = X.iloc[[patient_index]]
        true_label = y.iloc[patient_index]

        rows = []

        for name, model_info in models_dict.items():

            model = model_info["model"]
            features = model_info["features"]

            pred, prob = self.predict_patient(
                model=model,
                scaler=scaler,
                patient_df=patient,
                feature_subset=features
            )

            rows.append({
                "model_group": name,
                "true_label": true_label,
                "prediction": pred,
                "probability": round(prob, 4)
            })

        results_df = pd.DataFrame(rows)

        return patient, results_df
    

    def generate_prediction_report(
        self,
        model,
        X,
        y
        ):
        """
        Generate prediction report for all patients.
        """

        probs = model.predict_proba(X)[:, 1]
        preds = model.predict(X)

        report_df = pd.DataFrame({
            "true_label": y,
            "prediction": preds,
            "probability": probs
        })

        report_df["correct"] = (
            report_df["true_label"] ==
            report_df["prediction"]
        )

        report_df["confidence"] = abs(
            report_df["probability"] - 0.5
        )

        return report_df
    
    def select_cases(self, report_df):
        """
        Select important clinical cases.
        """

        selected = {}

        # Correct PMR
        pmr_correct = report_df[
            (report_df["true_label"] == 1) &
            (report_df["correct"])
        ]

        if len(pmr_correct) > 0:
            selected["best_pmr"] = pmr_correct.sort_values(
                "probability",
                ascending=False
            ).index[0]

        # Correct non-PMR
        non_pmr_correct = report_df[
            (report_df["true_label"] == 0) &
            (report_df["correct"])
        ]

        if len(non_pmr_correct) > 0:
            selected["best_non_pmr"] = non_pmr_correct.sort_values(
                "probability",
                ascending=True
            ).index[0]

        # Most uncertain
        uncertain = report_df.sort_values(
            "confidence",
            ascending=True
        )

        selected["most_uncertain"] = uncertain.index[0]

        # First misclassified
        incorrect = report_df[
            ~report_df["correct"]
        ]

        if len(incorrect) > 0:
            selected["misclassified"] = incorrect.index[0]

        return selected
if __name__ == "__main__":
    print("PatientValidator module loaded successfully.")