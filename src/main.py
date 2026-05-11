from pathlib import Path
import pandas as pd
import numpy as np
import random
import joblib

from src.config_loader import ConfigLoader
from src.utils.logger import setup_logger
from src.data_loader import DataLoader
from src.data_cleaner import DataCleaner
from src.eda_analyzer import EDAAnalyzer
from src.preprocessor import Preprocessor
from src.features.lasso_selector import LassoSelector
from src.features.stability_lasso import StabilityLasso
from src.utils.visualizer import Visualizer
from src.interpretability.shap_analyzer import ShapAnalyzer

# NEW IMPORTS
from src.models.experiment_runner import run_experiments
from src.interpretability.shap_runner import run_shap_analysis


def main():

    np.random.seed(42)
    random.seed(42)

    config = ConfigLoader()
    run_cfg = config.get("run")
    exp_name = config.get("experiment", "name")

    # ===== OUTPUT =====
    base_output = Path("outputs/experiments") / exp_name
    fig_path = base_output / "figures"
    log_path = base_output / "logs"
    model_path = base_output / "models"

    fig_path.mkdir(parents=True, exist_ok=True)
    log_path.mkdir(parents=True, exist_ok=True)
    model_path.mkdir(parents=True, exist_ok=True)

    # ===== LOGGER =====
    logger = setup_logger(log_path / "pipeline.log")

    # ===== LOAD =====
    loader = DataLoader(config.get_path("data", "raw_file"), logger=logger)
    df = loader.load_data()
    loader.validate_data(df)

    # ===== CLEAN =====
    cleaner = DataCleaner(logger=logger)
    df_clean = cleaner.encode_target(df)
    df_clean = cleaner.drop_columns(df_clean)
    df_clean = cleaner.handle_missing(df_clean)

    # ===== EDA =====
    eda = EDAAnalyzer(fig_path, logger)
    eda.basic_info(df_clean)
    eda.target_distribution(df_clean)
    eda.summary_statistics(df_clean)
    eda.skewness(df_clean)

    corr = eda.correlation_matrix(df_clean)
    eda.high_correlation(corr)

    # ===== PREPROCESS =====
    preprocessor = Preprocessor(
        test_size=config.get("model", "test_size"),
        random_state=config.get("model", "random_state"),
        logger=logger
    )

    X_train, X_test, y_train, y_test = preprocessor.split(df_clean)
    X_train_scaled, X_test_scaled = preprocessor.scale(X_train, X_test)

    # ======================
    # SAVE SCALER
    # ======================
    if run_cfg["save_models"]:
        joblib.dump(
            preprocessor.scaler,
            model_path / "scaler.pkl"
        )

        print("Saved: scaler.pkl")

    X_train_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_df = pd.DataFrame(X_test_scaled, columns=X_train.columns)

    # ===== FEATURE SELECTION =====
    if run_cfg["use_lasso"]:
        lasso = LassoSelector(
            cv_folds=config.get("lasso", "cv_folds"),
            random_state=config.get("model", "random_state"),
            logger=logger
        )
        lasso.fit(X_train_scaled, y_train)
        selected_features, _ = lasso.get_selected_features(X_train.columns)
        pd.Series(selected_features).to_csv(log_path / "lasso_selected_features.csv")

    if run_cfg["use_stability"]:
        stability = StabilityLasso(logger=logger)
        stability_scores = stability.run(X_train_df, y_train)
        stable, moderate, unstable = stability.categorize(stability_scores)
        stability_scores.to_csv(log_path / "stability_scores.csv")

    # ===== EXPERIMENTS =====
    results_df, best_models = run_experiments(
        X_train_df, X_test_df, y_train, y_test,
        stable, moderate,
        config, logger, log_path, model_path
    )

    # ===== VISUALIZATION =====
    visualizer = Visualizer(fig_path)
    visualizer.plot_model_performance(results_df)
    visualizer.plot_grouped_auc(results_df)
    visualizer.plot_auc_heatmap(results_df)

    visualizer.plot_feature_counts(
    all_count=X_train_df.shape[1],
    stable_count=len(stable.index),
    sm_count=len(set(stable.index) | set(moderate.index))
    )

    # ======================
    # FEATURE LISTS
    # ======================
    stable_features = list(stable.index)

    stable_moderate_features = list(
        set(stable.index) | set(moderate.index)
    )
    # ======================
    # SAVE FEATURE SUBSETS
    # ======================

    pd.Series(X_train.columns).to_csv(
        log_path / "all_features.csv",
        index=False
    )

    pd.Series(stable_features).to_csv(
        log_path / "stable_features.csv",
        index=False
    )

    pd.Series(stable_moderate_features).to_csv(
        log_path / "stable_moderate_features.csv",
        index=False
    )

    # ======================
    # CONFUSION MATRICES
    # ======================

    visualizer.plot_confusion_matrix(
        best_models["all"][1]["confusion_matrix"],
        "Confusion Matrix - All Features",
        "confusion_matrix_all.png"
    )

    visualizer.plot_confusion_matrix(
        best_models["stable"][1]["confusion_matrix"],
        "Confusion Matrix - Stable Features",
        "confusion_matrix_stable.png"
    )

    visualizer.plot_confusion_matrix(
        best_models["stable_moderate"][1]["confusion_matrix"],
        "Confusion Matrix - Stable + Moderate Features",
        "confusion_matrix_stable_moderate.png"
    )


    if run_cfg["use_stability"]:
        visualizer.plot_stability(stability_scores)

    # ===== SHAP =====
    if run_cfg["use_shap"]:
        shap_analyzer = ShapAnalyzer(fig_path)

        # Use ALL best model for explanation
        # import joblib
        best_model = joblib.load(model_path / "best_model_all.pkl")

        run_shap_analysis(
            shap_analyzer,
            best_model,
            X_test_df,
            stable,
            moderate,
            unstable
        )
        # ======================
        # WATERFALL PLOT
        # ======================

        shap_values = shap_analyzer.compute_shap(
            best_model,
            X_test_df
        )

        shap_analyzer.save_waterfall_plot(
            shap_values,
            patient_index=0,
            filename="shap_waterfall_patient0.png"
        )
        
    print("\nPipeline completed successfully.")

if __name__ == "__main__":
    main()