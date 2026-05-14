from pathlib import Path
import pandas as pd
import numpy as np
import random
import joblib

from src.config_loader import ConfigLoader
from src.utils.logger import setup_logger
from src.preprocessing.data_loader import DataLoader
from src.preprocessing.data_cleaner import DataCleaner
from src.eda_analyzer import EDAAnalyzer
from src.preprocessing.preprocessor import Preprocessor

from src.supervised.features.lasso_selector import LassoSelector
from src.supervised.features.stability_lasso import StabilityLasso
from src.utils.visualizer import Visualizer
from src.supervised.interpretability.shap_analyzer import ShapAnalyzer

from src.supervised.models.experiment_runner import run_experiments
from src.supervised.interpretability.shap_runner import run_shap_analysis
from src.supervised.validation.patient_validator import PatientValidator

# UNSUPERVISED LIBRARY
from src.unsupervised.dimensionality import DimensionalityReducer
from src.unsupervised.clusterer import Clusterer
from src.unsupervised.unsupervised_visualizer import UnsupervisedVisualizer
from src.unsupervised.cluster_profiler import ClusterProfiler
from src.unsupervised.cluster_stability import ClusterStability
from src.unsupervised.cluster_statistics import ClusterStatistics

#REPORT
from src.reporting.report_generator import ReportGenerator

def main():

    np.random.seed(42)
    random.seed(42)

    config = ConfigLoader()
    run_cfg = config.get("run")
    exp_name = config.get("experiment", "name")

    # ===== OUTPUT =====
    supervised_path = Path("outputs/supervised")
    fig_path = supervised_path / "figures"
    log_path = supervised_path / "logs"
    model_path = supervised_path / "models"
    report_path = supervised_path / "reports"

    # =====================================
    # UNSUPERVISED OUTPUTS
    # =====================================

    unsup_path = Path("outputs/unsupervised")

    unsup_fig_path = unsup_path / "figures"
    unsup_log_path = unsup_path / "logs"
    unsup_report_path = unsup_path / "reports"

    # Supervised
    fig_path.mkdir(parents=True, exist_ok=True)
    log_path.mkdir(parents=True, exist_ok=True)
    model_path.mkdir(parents=True, exist_ok=True)
    report_path.mkdir(parents=True, exist_ok=True)

    # Unsupervised
    unsup_fig_path.mkdir(parents=True, exist_ok=True)
    unsup_log_path.mkdir(parents=True, exist_ok=True)
    unsup_report_path.mkdir(parents=True, exist_ok=True)

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

    X_train_df = pd.DataFrame(X_train_scaled, columns=X_train.columns).reset_index(drop=True)
    X_test_df = pd.DataFrame(X_test_scaled, columns=X_train.columns).reset_index(drop=True)

    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

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

    X_test_sets = {
        "all": X_test_df,

        "stable":
        X_test_df[stable_features],

        "stable_moderate":
        X_test_df[stable_moderate_features]
    }


    # ======================
    # ROC CURVES
    # ======================
    roc_results = {
        "All": best_models["all"][1],
        "Stable": best_models["stable"][1],
        "Stable+Moderate": best_models["stable_moderate"][1]
    }

    visualizer.plot_roc_curves(
        best_models,
        X_test_sets,
        y_test
    )

    # ======================
    # CALIBRATION CURVES
    # ======================
    visualizer.plot_calibration_curves(
        best_models,
        X_test_sets,
        y_test,
        filename="calibration_curves.png"
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
        # PATIENT VALIDATION
        # ======================
        shap_values = shap_analyzer.compute_shap(
            best_model,
            X_test_df
        )

        validator = PatientValidator(
            logger=loader.logger
        )

        report_df = validator.generate_prediction_report(
            best_model,
            X_test_df,
            y_test
        )

        report_df.to_csv(
            log_path / "patient_prediction_report.csv",
            index=False
        )

        selected_cases = validator.select_cases(report_df)

        print("\nSelected validation cases:")
        print(selected_cases)

        # Generate waterfall plots
        for case_name, patient_idx in selected_cases.items():

            shap_analyzer.save_waterfall_plot(
                shap_values,
                patient_index=patient_idx,
                filename=f"waterfall_{case_name}.png"
            )
        # ======================
        # WATERFALL PLOT
        # ======================
        shap_analyzer.save_waterfall_plot(
            shap_values,
            patient_index=0,
            filename="shap_waterfall_patient0.png"
        )

    # =====================================
    # UNSUPERVISED LEARNING
    # =====================================
    reducer = DimensionalityReducer(
        logger=logger
    )

    pca_df, explained = reducer.run_pca(
        X_train_df
    )

    clusterer = Clusterer(
        random_state=42,
        logger=logger
    )

    kmeans_model, cluster_labels, silhouette = (
        clusterer.run_kmeans(
            X_train_df,
            n_clusters=2
        )
    )

    unsup_visualizer = UnsupervisedVisualizer(
        unsup_fig_path
    )

    unsup_visualizer.plot_pca_clusters(
        pca_df,
        cluster_labels
    )


    # =====================================
    # UMAP
    # =====================================

    umap_df = reducer.run_umap(
        X_train_df
    )

    unsup_visualizer.plot_umap_clusters(
        umap_df,
        cluster_labels
    )

    unsup_visualizer.plot_umap_true_labels(
        umap_df,
        y_train
    )

    # =====================================
    # MULTI-K EVALUATION
    # =====================================

    k_results = clusterer.evaluate_k_range(
        X_train_df
    )

    k_results.to_csv(
        unsup_log_path / "kmeans_evaluation.csv",
        index=False
    )

    unsup_visualizer.plot_elbow_curve(
        k_results
    )

    unsup_visualizer.plot_silhouette_scores(
        k_results
    )

    # =====================================
    # CLUSTER vs TRUE LABELS
    # =====================================

    comparison_df, summary_df = (
        clusterer.compare_clusters_to_labels(
            cluster_labels,
            y_train
        )
    )

    comparison_df.to_csv(
        unsup_log_path / "cluster_vs_labels.csv",
        index=False
    )

    summary_df.to_csv(
        unsup_log_path / "cluster_label_summary.csv"
    )

    unsup_visualizer.plot_cluster_label_heatmap(
        summary_df
    )

    # =====================================
    # CLUSTER PROFILING
    # =====================================

    profiler = ClusterProfiler(
        unsup_log_path,
        logger=logger
    )

    cluster_means = profiler.compute_cluster_means(
        X_train_df,
        cluster_labels
    )

    cluster_std = profiler.compute_cluster_std(
        X_train_df,
        cluster_labels
    )

    profiler.plot_cluster_heatmap(
        cluster_means,
        top_n=25
    )

    # =====================================
    # CLUSTER STABILITY
    # =====================================

    stability_checker = ClusterStability(
        n_runs=20,
        n_clusters=2,
        random_state=42,
        logger=logger
    )

    stability_df = stability_checker.evaluate(
        X_train_df
    )

    stability_df.to_csv(
        unsup_log_path / "cluster_stability_scores.csv",
        index=False
    )

    stability_summary = stability_checker.summarize(
        stability_df
    )

    stability_summary.to_csv(
        unsup_log_path / "cluster_stability_summary.csv",
        index=False
    )

    print("\nCluster Stability Summary:")
    print(stability_summary)

    # =====================================
    # CLUSTER STATISTICAL ANALYSIS
    # =====================================

    cluster_stats = ClusterStatistics(
        logger=logger
    )

    stats_df = cluster_stats.compare_clusters(
        X_train_df,
        cluster_labels
    )

    # =====================================
    # FDR CORRECTION
    # =====================================

    stats_df = cluster_stats.apply_fdr_correction(
        stats_df,
        alpha=0.05
    )

    # =====================================
    # EFFECT SIZE ANALYSIS
    # =====================================

    stats_df = cluster_stats.compute_effect_sizes(
        X_train_df,
        cluster_labels,
        stats_df
    )

    # Save full statistical table
    stats_df.to_csv(
        unsup_log_path / "cluster_statistical_tests.csv",
        index=False
    )

    # Extract significant features only
    significant_df = stats_df[
        stats_df["significant_fdr"] == True
    ]

    significant_df.to_csv(
        unsup_log_path / "significant_cluster_features.csv",
        index=False
    )

    # =====================================
    # VOLCANO PLOT
    # =====================================

    unsup_visualizer.plot_volcano(
        stats_df
    )

    print("\nTop Significant Cluster Features:")
    print(significant_df.head(10))

    # =====================================
    # AUTOMATED REPORT
    # =====================================

    report_path = base_output / "report"

    reporter = ReportGenerator(
        report_path
    )

    reporter.generate_markdown_report(
        experiment_name=exp_name,
        results_df=results_df,
        stability_summary=stability_summary,
        significant_df=significant_df,
        stable_count=len(stable.index),
        sm_count=len(
            set(stable.index) | set(moderate.index)
        ),
        all_count=X_train_df.shape[1]
    )

    print("\nPipeline completed successfully.")

if __name__ == "__main__":
    main()