def run_shap_analysis(shap_analyzer, best_model, X_test_df,
                      stable, moderate, unstable):

    shap_values = shap_analyzer.compute_shap(best_model, X_test_df)

    shap_analyzer.save_summary_plot(shap_values, X_test_df, "shap_summary.png")
    shap_analyzer.save_bar_plot(shap_values, "shap_bar.png")

    def run_subset(feature_index, name, plot_type):
        valid = [f for f in feature_index if f in X_test_df.columns]
        if not valid:
            return

        X_sub = X_test_df[valid]
        shap_sub = shap_values[:, valid]

        if plot_type == "summary":
            shap_analyzer.save_summary_plot(shap_sub, X_sub, f"shap_summary_{name}.png")

        elif plot_type == "violin":
            shap_analyzer.save_violin_plot(shap_sub, X_sub, f"shap_violin_{name}.png")

        elif plot_type == "heatmap":
            shap_analyzer.save_heatmap_plot(shap_sub, f"shap_heatmap_{name}.png")

    run_subset(stable.index, "stable", "summary")
    run_subset(moderate.index, "moderate", "violin")

    top_unstable = unstable.sort_values(ascending=False).head(30)
    run_subset(top_unstable.index, "unstable", "heatmap")