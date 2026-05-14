import pandas as pd
import joblib
from src.supervised.models.model_trainer import ModelTrainer


def train_and_evaluate_subset(name, X_train, X_test, y_train, y_test, config, logger):
    """
    Train and evaluate models on a given feature subset.
    """
    trainer = ModelTrainer(
        random_state=config.get("model", "random_state"),
        logger=logger
    )

    models = trainer.train_models(X_train, y_train)
    results = trainer.evaluate_models(models, X_test, y_test)

    rows = []
    for model_name, metrics in results.items():
        rows.append({
            "dataset": name,
            "model": model_name,
            "AUC": metrics["AUC"]
        })

    return models, results, pd.DataFrame(rows)


def save_best_model(models, results, name, model_path):
    """
    Save best model for a dataset.
    """
    best_model_name = max(results, key=lambda x: results[x]["AUC"])
    best_model = models[best_model_name]

    joblib.dump(best_model, model_path / f"best_model_{name}.pkl")

    print(f"Saved best model for {name}: {best_model_name}")


def run_experiments(X_train_df, X_test_df, y_train, y_test,
                    stable, moderate,
                    config, logger, log_path, model_path):

    # ===== FEATURE SETS =====
    stable_features = list(stable.index)
    stable_moderate_features = list(set(stable.index) | set(moderate.index))

    X_train_all = X_train_df.values
    X_test_all = X_test_df.values

    X_train_stable = X_train_df[stable_features].values
    X_test_stable = X_test_df[stable_features].values

    X_train_sm = X_train_df[stable_moderate_features].values
    X_test_sm = X_test_df[stable_moderate_features].values

    all_results = []

    # ===== RUN =====
    models_all, results_all, df_all = train_and_evaluate_subset(
        "all", X_train_all, X_test_all, y_train, y_test, config, logger
    )
    all_results.append(df_all)

    models_stable, results_stable, df_stable = train_and_evaluate_subset(
        "stable", X_train_stable, X_test_stable, y_train, y_test, config, logger
    )
    all_results.append(df_stable)

    models_sm, results_sm, df_sm = train_and_evaluate_subset(
        "stable_moderate", X_train_sm, X_test_sm, y_train, y_test, config, logger
    )
    all_results.append(df_sm)

    # ===== SAVE RESULTS =====
    final_df = pd.concat(all_results, ignore_index=True)
    final_df.to_csv(log_path / "all_model_results.csv", index=False)

    # ===== SAVE MODELS =====
    save_best_model(models_all, results_all, "all", model_path)
    save_best_model(models_stable, results_stable, "stable", model_path)
    save_best_model(models_sm, results_sm, "stable_moderate", model_path)


    best_models = {
        "all": (
            models_all[max(results_all, key=lambda x: results_all[x]["AUC"])],
            results_all[max(results_all, key=lambda x: results_all[x]["AUC"])]
        ),

        "stable": (
            models_stable[max(results_stable, key=lambda x: results_stable[x]["AUC"])],
            results_stable[max(results_stable, key=lambda x: results_stable[x]["AUC"])]
        ),

        "stable_moderate": (
            models_sm[max(results_sm, key=lambda x: results_sm[x]["AUC"])],
            results_sm[max(results_sm, key=lambda x: results_sm[x]["AUC"])]
        )
    }

    return final_df, best_models

