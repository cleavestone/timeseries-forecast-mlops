"""
Optuna hyperparameter tuning for LightGBM.
"""

import mlflow
import optuna
import pandas as pd

from models.lgbm_model import train_lgbm, predict_lgbm, get_feature_columns


def wmape(y_true, y_pred):
    import numpy as np
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))


def make_objective(train_df: pd.DataFrame, val_df: pd.DataFrame, feature_cols: list, target_col: str):
    def objective(trial: optuna.Trial) -> float:
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 15, 255),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 200),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        }

        with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
            mlflow.log_params(params)

            model = train_lgbm(train_df, val_df, feature_cols, target_col=target_col, params=params)
            val_preds = predict_lgbm(model, val_df, feature_cols)

            score = wmape(val_df["Sales"].values, val_preds)
            mlflow.log_metric("val_wmape", score)

        return score

    return objective


def run_tuning_study(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    target_col: str = "Sales",
    n_trials: int = 25,
) -> optuna.Study:
    feature_cols = get_feature_columns(train_df, target_col)
    objective = make_objective(train_df, val_df, feature_cols, "Sales_transformed")

    study = optuna.create_study(direction="minimize", study_name="lightgbm_wmape_tuning")
    study.optimize(objective, n_trials=n_trials)

    return study