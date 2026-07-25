"""
Train and evaluate forecasting models.
Usage: uv run python src/train.py --model seasonal_naive
       uv run python src/train.py --model lightgbm
"""

import argparse
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features.build_features import load_params, build_features
from models.lgbm_model import train_lgbm, predict_lgbm, get_feature_columns


MLFLOW_TRACKING_URI = "http://localhost:5001"
EXPERIMENT_NAME = "rossmann-forecasting"


def wmape(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def split_data(df: pd.DataFrame, params: dict, date_col: str = "Date"):
    s = params["split"]
    train = df[(df[date_col] >= s["train_start"]) & (df[date_col] <= s["train_end"])]
    val = df[(df[date_col] >= s["val_start"]) & (df[date_col] <= s["val_end"])]
    test = df[(df[date_col] >= s["test_start"]) & (df[date_col] <= s["test_end"])]
    return train, val, test


def evaluate_predictions(y_true: pd.Series, y_pred) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = ~np.isnan(y_pred) & ~np.isnan(y_true)
    y_true, y_pred = y_true[mask], y_pred[mask]
    return {
        "wmape": wmape(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "n_rows_evaluated": int(mask.sum()),
        "n_rows_dropped": int((~mask).sum()),
    }


def run_seasonal_naive(train, val, test, target_col):
    val_metrics = evaluate_predictions(val[target_col], val[f"{target_col}_lag_7"])
    test_metrics = evaluate_predictions(test[target_col], test[f"{target_col}_lag_7"])
    params_to_log = {"model_type": "seasonal_naive", "lag_used": 7}
    return params_to_log, val_metrics, test_metrics, None


def run_lightgbm(train, val, test, target_col):
    feature_cols = get_feature_columns(train, target_col)
    model = train_lgbm(train, val, feature_cols, target_col="Sales_transformed")

    val_preds = predict_lgbm(model, val, feature_cols)
    test_preds = predict_lgbm(model, test, feature_cols)

    val_metrics = evaluate_predictions(val[target_col], val_preds)
    test_metrics = evaluate_predictions(test[target_col], test_preds)

    params_to_log = {
        "model_type": "lightgbm",
        "num_features": len(feature_cols),
        "best_iteration": model.best_iteration,
    }
    return params_to_log, val_metrics, test_metrics, model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["seasonal_naive", "lightgbm"], required=True)
    args = parser.parse_args()

    params = load_params("params.yaml")
    target_col = params["train"]["target_col"]

    train_raw = pd.read_csv("data/raw/train.csv", parse_dates=["Date"])
    store_raw = pd.read_csv("data/raw/store.csv")

    df = build_features(train_raw, store_raw, params, filter_open_only=True)
    train, val, test = split_data(df, params)
    print(f"Train rows: {len(train)}, Val rows: {len(val)}, Test rows: {len(test)}")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    runners = {
        "seasonal_naive": run_seasonal_naive,
        "lightgbm": run_lightgbm,
    }

    with mlflow.start_run(run_name=args.model):
        model_params, val_metrics, test_metrics, model = runners[args.model](train, val, test, target_col)

        mlflow.log_params(model_params)
        mlflow.log_params({
            "train_start": params["split"]["train_start"],
            "train_end": params["split"]["train_end"],
        })
        for k, v in val_metrics.items():
            mlflow.log_metric(f"val_{k}", v)
        for k, v in test_metrics.items():
            mlflow.log_metric(f"test_{k}", v)

        if model is not None:
            mlflow.lightgbm.log_model(model, artifact_path="model")

        print("Validation metrics:", val_metrics)
        print("Test metrics:", test_metrics)


if __name__ == "__main__":
    main()