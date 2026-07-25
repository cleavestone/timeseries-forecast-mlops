"""
Train and evaluate the seasonal naive baseline.
Prediction(t) = actual(t - 7), i.e. the Sales_lag_7 feature directly.

This is the floor every subsequent model must beat.
"""

import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features.build_features import load_params, build_features


MLFLOW_TRACKING_URI = "http://localhost:5001"
EXPERIMENT_NAME = "rossmann-forecasting"


def wmape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean(np.abs(y_true - y_pred))


def split_data(df: pd.DataFrame, params: dict, date_col: str = "Date"):
    s = params["split"]
    train = df[(df[date_col] >= s["train_start"]) & (df[date_col] <= s["train_end"])]
    val = df[(df[date_col] >= s["val_start"]) & (df[date_col] <= s["val_end"])]
    test = df[(df[date_col] >= s["test_start"]) & (df[date_col] <= s["test_end"])]
    return train, val, test


def evaluate_predictions(y_true: pd.Series, y_pred: pd.Series) -> dict:
    mask = y_pred.notnull() & y_true.notnull()
    y_true, y_pred = y_true[mask].values, y_pred[mask].values
    return {
        "wmape": wmape(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "n_rows_evaluated": int(mask.sum()),
        "n_rows_dropped_no_lag": int((~mask).sum()),
    }


def main():
    params = load_params("params.yaml")
    target_col = params["train"]["target_col"]

    train_raw = pd.read_csv("data/raw/train.csv", parse_dates=["Date"])
    store_raw = pd.read_csv("data/raw/store.csv")

    df = build_features(train_raw, store_raw, params, filter_open_only=True)
    train, val, test = split_data(df, params)

    print(f"Train rows: {len(train)}, Val rows: {len(val)}, Test rows: {len(test)}")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="seasonal_naive_baseline"):
        mlflow.log_params({
            "model_type": "seasonal_naive",
            "lag_used": 7,
            "train_start": params["split"]["train_start"],
            "train_end": params["split"]["train_end"],
            "val_start": params["split"]["val_start"],
            "val_end": params["split"]["val_end"],
        })

        val_metrics = evaluate_predictions(val[target_col], val[f"{target_col}_lag_7"])
        test_metrics = evaluate_predictions(test[target_col], test[f"{target_col}_lag_7"])

        for k, v in val_metrics.items():
            mlflow.log_metric(f"val_{k}", v)
        for k, v in test_metrics.items():
            mlflow.log_metric(f"test_{k}", v)

        print("Validation metrics:", val_metrics)
        print("Test metrics:", test_metrics)


if __name__ == "__main__":
    main()