"""
Train and evaluate forecasting models.
Usage: uv run python src/train.py --model seasonal_naive
       uv run python src/train.py --model lightgbm
       uv run python src/train.py --model lightgbm_tuned
       uv run python src/train.py --model nbeats
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
from models.tuning import run_tuning_study
from models.nbeats_model import prepare_long_format, train_nbeats, predict_nbeats


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


def run_lightgbm_tuned(train, val, test, target_col, n_trials=25):
    study = run_tuning_study(train, val, target_col=target_col, n_trials=n_trials)

    best_params = study.best_params
    feature_cols = get_feature_columns(train, target_col)
    model = train_lgbm(train, val, feature_cols, target_col="Sales_transformed", params=best_params)

    val_preds = predict_lgbm(model, val, feature_cols)
    test_preds = predict_lgbm(model, test, feature_cols)

    val_metrics = evaluate_predictions(val[target_col], val_preds)
    test_metrics = evaluate_predictions(test[target_col], test_preds)

    params_to_log = {
        "model_type": "lightgbm_tuned",
        "n_trials": n_trials,
        "num_features": len(feature_cols),
        "best_iteration": model.best_iteration,
        **{f"best_{k}": v for k, v in best_params.items()},
    }
    return params_to_log, val_metrics, test_metrics, model


def run_nbeats(train_raw, val, test, target_col, params):
    """
    N-BEATS operates on raw long-format data, not the engineered feature table.
    train_raw here is the ORIGINAL raw dataframe (before build_features), filtered
    internally to the train period + Open==1.

    Note: this only forecasts the val horizon from train_end. Evaluating on the
    test period would require a second forecast from a later origin (rolling-origin
    evaluation) — out of scope here, documented as a deliberate limitation.
    """
    s = params["split"]
    train_period = train_raw[
        (train_raw["Date"] >= s["train_start"]) & (train_raw["Date"] <= s["train_end"])
    ]
    long_train = prepare_long_format(train_period)

    horizon = (pd.Timestamp(s["val_end"]) - pd.Timestamp(s["val_start"])).days + 1

    nf = train_nbeats(long_train, horizon=horizon, max_steps=300)
    preds = predict_nbeats(nf)

    preds["unique_id"] = preds["unique_id"].astype(int)
    preds = preds.rename(columns={"unique_id": "Store", "ds": "Date", "NBEATS": "y_pred"})

    val_eval = val[["Store", "Date", target_col]].merge(preds, on=["Store", "Date"], how="left")
    val_metrics = evaluate_predictions(val_eval[target_col], val_eval["y_pred"])

    # No test-period evaluation for this model — see docstring. Use NaN placeholders
    # (not None/strings) so downstream mlflow.log_metric calls don't break.
    test_metrics = {
        "wmape": float("nan"),
        "rmse": float("nan"),
        "mae": float("nan"),
        "n_rows_evaluated": 0,
        "n_rows_dropped": 0,
    }

    params_to_log = {
        "model_type": "nbeats",
        "horizon": horizon,
        "input_size": horizon * 4,
        "max_steps": 300,
        "test_eval_note": "not evaluated - single-horizon forecast from train_end only",
    }
    return params_to_log, val_metrics, test_metrics, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["seasonal_naive", "lightgbm", "lightgbm_tuned", "nbeats"],
        required=True,
    )
    parser.add_argument("--n-trials", type=int, default=25, help="Number of Optuna trials (lightgbm_tuned only)")
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

    with mlflow.start_run(run_name=args.model):
        if args.model == "seasonal_naive":
            model_params, val_metrics, test_metrics, model = run_seasonal_naive(train, val, test, target_col)
        elif args.model == "lightgbm":
            model_params, val_metrics, test_metrics, model = run_lightgbm(train, val, test, target_col)
        elif args.model == "lightgbm_tuned":
            model_params, val_metrics, test_metrics, model = run_lightgbm_tuned(
                train, val, test, target_col, n_trials=args.n_trials
            )
        elif args.model == "nbeats":
            model_params, val_metrics, test_metrics, model = run_nbeats(train_raw, val, test, target_col, params)

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