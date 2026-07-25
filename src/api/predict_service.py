"""
Batch prediction service.

Loads the Production-stage model from the MLflow Model Registry and serves
forecasts using the SAME build_features() pipeline used in training — this
is what guarantees train/serve parity.

Uses recursive forecasting: each day's prediction is fed back into the
working history so subsequent days' lag/rolling features can reference it.
"""

import sys
from pathlib import Path
from datetime import timedelta

import mlflow
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from features.build_features import load_params, build_features


MLFLOW_TRACKING_URI = "http://localhost:5001"
MODEL_NAME = "rossmann-sales-forecaster"
MODEL_STAGE = "Production"

MIN_HISTORY_DAYS = 35  # must exceed our longest lag/rolling window (28) with margin


class PredictionService:
    def __init__(self):
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        self.model = mlflow.lightgbm.load_model(f"models:/{MODEL_NAME}/{MODEL_STAGE}")
        self.params = load_params("params.yaml")
        self.target_col = self.params["train"]["target_col"]

        # TEMP: reading from raw CSV until Postgres-backed history store is wired in.
        self.train_raw = pd.read_csv("data/raw/train.csv", parse_dates=["Date"])
        self.store_raw = pd.read_csv("data/raw/store.csv")

    def _get_store_history(self, store_id: int, forecast_start: pd.Timestamp) -> pd.DataFrame:
        history = self.train_raw[
            (self.train_raw["Store"] == store_id) & (self.train_raw["Date"] < forecast_start)
        ].copy()
        return history

    def _get_feature_columns(self):
        from models.lgbm_model import get_feature_columns, prepare_lgbm_data
        return get_feature_columns, prepare_lgbm_data

    def predict_store(self, store_id: int, forecast_start: str, forecast_end: str) -> pd.DataFrame:
        forecast_start = pd.Timestamp(forecast_start)
        forecast_end = pd.Timestamp(forecast_end)

        history = self._get_store_history(store_id, forecast_start)
        if history["Date"].nunique() < MIN_HISTORY_DAYS:
            raise ValueError(
                f"Store {store_id} has insufficient history "
                f"({history['Date'].nunique()} days) before {forecast_start.date()} "
                f"to compute reliable lag/rolling features. Minimum: {MIN_HISTORY_DAYS} days."
            )

        get_feature_columns, prepare_lgbm_data = self._get_feature_columns()

        working_df = history.copy()
        forecast_dates = pd.date_range(forecast_start, forecast_end, freq="D")
        predictions = []

        for current_date in forecast_dates:
            # Append a placeholder row for the date we're about to predict.
            # Open/Promo/etc for future dates are assumed known (as they would be
            # in a real batch job — promo calendars are typically planned in advance).
            new_row = self._build_future_row(store_id, current_date)
            working_df = pd.concat([working_df, new_row], ignore_index=True)

            featured = build_features(working_df, self.store_raw, self.params, filter_open_only=False)
            today_row = featured[featured["Date"] == current_date]

            if today_row.empty:
                continue

            feature_cols = get_feature_columns(featured, self.target_col)
            X = prepare_lgbm_data(today_row, feature_cols)
            pred_log = self.model.predict(X, num_iteration=self.model.best_iteration)
            pred_sales = float(np.expm1(pred_log[0]))
            pred_sales = max(pred_sales, 0.0)  # sales can't be negative

            predictions.append({
                "Store": store_id,
                "Date": current_date,
                "predicted_sales": pred_sales,
            })

            # Feed the prediction back in as if it were observed, so the NEXT
            # iteration's lag/rolling features can reference it.
            working_df.loc[working_df["Date"] == current_date, "Sales"] = pred_sales

        return pd.DataFrame(predictions)

    def _build_future_row(self, store_id: int, date: pd.Timestamp) -> pd.DataFrame:
        """
        Constructs a single future row with assumed-known operational fields
        (Open, Promo, StateHoliday, SchoolHoliday). In a real system these would
        come from a promo/holiday calendar input, not guessed — this is a
        simplification worth flagging explicitly for v1.
        """
        return pd.DataFrame([{
            "Store": store_id,
            "DayOfWeek": date.dayofweek + 1,
            "Date": date,
            "Sales": np.nan,  # to be filled after prediction
            "Open": 1,        # assumption: v1 only forecasts days the store is planned to be open
            "Promo": 0,       # assumption: no promo known; caller can extend this later
            "StateHoliday": "0",
            "SchoolHoliday": 0,
        }])


def get_service() -> PredictionService:
    return PredictionService()