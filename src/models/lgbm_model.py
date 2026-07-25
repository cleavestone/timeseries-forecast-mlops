"""
LightGBM model training logic.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd


# Columns that must never be fed to the model — identifiers, raw target,
# or leakage-prone fields already excluded in build_features.py.
NON_FEATURE_COLS = ["Date", "Store", "Sales", "Sales_transformed", "Open"]

CATEGORICAL_COLS = ["StoreType", "Assortment", "StateHoliday"]


def get_feature_columns(df: pd.DataFrame, target_col: str = "Sales") -> list:
    exclude = set(NON_FEATURE_COLS)
    return [c for c in df.columns if c not in exclude]


def prepare_lgbm_data(df: pd.DataFrame, feature_cols: list):
    X = df[feature_cols].copy()
    for col in CATEGORICAL_COLS:
        if col in X.columns:
            X[col] = X[col].astype("category")
    return X


def train_lgbm(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list,
    target_col: str = "Sales_transformed",
    params: dict = None,
) -> lgb.Booster:
    default_params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbosity": -1,
        "seed": 42,
    }
    if params:
        default_params.update(params)

    X_train = prepare_lgbm_data(train_df, feature_cols)
    y_train = train_df[target_col]
    X_val = prepare_lgbm_data(val_df, feature_cols)
    y_val = val_df[target_col]

    train_set = lgb.Dataset(X_train, label=y_train, categorical_feature=CATEGORICAL_COLS)
    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set, categorical_feature=CATEGORICAL_COLS)

    model = lgb.train(
        default_params,
        train_set,
        num_boost_round=1000,
        valid_sets=[train_set, val_set],
        valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=50)],
    )
    return model


def predict_lgbm(model: lgb.Booster, df: pd.DataFrame, feature_cols: list) -> np.ndarray:
    X = prepare_lgbm_data(df, feature_cols)
    preds_log = model.predict(X, num_iteration=model.best_iteration)
    return np.expm1(preds_log)  # invert log1p transform back to raw sales scale