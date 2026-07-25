"""
Shared feature engineering module.

Used identically by both training (src/train.py) and inference
(src/api/predict_service.py) to guarantee train/serve parity.
"""

import numpy as np
import pandas as pd
import yaml


def load_params(params_path: str = "params.yaml") -> dict:
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


def clean_state_holiday(df: pd.DataFrame) -> pd.DataFrame:
    """StateHoliday arrives with mixed int/str '0' values — normalize to string."""
    df = df.copy()
    df["StateHoliday"] = df["StateHoliday"].astype(str)
    return df


def reindex_to_full_calendar(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    """
    Reindex each store's series to a full continuous daily date range.
    This must happen BEFORE any lag/rolling computation — otherwise lag_7
    computed on an Open==1-filtered row sequence does not mean 'same weekday
    last week' for stores that close on Sundays or have closure gaps.
    """
    def _reindex_store(group: pd.DataFrame) -> pd.DataFrame:
        store_id = group["Store"].iloc[0]
        full_range = pd.date_range(group[date_col].min(), group[date_col].max(), freq="D")
        group = group.set_index(date_col).reindex(full_range)
        group["Store"] = store_id
        group.index.name = date_col
        return group.reset_index()

    return df.groupby("Store", group_keys=False)[df.columns].apply(_reindex_store)


def add_calendar_features(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    df = df.copy()
    df["DayOfWeek"] = df[date_col].dt.dayofweek + 1  # 1=Mon..7=Sun, matches raw Rossmann encoding
    df["Month"] = df[date_col].dt.month
    df["Year"] = df[date_col].dt.year
    df["IsWeekend"] = df["DayOfWeek"].isin([6, 7]).astype(int)
    df["IsDecember"] = (df["Month"] == 12).astype(int)
    return df


def add_promo2_active(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    """
    Vectorized version of the row-wise .apply used in EDA.
    Promo2Active = 1 if Promo2==1 AND current month is in PromoInterval.
    """
    df = df.copy()
    month_map = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                 7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
    df["MonthStr"] = df[date_col].dt.month.map(month_map)

    # Handle 'Sept' vs 'Sep' inconsistency in raw PromoInterval values explicitly,
    # by splitting on commas and checking exact membership (not substring matching).
    def _interval_set(interval):
        if not isinstance(interval, str):
            return set()
        parts = [p.strip() for p in interval.split(",")]
        parts = ["Sep" if p == "Sept" else p for p in parts]
        return set(parts)

    interval_sets = df["PromoInterval"].apply(_interval_set)
    is_active_month = pd.Series(
        [month in s for month, s in zip(df["MonthStr"], interval_sets)],
        index=df.index,
    )
    df["Promo2Active"] = ((df["Promo2"] == 1) & is_active_month).astype(int)

    df = df.drop(columns=["MonthStr"])
    return df


def add_lag_and_rolling_features(
    df: pd.DataFrame,
    target_col: str,
    lags: list,
    rolling_windows: list,
    rolling_stats: list,
    date_col: str = "Date",
) -> pd.DataFrame:
    """
    Computed per-store, on the full calendar-reindexed series (gaps included as NaN).
    Must be called AFTER reindex_to_full_calendar.
    """
    df = df.sort_values(["Store", date_col]).copy()

    for lag in lags:
        df[f"{target_col}_lag_{lag}"] = df.groupby("Store")[target_col].shift(lag)

    for window in rolling_windows:
        grouped = df.groupby("Store")[target_col]
        shifted = grouped.shift(1)  # avoid leaking current day into its own rolling stat
        for stat in rolling_stats:
            col_name = f"{target_col}_rolling_{stat}_{window}"
            if stat == "mean":
                df[col_name] = shifted.groupby(df["Store"]).rolling(window).mean().reset_index(level=0, drop=True)
            elif stat == "std":
                df[col_name] = shifted.groupby(df["Store"]).rolling(window).std().reset_index(level=0, drop=True)

    return df


def apply_sentinel_imputation(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    df = df.copy()
    f = params["features"]
    df["CompetitionDistance"] = df["CompetitionDistance"].fillna(f["competition_distance_sentinel"])
    df["CompetitionOpenSinceMonth"] = df["CompetitionOpenSinceMonth"].fillna(f["competition_since_sentinel"])
    df["CompetitionOpenSinceYear"] = df["CompetitionOpenSinceYear"].fillna(f["competition_since_sentinel"])
    df["Promo2SinceWeek"] = df["Promo2SinceWeek"].fillna(f["promo2_since_sentinel"])
    df["Promo2SinceYear"] = df["Promo2SinceYear"].fillna(f["promo2_since_sentinel"])
    return df


def build_features(
    train_df: pd.DataFrame,
    store_df: pd.DataFrame,
    params: dict,
    filter_open_only: bool = True,
) -> pd.DataFrame:
    """
    Main entrypoint. Called identically by training and inference.

    Args:
        train_df: raw sales rows (Store, Date, Sales, Open, Promo, etc.)
        store_df: raw store metadata
        params: loaded from params.yaml
        filter_open_only: if True, drops Open==0 rows AFTER feature computation.
                           Training should pass True. Inference building history
                           context should pass False, then filter downstream itself
                           if needed.
    """
    target_col = params["train"]["target_col"]
    f = params["features"]

    df = train_df.merge(store_df, on="Store", how="left")
    df = clean_state_holiday(df)
    df = reindex_to_full_calendar(df)

    # Open/Promo/etc become NaN on reindexed gap rows — fill sensibly.
    # A gap day (store closed for renovation) should behave like a closed day.
    df["Open"] = df["Open"].fillna(0).astype(int)
    for col in ["Promo", "SchoolHoliday"]:
        df[col] = df[col].fillna(0).astype(int)
    df["StateHoliday"] = df["StateHoliday"].fillna("0")
    df[target_col] = df[target_col].fillna(0)

    df = add_calendar_features(df)
    df = add_promo2_active(df)
    df = add_lag_and_rolling_features(
        df,
        target_col=target_col,
        lags=f["lags"],
        rolling_windows=f["rolling_windows"],
        rolling_stats=f["rolling_stats"],
    )
    df = apply_sentinel_imputation(df, params)

    # target transform
    if f["target_transform"] == "log1p":
        df[f"{target_col}_transformed"] = np.log1p(df[target_col])

    # Customers is not known ahead of time for a future date — would leak target-adjacent
    # information if left in. PromoInterval was only needed to derive Promo2Active above.
    df = df.drop(columns=["Customers", "PromoInterval"], errors="ignore")

    if filter_open_only:
        df = df[df["Open"] == 1].reset_index(drop=True)

    return df