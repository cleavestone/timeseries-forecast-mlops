"""
N-BEATS global deep learning model, via neuralforecast.

Unlike LightGBM, this model consumes raw (unique_id, ds, y) long-format
data directly rather than our engineered feature table — it learns
temporal patterns (trend, seasonality) internally instead of relying on
hand-crafted lag/rolling features.
"""

import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS


def prepare_long_format(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw Rossmann train data into neuralforecast's expected long format.
    Filters to Open==1 only, consistent with the LightGBM training set.
    """
    df = train_df[train_df["Open"] == 1].copy()
    df = df[["Store", "Date", "Sales"]].rename(
        columns={"Store": "unique_id", "Date": "ds", "Sales": "y"}
    )
    df["unique_id"] = df["unique_id"].astype(str)
    df = df.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    return df


def train_nbeats(
    long_df: pd.DataFrame,
    horizon: int,
    input_size: int = None,
    max_steps: int = 300,
) -> NeuralForecast:
    if input_size is None:
        input_size = horizon * 4

    model = NBEATS(
        h=horizon,
        input_size=input_size,
        max_steps=max_steps,
        random_seed=42,
    )
    nf = NeuralForecast(models=[model], freq="D")
    nf.fit(df=long_df)
    return nf


def predict_nbeats(nf: NeuralForecast) -> pd.DataFrame:
    """Returns a dataframe with unique_id, ds, NBEATS (predicted column)."""
    return nf.predict()