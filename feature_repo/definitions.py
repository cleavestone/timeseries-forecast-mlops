"""
Feast feature repository definitions.

Registers:
- store: the entity (join key) features are keyed on
- store_features_source: the Postgres table build_features() writes into
- store_features_view: the registered FeatureView training/serving code queries
"""

from datetime import timedelta

from feast import Entity, FeatureView, Field
from feast.infra.offline_stores.contrib.postgres_offline_store.postgres_source import PostgreSQLSource
from feast.types import Float32, Int32, String


store = Entity(
    name="store",
    join_keys=["store_id"],
    description="A Rossmann retail store",
)

store_features_source = PostgreSQLSource(
    name="store_features_source",
    query="SELECT * FROM store_features",
    timestamp_field="event_timestamp",
)

store_features_view = FeatureView(
    name="store_features",
    entities=[store],
    ttl=timedelta(days=60),  # matches our FEATURE_WINDOW_DAYS convention from earlier
    schema=[
        Field(name="day_of_week", dtype=Int32),
        Field(name="month", dtype=Int32),
        Field(name="year", dtype=Int32),
        Field(name="is_weekend", dtype=Int32),
        Field(name="is_december", dtype=Int32),
        Field(name="promo", dtype=Int32),
        Field(name="promo2active", dtype=Int32),
        Field(name="state_holiday", dtype=String),
        Field(name="school_holiday", dtype=Int32),
        Field(name="store_type", dtype=String),
        Field(name="assortment", dtype=String),
        Field(name="competition_distance", dtype=Float32),
        Field(name="competition_open_since_month", dtype=Float32),
        Field(name="competition_open_since_year", dtype=Float32),
        Field(name="promo2", dtype=Int32),
        Field(name="promo2_since_week", dtype=Float32),
        Field(name="promo2_since_year", dtype=Float32),
        Field(name="sales_lag_1", dtype=Float32),
        Field(name="sales_lag_7", dtype=Float32),
        Field(name="sales_lag_14", dtype=Float32),
        Field(name="sales_lag_21", dtype=Float32),
        Field(name="sales_lag_28", dtype=Float32),
        Field(name="sales_rolling_mean_7", dtype=Float32),
        Field(name="sales_rolling_std_7", dtype=Float32),
        Field(name="sales_rolling_mean_28", dtype=Float32),
        Field(name="sales_rolling_std_28", dtype=Float32),
        Field(name="sales", dtype=Float32),
        Field(name="sales_transformed", dtype=Float32),
    ],
    source=store_features_source,
    online=True,
)