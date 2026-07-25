"""
Pydantic request/response schemas for the batch prediction API.
"""

from datetime import date
from pydantic import BaseModel, Field, field_validator


class ForecastRequestItem(BaseModel):
    store_id: int = Field(..., description="Store ID to forecast for")
    forecast_start: date = Field(..., description="First date to forecast (inclusive)")
    forecast_end: date = Field(..., description="Last date to forecast (inclusive)")

    @field_validator("forecast_end")
    @classmethod
    def end_after_start(cls, v, info):
        start = info.data.get("forecast_start")
        if start and v < start:
            raise ValueError("forecast_end must be on or after forecast_start")
        return v


class BatchForecastRequest(BaseModel):
    requests: list[ForecastRequestItem] = Field(..., min_length=1, max_length=100)


class ForecastResult(BaseModel):
    store_id: int
    date: date
    predicted_sales: float


class StoreForecastError(BaseModel):
    store_id: int
    error: str


class BatchForecastResponse(BaseModel):
    predictions: list[ForecastResult]
    errors: list[StoreForecastError] = []
    model_name: str
    model_stage: str