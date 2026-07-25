"""
API route definitions for batch prediction.
"""

import logging

from fastapi import APIRouter, HTTPException

from api.schemas import (
    BatchForecastRequest,
    BatchForecastResponse,
    ForecastResult,
    StoreForecastError,
)
from api import state

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok", "model_loaded": state.service is not None}


@router.post("/predict/batch", response_model=BatchForecastResponse)
def predict_batch(request: BatchForecastRequest):
    if state.service is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    predictions = []
    errors = []

    for item in request.requests:
        try:
            preds_df = state.service.predict_store(
                store_id=item.store_id,
                forecast_start=str(item.forecast_start),
                forecast_end=str(item.forecast_end),
            )
            for _, row in preds_df.iterrows():
                predictions.append(ForecastResult(
                    store_id=int(row["Store"]),
                    date=row["Date"].date(),
                    predicted_sales=round(float(row["predicted_sales"]), 2),
                ))
        except Exception as e:
            logger.warning(f"Prediction failed for store {item.store_id}: {e}")
            errors.append(StoreForecastError(store_id=item.store_id, error=str(e)))

    from api.predict_service import MODEL_NAME, MODEL_STAGE
    return BatchForecastResponse(
        predictions=predictions,
        errors=errors,
        model_name=MODEL_NAME,
        model_stage=MODEL_STAGE,
    )