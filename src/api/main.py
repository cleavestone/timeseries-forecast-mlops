"""
FastAPI app entrypoint. Wires together instrumentation, startup logic,
and route registration. Route logic itself lives in api/routes.py.
"""

import logging

from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

from api import state
from api.routes import router
from api.predict_service import get_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Rossmann Sales Forecasting API",
    description="Batch demand forecasting service backed by a LightGBM model "
                 "registered in MLflow.",
    version="0.1.0",
)

Instrumentator().instrument(app).expose(app)
app.include_router(router)


@app.on_event("startup")
def load_model():
    logger.info("Loading production model from MLflow registry...")
    state.service = get_service()
    logger.info("Model loaded successfully.")