"""FastAPI app that serves the model."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from ml_service import __version__
from ml_service.model import ModelBundle, load

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/model.joblib"))
MAX_BATCH = 256


class PredictRequest(BaseModel):
    rows: list[list[float]] = Field(
        ...,
        min_length=1,
        max_length=MAX_BATCH,
        description="One list of 13 feature values per sample, in the order from /model-info.",
    )


class Prediction(BaseModel):
    label: str
    confidence: float
    probabilities: dict[str, float]


class PredictResponse(BaseModel):
    predictions: list[Prediction]
    model_version: str


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.model = load(MODEL_PATH)
    yield


app = FastAPI(title="ML Service Template", version=__version__, lifespan=lifespan)


def get_model(request: Request) -> ModelBundle:
    model: ModelBundle = request.app.state.model
    return model


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.get("/model-info")
def model_info(request: Request) -> dict[str, object]:
    model = get_model(request)
    return {
        "feature_names": model.feature_names,
        "class_names": model.class_names,
        "metrics": model.metrics,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest, request: Request) -> PredictResponse:
    model = get_model(request)
    n = len(model.feature_names)
    for i, row in enumerate(body.rows):
        if len(row) != n:
            raise HTTPException(422, f"row {i} has {len(row)} values, expected {n}")
    preds = [Prediction.model_validate(p) for p in model.predict(body.rows)]
    return PredictResponse(predictions=preds, model_version=__version__)
