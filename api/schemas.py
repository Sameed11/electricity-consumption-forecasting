"""Pydantic request and response models for the forecast API."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    model_name: str
    model_version: str
    model_flavor: str
    last_observed: datetime | None = Field(
        default=None,
        description="Latest timestamp available in the historic series.",
    )


class ForecastPoint(BaseModel):
    datetime: datetime
    hour_ahead: int = Field(..., ge=1, le=168, description="Hours from the anchor timestamp.")
    prediction_mwh: float


class ForecastResponse(BaseModel):
    model_name: str
    model_version: str
    anchor: datetime = Field(
        ...,
        description=(
            "The last known timestamp used to build features. The forecast "
            "covers anchor+1h through anchor+N_hours."
        ),
    )
    horizon_hours: int
    forecast: list[ForecastPoint]


class PredictRequest(BaseModel):
    """POST a single feature row (61 columns) for one prediction."""

    features: dict[str, float]


class PredictResponse(BaseModel):
    model_name: str
    model_version: str
    prediction_mwh: float
