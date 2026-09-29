"""FastAPI service for the day-ahead electricity forecaster.

Loads the champion model by alias from the local MLflow registry and exposes:

    GET  /health              service + model version + last observed hour
    GET  /forecast?hours=24   next N hours forecast, built from the historic tail
    POST /predict             single-row prediction from raw features

Run locally from the repo root:

    conda activate ML
    uvicorn api.main:app --reload --port 8000

Interactive docs at http://localhost:8000/docs.
"""

from __future__ import annotations

import os
import sys
from datetime import timedelta
from functools import lru_cache
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException, Query

# --- Repo root + MLflow setup ---------------------------------------------

_here = Path(__file__).resolve()
REPO_ROOT = None
for _p in [_here, *_here.parents]:
    if (_p / "src").is_dir() and (_p / "data").is_dir():
        REPO_ROOT = _p
        break
if REPO_ROOT is None:
    raise RuntimeError("Could not locate repo root (need src/ and data/).")

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# TRACKING_URI is read from env when running in a container; falls back to the
# local SQLite store when running from a dev machine.
TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    f"sqlite:///{(REPO_ROOT / 'mlflow_store' / 'mlflow.db').as_posix()}",
)
REGISTERED_NAME = os.getenv("MODEL_NAME", "electricity-forecaster")
ALIAS = os.getenv("MODEL_ALIAS", "champion")
DATA_PATH = REPO_ROOT / "data" / "Electricity Consumption 2015-2020.csv"

mlflow.set_tracking_uri(TRACKING_URI)


# --- Loaders (cached) -----------------------------------------------------

@lru_cache(maxsize=1)
def _load_champion() -> tuple[object, str]:
    """Load the champion model and return (model, version_string).

    lru_cache means the model is loaded once at first request. To hot-swap
    a new champion, restart the service (or expose /reload later).
    """
    client = mlflow.MlflowClient()
    version_obj = client.get_model_version_by_alias(REGISTERED_NAME, ALIAS)
    model = mlflow.sklearn.load_model(f"models:/{REGISTERED_NAME}@{ALIAS}")
    return model, str(version_obj.version)


@lru_cache(maxsize=1)
def _load_series() -> pd.Series:
    from src.data import load_and_clean

    series, _ = load_and_clean(str(DATA_PATH))
    return series


# --- Feature building for the forecast horizon ----------------------------

def _future_frame(hours: int) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Build the feature frame covering the next `hours` timestamps.

    Extends the historic series by `hours` NaN-filled slots, calls the same
    make_features used in training, and returns the tail rows. All lag / roll
    columns still come from the real history — the appended slots exist only
    so make_features has rows to produce for those timestamps.
    """
    from src.features import make_features

    series = _load_series()
    anchor = series.index[-1]
    future_index = pd.date_range(
        anchor + timedelta(hours=1), periods=hours, freq="h"
    )
    # Placeholder value so make_features doesn't drop these rows on the y NaN
    # check. The value never enters a lag window because every feature is
    # shifted by at least 24 hours.
    extended = pd.concat(
        [series, pd.Series(series.iloc[-1], index=future_index, name=series.name)]
    )
    frame = make_features(extended, min_lag=24, country="TR")
    forecast_rows = frame.loc[future_index].drop(columns="y")
    return forecast_rows, anchor


# --- App ------------------------------------------------------------------

app = FastAPI(
    title="Electricity Load Forecaster",
    description=(
        "Day-ahead hourly electricity load forecast for the Turkish grid, "
        "served from the MLflow model registry."
    ),
    version="0.1.0",
)


@app.get("/health")
def health():
    from api.schemas import HealthResponse

    try:
        model, version = _load_champion()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Model load failed: {e}")

    series = _load_series()
    return HealthResponse(
        model_name=REGISTERED_NAME,
        model_version=version,
        model_flavor=type(model).__name__,
        last_observed=series.index[-1].to_pydatetime(),
    )


@app.get("/forecast")
def forecast(
    hours: int = Query(
        24,
        ge=1,
        le=168,
        description=(
            "How many hours ahead to forecast (1–168). "
            "The training horizon was day-ahead (24h); anything longer than "
            "that is served but relies on lag windows that reach further back."
        ),
    ),
):
    from api.schemas import ForecastPoint, ForecastResponse

    try:
        model, version = _load_champion()
        X, anchor = _future_frame(hours)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    preds = model.predict(X)
    points = [
        ForecastPoint(
            datetime=ts.to_pydatetime(),
            hour_ahead=i + 1,
            prediction_mwh=float(p),
        )
        for i, (ts, p) in enumerate(zip(X.index, preds))
    ]
    return ForecastResponse(
        model_name=REGISTERED_NAME,
        model_version=version,
        anchor=anchor.to_pydatetime(),
        horizon_hours=hours,
        forecast=points,
    )


@app.post("/predict")
def predict(payload: dict):
    from api.schemas import PredictRequest, PredictResponse

    try:
        req = PredictRequest(**payload)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

    try:
        model, version = _load_champion()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Model load failed: {e}")

    row = pd.DataFrame([req.features])
    try:
        pred = float(model.predict(row)[0])
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Model prediction failed: {e}. "
                "Feature row must include every column the model was trained on."
            ),
        )

    return PredictResponse(
        model_name=REGISTERED_NAME,
        model_version=version,
        prediction_mwh=pred,
    )
