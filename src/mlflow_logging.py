"""Thin MLflow helpers for the electricity forecasting project.

Every backtest is logged as::

    parent_run   (model-level: mean metrics, params, tags, artifacts)
    ├── fold_1   (nested: per-fold MAE, RMSE, MAPE)
    ├── fold_2
    ├── fold_3
    └── fold_4

The store is local SQLite so the model registry works (a plain ``file://``
store cannot register models). Artifacts live next to the DB in the same
folder, so the whole MLflow state is one directory Sam can zip and share.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd

# Fold-results column names produced by src.backtest — logged as metrics when
# present. The frame actually carries "fit_seconds" (not "fit_s"), so list that
# here and rename to a shorter key at log time via _METRIC_RENAME.
_METRIC_COLS = ["MAE", "RMSE", "MAPE_%", "fit_seconds"]

# MLflow only accepts [alphanumerics _ - . space : /] in metric names.
# Rename anything that trips that rule, and shorten fit_seconds -> fit_s so the
# dashboard column stays compact.
_METRIC_RENAME = {"MAPE_%": "MAPE_pct", "fit_seconds": "fit_s"}


def _metric_key(col: str) -> str:
    return _METRIC_RENAME.get(col, col)


def setup_tracking(
    experiment_name: str,
    store_dir: str | Path = "mlflow_store",
) -> str:
    """Point MLflow at a local SQLite store and set the active experiment.

    Returns the tracking URI actually set — useful for launching ``mlflow ui``.
    """
    store_dir = Path(store_dir).resolve()
    store_dir.mkdir(parents=True, exist_ok=True)

    # .as_posix() so Windows backslashes don't confuse SQLite's URI parser.
    tracking_uri = f"sqlite:///{(store_dir / 'mlflow.db').as_posix()}"
    artifact_location = (store_dir / "artifacts").as_uri()

    mlflow.set_tracking_uri(tracking_uri)

    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(experiment_name, artifact_location=artifact_location)
    mlflow.set_experiment(experiment_name)
    return tracking_uri


def log_backtest(
    model_name: str,
    fold_results: pd.DataFrame,
    params: dict[str, Any] | None = None,
    importance: pd.DataFrame | None = None,
    predictions: pd.DataFrame | None = None,
    tags: dict[str, str] | None = None,
) -> str:
    """Log one model's backtest as a parent run with one nested run per fold.

    ``fold_results`` is expected to be the frame returned by
    :func:`src.backtest.backtest_model` (or the baseline / LSTM equivalents).

    Returns the parent run id.
    """
    with mlflow.start_run(run_name=model_name) as parent:
        if params:
            mlflow.log_params(params)
        if tags:
            mlflow.set_tags(tags)

        # Model-level aggregates on the parent.
        for col in _METRIC_COLS:
            if col in fold_results.columns:
                key = _metric_key(col)
                mlflow.log_metric(f"{key}_mean", float(fold_results[col].mean()))
                if len(fold_results) > 1:
                    mlflow.log_metric(f"{key}_std", float(fold_results[col].std()))

        # One nested run per fold — powers the per-fold metric chart in the UI.
        for _, row in fold_results.iterrows():
            with mlflow.start_run(run_name=f"fold_{int(row['fold'])}", nested=True):
                mlflow.log_param("fold", int(row["fold"]))
                for col in _METRIC_COLS:
                    if col in row and pd.notna(row[col]):
                        mlflow.log_metric(_metric_key(col), float(row[col]))

        # Artifacts through a temp dir so nothing leaks into the repo.
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            if importance is not None:
                p = td_path / "importance.csv"
                importance.to_csv(p)
                mlflow.log_artifact(str(p), artifact_path="importance")
            if predictions is not None:
                p = td_path / "predictions.csv"
                predictions.to_csv(p, index=False)
                mlflow.log_artifact(str(p), artifact_path="predictions")

        return parent.info.run_id


def register_production_model(
    model,
    input_example: pd.DataFrame,
    registered_name: str = "electricity-forecaster",
    alias: str = "champion",
) -> str:
    """Log an sklearn-compatible model to the active run and set an alias on it.

    Works with plain sklearn estimators, Pipelines and LightGBM regressors alike
    because we go through ``mlflow.sklearn`` rather than ``mlflow.lightgbm``.
    Requires an active MLflow run.
    """
    import mlflow.sklearn

    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        input_example=input_example,
        registered_model_name=registered_name,
    )

    client = mlflow.MlflowClient()
    versions = client.search_model_versions(f"name='{registered_name}'")
    latest = max(int(v.version) for v in versions)
    client.set_registered_model_alias(registered_name, alias, str(latest))
    return str(latest)


def load_registered(
    name: str = "electricity-forecaster", alias: str = "champion"
):
    """Load a registered model by alias. Returns the sklearn estimator itself."""
    import mlflow.sklearn

    return mlflow.sklearn.load_model(f"models:/{name}@{alias}")
