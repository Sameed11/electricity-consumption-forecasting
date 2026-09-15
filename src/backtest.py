"""Rolling-origin backtesting and forecast metrics.

A single train/test split gives one sample and no sense of variance. Every result
in this project is averaged over several expanding-window folds, each separated
from its training data by a gap equal to the forecast horizon.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

N_SPLITS = 4
TEST_HOURS = 24 * 90
GAP_HOURS = 24


def make_splitter(
    n_splits: int = N_SPLITS,
    test_hours: int = TEST_HOURS,
    gap_hours: int = GAP_HOURS,
) -> TimeSeriesSplit:
    return TimeSeriesSplit(n_splits=n_splits, test_size=test_hours, gap=gap_hours)


def metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE_%": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100),
    }


def pinball_loss(y_true, y_pred, quantile: float) -> float:
    """Proper scoring rule for a single quantile forecast."""
    delta = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.mean(np.maximum(quantile * delta, (quantile - 1) * delta)))


def describe_folds(frame: pd.DataFrame, splitter: TimeSeriesSplit) -> pd.DataFrame:
    rows = []
    for i, (train_idx, test_idx) in enumerate(splitter.split(frame), 1):
        rows.append({
            "fold": i,
            "train_hours": len(train_idx),
            "train_end": frame.index[train_idx[-1]].date(),
            "test_start": frame.index[test_idx[0]].date(),
            "test_end": frame.index[test_idx[-1]].date(),
            "test_hours": len(test_idx),
        })
    return pd.DataFrame(rows)


def backtest_model(
    frame: pd.DataFrame,
    model_fn,
    name: str,
    splitter: TimeSeriesSplit | None = None,
    collect_predictions: bool = False,
):
    """Refit ``model_fn()`` from scratch on each expanding window and score the fold.

    Returns ``(results, importances, predictions)``. Importances are averaged over
    folds when the estimator exposes them; predictions are None unless requested.
    """
    splitter = splitter or make_splitter()
    X = frame.drop(columns="y")
    y = frame["y"]

    rows, predictions, importances = [], [], []

    for i, (train_idx, test_idx) in enumerate(splitter.split(frame), 1):
        model = model_fn()
        started = time.time()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(X.iloc[test_idx])

        row = metrics(y.iloc[test_idx], preds)
        row.update({"model": name, "fold": i, "fit_seconds": round(time.time() - started, 1)})
        rows.append(row)

        if collect_predictions:
            predictions.append(pd.DataFrame({
                "datetime": frame.index[test_idx],
                "y_true": y.iloc[test_idx].values,
                "y_pred": preds,
                "model": name,
                "fold": i,
            }))

        if hasattr(model, "feature_importances_"):
            importances.append(pd.Series(model.feature_importances_, index=X.columns))
        elif hasattr(model, "steps") and hasattr(model.steps[-1][1], "coef_"):
            importances.append(pd.Series(np.abs(model.steps[-1][1].coef_), index=X.columns))

    results = pd.DataFrame(rows)
    mean_importance = (
        pd.concat(importances, axis=1).mean(axis=1).sort_values(ascending=False)
        if importances else None
    )
    all_predictions = pd.concat(predictions, ignore_index=True) if collect_predictions else None
    return results, mean_importance, all_predictions


def backtest_baseline(
    frame: pd.DataFrame, column: str, name: str, splitter: TimeSeriesSplit | None = None
) -> pd.DataFrame:
    """Score a naive baseline, which is simply an existing lag column used as-is."""
    splitter = splitter or make_splitter()
    rows = []
    for i, (_, test_idx) in enumerate(splitter.split(frame), 1):
        row = metrics(frame["y"].iloc[test_idx], frame[column].iloc[test_idx])
        row.update({"model": name, "fold": i, "fit_seconds": 0.0})
        rows.append(row)
    return pd.DataFrame(rows)


def summarise(results: pd.DataFrame, baseline_name: str | None = None) -> pd.DataFrame:
    """Collapse per-fold results into a ranked table, optionally with lift vs a baseline."""
    summary = (
        results.groupby("model")
        .agg(
            MAE_mean=("MAE", "mean"),
            MAE_std=("MAE", "std"),
            RMSE_mean=("RMSE", "mean"),
            MAPE_mean=("MAPE_%", "mean"),
            fit_s=("fit_seconds", "mean"),
        )
        .sort_values("MAE_mean")
        .round(2)
    )
    if baseline_name is not None:
        matches = [m for m in summary.index if baseline_name in m]
        if matches:
            base = summary.loc[matches[0], "MAE_mean"]
            summary["vs_baseline_%"] = ((base - summary["MAE_mean"]) / base * 100).round(1)
    return summary
