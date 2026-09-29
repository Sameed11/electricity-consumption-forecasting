"""Prediction intervals from quantile gradient boosting.

The raw quantile intervals are under-calibrated out of sample (65.7% empirical
coverage against a 90% target, README). This module gives the fit / predict /
score interface, plus a conformalized quantile regression (CQR) hook so the
calibration step from the README's known-limitations list can slot in.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import make_splitter, pinball_loss
from .models import make_quantile_gbm

DEFAULT_QUANTILES = (0.05, 0.5, 0.95)


def fit_quantiles(X_train, y_train, quantiles=DEFAULT_QUANTILES) -> dict:
    """Fit one quantile GBM per quantile. Returns ``{quantile: model}``."""
    return {q: make_quantile_gbm(q).fit(X_train, y_train) for q in quantiles}


def predict_quantiles(models: dict, X_test) -> pd.DataFrame:
    """Predict each quantile in ``models``. Columns are named ``q05`` etc."""
    return pd.DataFrame(
        {f"q{int(q * 100):02d}": model.predict(X_test) for q, model in models.items()}
    )


def interval_metrics(
    y_true, q_low, q_med, q_high, target_coverage: float = 0.9
) -> dict:
    """Coverage, mean width and pinball loss at each quantile."""
    y_true = np.asarray(y_true, float)
    q_low = np.asarray(q_low, float)
    q_med = np.asarray(q_med, float)
    q_high = np.asarray(q_high, float)
    covered = (y_true >= q_low) & (y_true <= q_high)
    return {
        "coverage_pct": float(covered.mean() * 100),
        "target_coverage_pct": target_coverage * 100,
        "mean_width": float((q_high - q_low).mean()),
        "pinball_low": pinball_loss(y_true, q_low, 0.05),
        "pinball_med": pinball_loss(y_true, q_med, 0.5),
        "pinball_high": pinball_loss(y_true, q_high, 0.95),
    }


def conformal_correction(
    cal_low, cal_high, cal_y, alpha: float = 0.1
) -> float:
    """Additive CQR correction that guarantees marginal (1 - alpha) coverage.

    Fit quantile models on a training block, predict on a calibration block, and
    pass those predictions plus the calibration truth here. The returned scalar
    ``q_hat`` widens future intervals to [q_low - q_hat, q_high + q_hat] and
    restores nominal coverage on exchangeable test data.
    """
    cal_y = np.asarray(cal_y, float)
    cal_low = np.asarray(cal_low, float)
    cal_high = np.asarray(cal_high, float)

    scores = np.maximum(cal_low - cal_y, cal_y - cal_high)
    n = len(scores)
    level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(scores, level))


def backtest_quantiles(
    frame: pd.DataFrame,
    splitter=None,
    quantiles=DEFAULT_QUANTILES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rolling-origin quantile forecasts. Returns (per-fold metrics, predictions)."""
    splitter = splitter or make_splitter()
    X = frame.drop(columns="y")
    y = frame["y"]

    rows, all_preds = [], []
    for i, (train_idx, test_idx) in enumerate(splitter.split(frame), 1):
        models = fit_quantiles(X.iloc[train_idx], y.iloc[train_idx], quantiles)
        preds = predict_quantiles(models, X.iloc[test_idx])
        preds.index = frame.index[test_idx]
        preds["y_true"] = y.iloc[test_idx].values
        preds["fold"] = i

        m = interval_metrics(
            preds["y_true"], preds["q05"], preds["q50"], preds["q95"]
        )
        m["fold"] = i
        rows.append(m)
        all_preds.append(preds.reset_index(names="datetime"))

    return pd.DataFrame(rows), pd.concat(all_preds, ignore_index=True)
