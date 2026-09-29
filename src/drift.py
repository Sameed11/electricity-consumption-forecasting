"""Feature drift monitoring for the day-ahead load forecaster.

Uses Evidently to compute normalized Wasserstein distance per numeric feature:

    distance(ref, current) / std(ref)

Wasserstein is preferred to a K-S p-value here because the reference window
(a year of hourly data) dwarfs the current window (a week), and any K-S test
with that sample-size gap flags every feature. Wasserstein is a magnitude, not
a hypothesis test, so it stays interpretable across window sizes.

The intended use is the notebook demo: replay Jan-Jun 2020 in weekly windows,
watch the drift signal jump around mid-March, and show that a drift-triggered
retrain recovers a chunk of the shock-induced error.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from evidently import Report
    from evidently.metrics import ValueDrift

    HAS_EVIDENTLY = True
except ImportError:  # pragma: no cover
    HAS_EVIDENTLY = False


# Features watched by default. Kept small on purpose — drift on every one of the
# 61 columns is noisy, and the lag/rolling family already captures the demand
# regime. Add columns via the `features` argument for a wider view.
DEFAULT_WATCH = [
    "lag_24", "lag_168", "lag_336",
    "roll_mean_24", "roll_mean_168", "roll_std_168",
    "diff_24_168",
]

# A per-feature Wasserstein distance in units of std_ref above this cutoff
# counts as drifted for the ``drifted`` list. 0.10 is Evidently's default.
PER_FEATURE_THRESHOLD = 0.10

# Any single feature drifting this far (in std_ref units) fires the retrain
# alert, regardless of how many other features drifted. Chosen from the demo
# sweep: pre-March windows sit at 1-8, post-lockdown at 15-30. 10 separates
# the two regimes with one false positive (Feb 12: 17.6) in six months.
MAGNITUDE_THRESHOLD = 10.0


def _wasserstein(reference: pd.Series, current: pd.Series) -> float:
    """Raw Wasserstein distance from Evidently for a single column."""
    ref_df = reference.to_frame(name="v")
    cur_df = current.to_frame(name="v")
    snap = Report([ValueDrift(column="v", method="wasserstein")]).run(
        reference_data=ref_df, current_data=cur_df
    )
    return float(snap.dict()["metrics"][0]["value"])


def drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    features: Iterable[str] | None = None,
    per_feature_threshold: float = PER_FEATURE_THRESHOLD,
    magnitude_threshold: float = MAGNITUDE_THRESHOLD,
) -> dict:
    """Score each watched feature by normalized Wasserstein distance.

    Alerting policy: fire when the largest single-feature distance crosses
    ``magnitude_threshold``. Share-of-features-drifted is still reported so
    the notebook can visualise it, but the alert is triggered by magnitude
    because in this dataset one strongly-drifted feature is a clean signal
    while the share moves slowly and lags the regime change.

    Returns
    -------
    dict with keys::

        by_feature  {name: distance / std_ref}   — 0 means identical
        drifted     [names above per_feature_threshold]
        n_drifted   count of drifted features
        n_total     features actually scored
        share       n_drifted / n_total
        max_distance  worst single-feature distance
        alert       max_distance >= magnitude_threshold
    """
    if not HAS_EVIDENTLY:
        raise RuntimeError("evidently is not installed — pip install evidently")

    cols = [
        c for c in (features or DEFAULT_WATCH)
        if c in reference.columns and c in current.columns
    ]
    if not cols:
        raise ValueError("No watched features found in both frames")

    # Evidently's Wasserstein value is *already* normalised: it returns the
    # raw distance divided by the reference standard deviation. So the value
    # is directly comparable across features of different scales, no manual
    # normalisation needed here.
    by_feature: dict[str, float] = {
        c: _wasserstein(reference[c], current[c]) for c in cols
    }

    drifted = [c for c, d in by_feature.items() if d >= per_feature_threshold]
    share = len(drifted) / len(cols)
    max_distance = max(by_feature.values())
    return {
        "by_feature": by_feature,
        "drifted": drifted,
        "n_drifted": len(drifted),
        "n_total": len(cols),
        "share": share,
        "max_distance": max_distance,
        "alert": max_distance >= magnitude_threshold,
        "magnitude_threshold": magnitude_threshold,
        "per_feature_threshold": per_feature_threshold,
    }


def sweep_drift(
    frame: pd.DataFrame,
    replay_start: str | pd.Timestamp,
    replay_end: str | pd.Timestamp,
    window_days: int = 7,
    features: Iterable[str] | None = None,
    per_feature_threshold: float = PER_FEATURE_THRESHOLD,
    magnitude_threshold: float = MAGNITUDE_THRESHOLD,
    reference_mode: str = "same_period_last_year",
) -> pd.DataFrame:
    """Weekly rolling drift sweep with a season-matched reference.

    ``reference_mode``
        ``same_period_last_year`` — each window is compared to the equivalent
            calendar window one year earlier. This controls for seasonality
            so the signal reflects regime change, not winter-vs-summer.
        ``rolling_year`` — each window is compared to the trailing 365 days.
            Useful when no full year prior exists.

    Returns one row per window with drift share, max per-feature distance and
    the alert flag.
    """
    replay_start = pd.Timestamp(replay_start)
    replay_end = pd.Timestamp(replay_end)
    step = timedelta(days=window_days)

    rows = []
    win_start = replay_start
    while win_start < replay_end:
        win_end = min(win_start + step, replay_end)
        current = frame.loc[win_start:win_end]
        if len(current) < 24:
            win_start = win_end
            continue

        if reference_mode == "same_period_last_year":
            ref_start = win_start - pd.DateOffset(years=1)
            ref_end = win_end - pd.DateOffset(years=1)
            reference = frame.loc[ref_start:ref_end]
        elif reference_mode == "rolling_year":
            ref_end = win_start
            ref_start = ref_end - pd.DateOffset(years=1)
            reference = frame.loc[ref_start:ref_end]
        else:
            raise ValueError(f"unknown reference_mode: {reference_mode}")

        if len(reference) < 24:
            win_start = win_end
            continue

        report = drift_report(
            reference, current, features,
            per_feature_threshold, magnitude_threshold,
        )
        rows.append({
            "window_start": win_start,
            "window_end": win_end,
            "drift_share": report["share"],
            "max_distance": report["max_distance"],
            "n_drifted": report["n_drifted"],
            "alert": report["alert"],
            "top_drifted": ", ".join(report["drifted"][:3]),
        })
        win_start = win_end

    return pd.DataFrame(rows)


def _season_matched_reference(frame: pd.DataFrame, win_start, win_end) -> pd.DataFrame:
    """The same calendar window one year earlier."""
    ref_start = win_start - pd.DateOffset(years=1)
    ref_end = win_end - pd.DateOffset(years=1)
    return frame.loc[ref_start:ref_end]


def replay_with_alerts(
    frame: pd.DataFrame,
    model_fn,
    replay_start: str | pd.Timestamp,
    replay_end: str | pd.Timestamp,
    window_days: int = 7,
    features: Iterable[str] | None = None,
    per_feature_threshold: float = PER_FEATURE_THRESHOLD,
    magnitude_threshold: float = MAGNITUDE_THRESHOLD,
) -> pd.DataFrame:
    """Per-window drift signal *and* forecast performance during the replay.

    Trains once on everything up to ``replay_start`` and evaluates on each
    replay window without retraining, so MAE shows what a frozen model does
    while the drift signal moves. Reference for drift is the same calendar
    window one year earlier — see sweep_drift for the rationale.
    """
    replay_start = pd.Timestamp(replay_start)
    replay_end = pd.Timestamp(replay_end)

    train = frame.loc[:replay_start]
    X_tr, y_tr = train.drop(columns="y"), train["y"]
    model = model_fn().fit(X_tr, y_tr)

    step = timedelta(days=window_days)
    rows = []
    win_start = replay_start
    while win_start < replay_end:
        win_end = min(win_start + step, replay_end)
        current = frame.loc[win_start:win_end]
        if len(current) < 24:
            win_start = win_end
            continue
        reference = _season_matched_reference(frame, win_start, win_end)
        if len(reference) < 24:
            win_start = win_end
            continue

        X_cur = current.drop(columns="y")
        y_cur = current["y"].values
        y_hat = model.predict(X_cur)

        report = drift_report(
            reference, current, features,
            per_feature_threshold, magnitude_threshold,
        )
        rows.append({
            "window_start": win_start,
            "window_end": win_end,
            "drift_share": report["share"],
            "max_distance": report["max_distance"],
            "alert": report["alert"],
            "MAE": float(np.mean(np.abs(y_hat - y_cur))),
            "MAPE_pct": 100 * float(np.mean(np.abs((y_hat - y_cur) / y_cur))),
            "n_hours": len(current),
        })
        win_start = win_end

    return pd.DataFrame(rows)


def drift_triggered_replay(
    frame: pd.DataFrame,
    model_fn,
    replay_start: str | pd.Timestamp,
    replay_end: str | pd.Timestamp,
    window_days: int = 7,
    features: Iterable[str] | None = None,
    per_feature_threshold: float = PER_FEATURE_THRESHOLD,
    magnitude_threshold: float = MAGNITUDE_THRESHOLD,
) -> pd.DataFrame:
    """Same as replay_with_alerts, but retrain whenever an alert fires.

    Policy: check drift on the incoming window; if the alert flag is set,
    refit the model on all data prior to the window before predicting.
    Compare MAE against ``replay_with_alerts`` to quantify the value of
    drift-triggered retraining.
    """
    replay_start = pd.Timestamp(replay_start)
    replay_end = pd.Timestamp(replay_end)

    train = frame.loc[:replay_start]
    X_tr, y_tr = train.drop(columns="y"), train["y"]
    model = model_fn().fit(X_tr, y_tr)

    step = timedelta(days=window_days)
    rows = []
    win_start = replay_start
    while win_start < replay_end:
        win_end = min(win_start + step, replay_end)
        current = frame.loc[win_start:win_end]
        if len(current) < 24:
            win_start = win_end
            continue
        reference = _season_matched_reference(frame, win_start, win_end)
        if len(reference) < 24:
            win_start = win_end
            continue

        report = drift_report(
            reference, current, features,
            per_feature_threshold, magnitude_threshold,
        )
        retrained = False
        if report["alert"]:
            train_now = frame.loc[:win_start]
            model = model_fn().fit(
                train_now.drop(columns="y"), train_now["y"]
            )
            retrained = True

        X_cur = current.drop(columns="y")
        y_cur = current["y"].values
        y_hat = model.predict(X_cur)

        rows.append({
            "window_start": win_start,
            "window_end": win_end,
            "drift_share": report["share"],
            "alert": report["alert"],
            "retrained": retrained,
            "MAE": float(np.mean(np.abs(y_hat - y_cur))),
            "MAPE_pct": 100 * float(np.mean(np.abs((y_hat - y_cur) / y_cur))),
            "n_hours": len(current),
        })
        win_start = win_end

    return pd.DataFrame(rows)
