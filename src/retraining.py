"""Retraining-cadence experiment.

Same features, same model, same evaluation window as ``backtest_model``. Only
the deployment cadence differs. On the 2020-Q2 shock, weekly retraining cuts
MAE by ~12% and closes ~23% of the shock-induced gap (README section on
distribution shift).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import make_splitter, metrics


def backtest_with_cadence(
    frame: pd.DataFrame,
    model_fn,
    cadence_days: int | None = None,
    splitter=None,
) -> pd.DataFrame:
    """Predict each fold with an optional retraining cadence.

    ``cadence_days=None`` trains once at the start of the fold.
    ``cadence_days=7`` refits every seven test days on all data up to that
    point, mirroring a weekly deployment.
    """
    splitter = splitter or make_splitter()
    X = frame.drop(columns="y")
    y = frame["y"]

    all_preds = []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(frame), 1):
        train_idx = np.asarray(train_idx)
        test_idx = np.asarray(test_idx)

        if cadence_days is None:
            model = model_fn().fit(X.iloc[train_idx], y.iloc[train_idx])
            preds = model.predict(X.iloc[test_idx])
        else:
            preds = np.empty(len(test_idx))
            step = cadence_days * 24
            for start in range(0, len(test_idx), step):
                end = min(start + step, len(test_idx))
                slice_test = test_idx[start:end]
                available = np.arange(slice_test[0])
                model = model_fn().fit(X.iloc[available], y.iloc[available])
                preds[start:end] = model.predict(X.iloc[slice_test])

        all_preds.append(
            pd.DataFrame(
                {
                    "datetime": frame.index[test_idx],
                    "y_true": y.iloc[test_idx].values,
                    "y_pred": preds,
                    "fold": fold,
                }
            )
        )

    return pd.concat(all_preds, ignore_index=True)


def cadence_comparison(
    frame: pd.DataFrame,
    model_fn,
    cadences: dict,
    fold: int | None = None,
    splitter=None,
) -> pd.DataFrame:
    """Score every cadence in ``cadences`` (``label -> days-or-None``) on the same folds.

    Pass ``fold=4`` to reproduce the README's shock-window numbers.
    """
    splitter = splitter or make_splitter()
    rows = []
    for label, days in cadences.items():
        preds = backtest_with_cadence(frame, model_fn, days, splitter)
        if fold is not None:
            preds = preds[preds["fold"] == fold]
        row = metrics(preds["y_true"], preds["y_pred"])
        row["strategy"] = label
        row["cadence_days"] = days if days is not None else 0
        rows.append(row)
    return pd.DataFrame(rows).set_index("strategy").round(2)
