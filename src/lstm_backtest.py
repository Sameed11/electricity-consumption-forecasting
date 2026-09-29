"""LSTM backtest wrapper.

The training loop lives in ``models.train_lstm``. This module handles the
sequence prep, per-fold scaling and the head-to-head comparison against the
GBM baseline on identical folds.

Scaling is fitted only on the training portion of each fold. The chronological
tail of the training window is held out for early stopping so the validation
set never contains information younger than the earliest test point.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .backtest import make_splitter, metrics
from .features import STATIC_COLS, build_sequences
from .models import HAS_TORCH

SEQ_LEN = 72
MIN_LAG = 24
VAL_HOURS = 24 * 30


def _fit_scalers(series: pd.Series, frame: pd.DataFrame, fit_idx: np.ndarray):
    """Fit sequence, static and target scalers on the training portion only."""
    fit_index = frame.index[fit_idx]
    seq_fit = build_sequences(series, fit_index, SEQ_LEN, MIN_LAG)

    seq_scaler = StandardScaler().fit(seq_fit.reshape(-1, 1))
    static_scaler = StandardScaler().fit(frame.loc[fit_index, STATIC_COLS])
    y_scaler = StandardScaler().fit(
        frame.loc[fit_index, "y"].values.reshape(-1, 1)
    )
    return seq_scaler, static_scaler, y_scaler


def _transform(
    series: pd.Series,
    frame: pd.DataFrame,
    idx: np.ndarray,
    seq_scaler,
    static_scaler,
    y_scaler,
):
    """Apply pre-fitted scalers to any slice of the frame."""
    index = frame.index[idx]
    seq = build_sequences(series, index, SEQ_LEN, MIN_LAG)
    seq = seq_scaler.transform(seq.reshape(-1, 1)).reshape(seq.shape)
    static = static_scaler.transform(frame.loc[index, STATIC_COLS])
    y = y_scaler.transform(frame.loc[index, "y"].values.reshape(-1, 1)).ravel()
    return seq, static, y


def backtest_lstm(
    series: pd.Series,
    frame: pd.DataFrame,
    splitter=None,
    val_hours: int = VAL_HOURS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Backtest the PyTorch LSTM on the same folds as every other model."""
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is not installed")

    import torch

    from .models import train_lstm

    splitter = splitter or make_splitter()
    rows, all_preds = [], []

    for fold, (train_idx, test_idx) in enumerate(splitter.split(frame), 1):
        train_idx = np.asarray(train_idx)
        test_idx = np.asarray(test_idx)

        # Chronological validation tail for early stopping.
        val_size = min(val_hours, len(train_idx) // 5)
        fit_idx, val_idx = train_idx[:-val_size], train_idx[-val_size:]

        seq_scaler, static_scaler, y_scaler = _fit_scalers(series, frame, fit_idx)
        args = (series, frame)
        scalers = (seq_scaler, static_scaler, y_scaler)
        seq_tr, static_tr, y_tr = _transform(*args, fit_idx, *scalers)
        seq_val, static_val, y_val = _transform(*args, val_idx, *scalers)
        seq_te, static_te, _ = _transform(*args, test_idx, *scalers)

        started = time.time()
        model, epochs = train_lstm(
            seq_tr, static_tr, y_tr, seq_val, static_val, y_val
        )

        model.eval()
        with torch.no_grad():
            preds_scaled = model(
                torch.tensor(seq_te, dtype=torch.float32).unsqueeze(-1),
                torch.tensor(static_te, dtype=torch.float32),
            ).numpy()
        preds = y_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).ravel()

        y_true = frame.loc[frame.index[test_idx], "y"].values
        row = metrics(y_true, preds)
        row.update(
            {
                "model": "LSTM",
                "fold": fold,
                "fit_seconds": round(time.time() - started, 1),
                "epochs": epochs,
            }
        )
        rows.append(row)

        all_preds.append(
            pd.DataFrame(
                {
                    "datetime": frame.index[test_idx],
                    "y_true": y_true,
                    "y_pred": preds,
                    "model": "LSTM",
                    "fold": fold,
                }
            )
        )

    return pd.DataFrame(rows), pd.concat(all_preds, ignore_index=True)
