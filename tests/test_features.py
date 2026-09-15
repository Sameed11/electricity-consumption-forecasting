"""Leakage tests.

The single most damaging bug in a forecasting pipeline is a feature that quietly
contains information from the future. These tests assert it cannot happen.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features import build_sequences, make_features  # noqa: E402


@pytest.fixture
def series() -> pd.Series:
    index = pd.date_range("2018-01-01", periods=5000, freq="h")
    rng = np.random.default_rng(0)
    values = (
        30000
        + 4000 * np.sin(2 * np.pi * index.hour / 24)
        + rng.normal(0, 300, len(index))
    )
    return pd.Series(values, index=index, name="usage")


@pytest.mark.parametrize("min_lag", [1, 24])
def test_lag_columns_point_to_the_past(series, min_lag):
    frame = make_features(series, min_lag=min_lag)
    sample = frame.sample(200, random_state=0)

    for column in frame.columns:
        if not column.startswith("lag_"):
            continue
        lag = int(column.split("_")[1])
        assert lag >= min_lag, f"{column} is newer than the {min_lag}h horizon"
        expected = series.reindex(sample.index - pd.Timedelta(hours=lag))
        np.testing.assert_allclose(sample[column].values, expected.values)


def test_rolling_features_exclude_the_target(series):
    """A rolling window must never include the value being predicted."""
    min_lag = 24
    frame = make_features(series, min_lag=min_lag)
    shifted = series.shift(min_lag)

    expected = shifted.rolling(24, min_periods=6).max().reindex(frame.index)
    np.testing.assert_allclose(frame["roll_max_24"].values, expected.values)
    assert (frame["roll_max_24"] != frame["y"]).mean() > 0.99


def test_no_missing_values_survive(series):
    frame = make_features(series, min_lag=24)
    assert not frame.isna().any().any()
    assert len(frame) > 0


def test_sequences_are_ordered_and_lagged(series):
    seq_len, min_lag = 72, 24
    index = series.index[1000:1010]
    sequences = build_sequences(series, index, seq_len=seq_len, min_lag=min_lag)

    assert sequences.shape == (len(index), seq_len)
    for row, timestamp in enumerate(index):
        assert sequences[row, -1] == series.loc[timestamp - pd.Timedelta(hours=min_lag)]
        assert sequences[row, 0] == series.loc[
            timestamp - pd.Timedelta(hours=min_lag + seq_len - 1)
        ]
        assert sequences[row, -1] != series.loc[timestamp]


def test_horizon_changes_the_earliest_usable_lag(series):
    day_ahead = make_features(series, min_lag=24)
    next_hour = make_features(series, min_lag=1)
    assert "lag_1" in next_hour.columns
    assert "lag_1" not in day_ahead.columns
