"""Feature engineering for day-ahead demand forecasting.

The central constraint is the ``min_lag`` parameter. Every feature derived from
the demand series is shifted by at least ``min_lag`` hours, so a model trained on
this frame only ever sees information that would genuinely have been available at
forecast time.

``min_lag=24`` gives the day-ahead task (what a grid operator actually solves).
``min_lag=1``  gives the next-hour task, which is far easier and mostly solved by
autocorrelation alone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import holidays as holidays_pkg

    HAS_HOLIDAYS = True
except ImportError:  # pragma: no cover
    HAS_HOLIDAYS = False

# Seasonal lags that matter for hourly load: same hour yesterday, two days back,
# three days back, same hour last week, and same hour a fortnight back.
SEASONAL_LAGS = [24, 25, 48, 72, 168, 169, 336, 720]
ROLLING_WINDOWS = [24, 168, 720]

# Feature subset handed to the LSTM head alongside the raw sequence.
STATIC_COLS = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos",
    "is_holiday", "is_holiday_eve", "is_weekend",
    "lag_168", "lag_336", "roll_mean_168",
]


def get_holiday_dates(index: pd.DatetimeIndex, country: str = "TR") -> set:
    """Public holiday dates covering the span of ``index``. Empty set if unavailable."""
    if not HAS_HOLIDAYS:
        return set()
    try:
        years = range(index.year.min(), index.year.max() + 1)
        return set(holidays_pkg.country_holidays(country, years=years).keys())
    except Exception:  # pragma: no cover - country code not supported
        return set()


def make_features(
    series: pd.Series, min_lag: int, country: str = "TR"
) -> pd.DataFrame:
    """Build a supervised frame whose features are all >= ``min_lag`` hours old.

    Column ``y`` is the target. Rows containing NaN from the warm-up period are
    dropped, so the frame starts once every feature is defined.
    """
    df = pd.DataFrame({"y": series})
    index = df.index

    recent = list(range(min_lag, min_lag + 24))
    lags = sorted(set(recent + [lag for lag in SEASONAL_LAGS if lag >= min_lag]))
    for lag in lags:
        df[f"lag_{lag}"] = series.shift(lag)

    # Shift first, then roll. Rolling an unshifted series would let the window
    # include the target itself.
    base = series.shift(min_lag)
    for window in ROLLING_WINDOWS:
        roll = base.rolling(window, min_periods=max(2, window // 4))
        df[f"roll_mean_{window}"] = roll.mean()
        df[f"roll_std_{window}"] = roll.std()
        df[f"roll_min_{window}"] = roll.min()
        df[f"roll_max_{window}"] = roll.max()

    df["diff_24_168"] = df["roll_mean_24"] - df["roll_mean_168"]
    df["diff_lag_roll"] = df[f"lag_{min_lag}"] - df["roll_mean_168"]

    # Calendar attributes of the target timestamp are known arbitrarily far ahead.
    df["hour"] = index.hour
    df["dow"] = index.dayofweek
    df["month"] = index.month
    df["doy"] = index.dayofyear
    df["year"] = index.year
    df["is_weekend"] = (index.dayofweek >= 5).astype(int)

    for col, period in [("hour", 24), ("dow", 7), ("month", 12), ("doy", 365.25)]:
        df[f"{col}_sin"] = np.sin(2 * np.pi * df[col] / period)
        df[f"{col}_cos"] = np.cos(2 * np.pi * df[col] / period)

    holiday_dates = get_holiday_dates(index, country)
    today = pd.Series(index.date, index=index)
    tomorrow = pd.Series(
        pd.to_datetime(index.date) + pd.Timedelta(days=1), index=index
    ).dt.date
    df["is_holiday"] = today.isin(holiday_dates).astype(int).values
    df["is_holiday_eve"] = tomorrow.isin(holiday_dates).astype(int).values

    df["t_index"] = np.arange(len(df))

    return df.dropna()


def build_sequences(
    series: pd.Series, index: pd.DatetimeIndex, seq_len: int, min_lag: int
) -> np.ndarray:
    """Sequence input for recurrent models, oldest observation first.

    The final column of each row is exactly ``min_lag`` hours before the target,
    so the day-ahead constraint holds for sequence models too.
    """
    columns = [
        series.shift(min_lag + k).reindex(index).values
        for k in range(seq_len - 1, -1, -1)
    ]
    return np.stack(columns, axis=1)
