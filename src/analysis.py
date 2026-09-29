"""Error analysis for day-ahead forecasts.

Takes a predictions frame (as produced by
``backtest.backtest_model(..., collect_predictions=True)``) and cuts it by
demand period, weekday and day type. Period boundaries are derived from the
data via terciles of hourly mean load, not hard-coded.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import holidays as holidays_pkg

    HAS_HOLIDAYS = True
except ImportError:  # pragma: no cover
    HAS_HOLIDAYS = False


def _prep(preds: pd.DataFrame) -> pd.DataFrame:
    """Attach absolute error, signed error and calendar attributes."""
    out = preds.copy()
    out["datetime"] = pd.to_datetime(out["datetime"])
    out["error"] = out["y_pred"] - out["y_true"]
    out["abs_error"] = out["error"].abs()
    out["ape"] = (out["abs_error"] / out["y_true"]).replace([np.inf, -np.inf], np.nan)
    out["hour"] = out["datetime"].dt.hour
    out["dow"] = out["datetime"].dt.day_name()
    out["is_weekend"] = out["datetime"].dt.dayofweek >= 5
    return out


def label_hour_periods(series: pd.Series) -> pd.Series:
    """Rank the 24 hours by mean load, cut into terciles: off-peak / transition / peak."""
    hourly_mean = series.groupby(series.index.hour).mean()
    lo, hi = hourly_mean.quantile([1 / 3, 2 / 3])

    def label(h: int) -> str:
        m = hourly_mean.loc[h]
        if m <= lo:
            return "Off-peak"
        if m >= hi:
            return "Peak"
        return "Transition"

    return pd.Series({h: label(h) for h in range(24)}, name="period")


def error_by_period(preds: pd.DataFrame, series: pd.Series) -> pd.DataFrame:
    """MAE, MAPE, bias and mean load broken down by demand period."""
    df = _prep(preds)
    labels = label_hour_periods(series)
    df["period"] = df["hour"].map(labels)

    out = (
        df.groupby("period")
        .agg(
            MAE=("abs_error", "mean"),
            MAPE_pct=("ape", lambda x: 100 * x.mean()),
            mean_load=("y_true", "mean"),
            bias=("error", "mean"),
            n=("error", "size"),
        )
        .round(2)
    )
    order = ["Off-peak", "Transition", "Peak"]
    return out.reindex([p for p in order if p in out.index])


def error_by_hour(preds: pd.DataFrame) -> pd.DataFrame:
    """MAE and signed bias for every hour of the day. Feeds the diurnal bias chart."""
    df = _prep(preds)
    return (
        df.groupby("hour")
        .agg(MAE=("abs_error", "mean"), bias=("error", "mean"), n=("error", "size"))
        .round(2)
    )


def error_by_day_type(preds: pd.DataFrame, country: str = "TR") -> pd.DataFrame:
    """MAE and MAPE broken out by weekday, weekend and public holiday.

    Holidays are pulled from the ``holidays`` package for ``country``. Weekday
    rows exclude weekends and holidays so the segments are disjoint.
    """
    df = _prep(preds)

    if HAS_HOLIDAYS:
        years = range(
            df["datetime"].dt.year.min(), df["datetime"].dt.year.max() + 1
        )
        holiday_dates = set(
            holidays_pkg.country_holidays(country, years=years).keys()
        )
        df["is_holiday"] = df["datetime"].dt.date.isin(holiday_dates)
    else:
        df["is_holiday"] = False

    rows = []
    for name, mask in [
        ("Public holiday", df["is_holiday"]),
        ("Weekend", df["is_weekend"] & ~df["is_holiday"]),
        ("Normal day", ~df["is_weekend"] & ~df["is_holiday"]),
    ]:
        sub = df[mask]
        rows.append(
            {
                "segment": name,
                "MAE": sub["abs_error"].mean(),
                "MAPE_pct": 100 * sub["ape"].mean(),
                "n": len(sub),
            }
        )

    weekday_only = df[~df["is_weekend"] & ~df["is_holiday"]]
    for day, sub in weekday_only.groupby("dow"):
        rows.append(
            {
                "segment": day,
                "MAE": sub["abs_error"].mean(),
                "MAPE_pct": 100 * sub["ape"].mean(),
                "n": len(sub),
            }
        )
    return pd.DataFrame(rows).set_index("segment").round(2)


def regime_shift_split(
    preds: pd.DataFrame, cut: str = "2020-03-16"
) -> pd.DataFrame:
    """Compare MAE, RMSE and MAPE before and after a distribution-shift cut date."""
    df = _prep(preds)
    cut_ts = pd.Timestamp(cut)

    def score(sub: pd.DataFrame) -> dict:
        return {
            "MAE": sub["abs_error"].mean(),
            "RMSE": float(np.sqrt((sub["error"] ** 2).mean())),
            "MAPE_pct": 100 * sub["ape"].mean(),
            "n": len(sub),
        }

    out = pd.DataFrame(
        {
            "Before": score(df[df["datetime"] < cut_ts]),
            "After": score(df[df["datetime"] >= cut_ts]),
        }
    ).T
    out.loc["Degradation_x"] = out.loc["After"] / out.loc["Before"]
    return out.round(2)


def year_over_year_load(
    series: pd.Series, baseline_year: int, shock_year: int
) -> pd.DataFrame:
    """Weekly load in ``shock_year`` compared to the same ISO weeks in ``baseline_year``."""
    df = series.to_frame("load")
    df["year"] = df.index.year
    df["week"] = df.index.isocalendar().week

    weekly = df.groupby(["year", "week"])["load"].sum().unstack(0)
    if baseline_year not in weekly.columns or shock_year not in weekly.columns:
        return pd.DataFrame()

    out = pd.DataFrame(
        {
            f"{baseline_year}_MWh": weekly[baseline_year],
            f"{shock_year}_MWh": weekly[shock_year],
        }
    ).dropna()
    out["change_pct"] = (
        100
        * (out[f"{shock_year}_MWh"] - out[f"{baseline_year}_MWh"])
        / out[f"{baseline_year}_MWh"]
    )
    return out.round(2)
