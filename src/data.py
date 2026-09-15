"""Loading and cleaning of the hourly electricity demand series."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TARGET_COL = "Consumption Amount (MWh)"


def load_and_clean(path: str | Path) -> tuple[pd.Series, dict]:
    """Load the raw CSV and return a gap-free hourly series plus a data-quality audit.

    The raw file uses European number formatting ('.' thousands, ',' decimal) and
    contains a small number of daylight-saving artefacts. Rows are never dropped:
    deleting a row from a time series does not leave a hole, it shifts every
    subsequent observation and silently corrupts every lag computed afterwards.
    Bad values are marked missing and the series is reindexed onto a complete
    hourly grid before interpolation.

    Returns
    -------
    series : pd.Series
        Hourly demand in MWh on a complete DatetimeIndex.
    audit : dict
        Row counts, duplicate and zero counts, and the share of imputed values.
    """
    path = Path(path)
    raw = pd.read_csv(path)
    audit: dict = {"raw_rows": len(raw)}

    raw["Date"] = pd.to_datetime(raw["Date"], format="%d.%m.%Y")

    if not pd.api.types.is_numeric_dtype(raw[TARGET_COL]):
        raw[TARGET_COL] = (
            raw[TARGET_COL]
            .astype(str)
            .str.replace(".", "", regex=False)
            .str.replace(",", ".", regex=False)
            .astype(float)
        )

    raw["DateTime"] = pd.to_datetime(raw["Date"].astype(str) + " " + raw["Time"])
    raw = raw[["DateTime", TARGET_COL]].rename(columns={TARGET_COL: "usage"})

    audit["duplicate_timestamps"] = int(raw["DateTime"].duplicated().sum())
    raw = raw.drop_duplicates(subset="DateTime", keep="first")
    raw = raw.set_index("DateTime").sort_index()

    # Zero national grid load is physically impossible -> treat as missing.
    audit["zero_values"] = int((raw["usage"] == 0).sum())
    audit["zero_timestamps"] = [str(t) for t in raw.index[raw["usage"] == 0]]
    raw.loc[raw["usage"] == 0, "usage"] = np.nan

    full_index = pd.date_range(raw.index.min(), raw.index.max(), freq="h")
    audit["missing_hours_in_file"] = int(len(full_index) - len(raw))
    series = raw["usage"].reindex(full_index)

    audit["nan_before_fill"] = int(series.isna().sum())
    series = series.interpolate(method="time", limit_direction="both")
    audit["nan_after_fill"] = int(series.isna().sum())

    audit["final_rows"] = len(series)
    audit["imputed_pct"] = round(100 * audit["nan_before_fill"] / len(series), 4)
    audit["date_range"] = f"{series.index.min()} -> {series.index.max()}"
    audit["years_covered"] = round(
        (series.index.max() - series.index.min()).days / 365.25, 2
    )

    series.name = "usage"
    series.index.name = "datetime"
    return series, audit
