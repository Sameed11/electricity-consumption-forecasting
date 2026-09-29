"""Reproduce the headline backtest table from the command line.

    python scripts/run_backtest.py --data "data/Electricity Consumption 2015-2020.csv"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtest import (  # noqa: E402
    backtest_baseline, backtest_model, describe_folds, make_splitter, summarise,
)
from src.data import load_and_clean  # noqa: E402
from src.features import make_features  # noqa: E402
from src.models import make_gbm, make_random_forest, make_ridge  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Day-ahead electricity demand backtest")
    parser.add_argument("--data", required=True, help="path to the raw CSV")
    parser.add_argument("--output", default="outputs", help="directory for result files")
    parser.add_argument("--horizon", type=int, default=24,
                        help="forecast horizon in hours (24 = day-ahead, 1 = next-hour)")
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--country", default="TR", help="ISO code used for holiday features")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    series, audit = load_and_clean(args.data)
    print("Data quality audit")
    for key, value in audit.items():
        print(f"  {key:26s}: {value}")

    frame = make_features(series, min_lag=args.horizon, country=args.country)
    print(f"\nFrame: {frame.shape[0]:,} rows x {frame.shape[1] - 1} features "
          f"(horizon {args.horizon}h)\n")

    splitter = make_splitter(n_splits=args.folds)
    print(describe_folds(frame, splitter).to_string(index=False), "\n")

    results = [
        backtest_baseline(frame, "lag_24", "Seasonal naive (t-24)", splitter),
        backtest_baseline(frame, "lag_168", "Weekly naive (t-168)", splitter),
    ]
    predictions = []

    models = {
        "Ridge": make_ridge,
        "Random Forest": make_random_forest,
        "LightGBM": make_gbm,
    }
    for name, factory in models.items():
        print(f"Backtesting {name} ...", end=" ", flush=True)
        fold_results, importance, preds = backtest_model(
            frame, factory, name, splitter, collect_predictions=True
        )
        print(f"mean MAE {fold_results['MAE'].mean():,.1f}")
        results.append(fold_results)
        predictions.append(preds)
        if importance is not None:
            importance.to_csv(output_dir / f"importance_{name.lower().replace(' ', '_')}.csv")

    all_results = pd.concat(results, ignore_index=True)
    summary = summarise(all_results, baseline_name="Seasonal naive")

    print("\nRolling-origin backtest, MAE in MWh\n")
    print(summary.to_string())

    all_results.to_csv(output_dir / "backtest_results.csv", index=False)
    summary.to_csv(output_dir / "backtest_summary.csv")
    pd.concat(predictions, ignore_index=True).to_csv(output_dir / "predictions.csv", index=False)
    pd.Series(audit).to_csv(output_dir / "data_audit.csv")
    print(f"\nResults written to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
