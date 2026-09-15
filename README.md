# Day-Ahead Electricity Demand Forecasting

Day-ahead (24-hour horizon) forecasting of Turkish national grid load, benchmarked
against seasonal-naive baselines over a rolling-origin backtest. The project covers
the full pipeline: data quality auditing, leakage-safe feature engineering, model
comparison, error diagnosis, uncertainty quantification, and a retraining-cadence
experiment run over a real distribution shift.

**Data**: 39,456 hourly observations, 2015-12-31 to 2020-06-30 (4.5 years).

---

## Headline results

Day-ahead forecasting, mean over 4 expanding-window folds of 90 days each,
with a 24-hour gap between train and test. MAE and RMSE in MWh.

| Model | MAE | RMSE | MAPE | vs. seasonal naive | Fit time |
|---|---|---|---|---|---|
| LSTM (PyTorch) | **826.4** | **1193.5** | **2.74%** | 55.9% | 79–153 s/fold |
| LightGBM | 841.8 | 1229.0 | 2.79% | 55.0% | 8 s/fold |
| Random Forest | 833.7 | 1245.3 | 2.81% | 55.5% | 60 s/fold |
| Ridge | 1127.6 | 1483.8 | 3.67% | 39.8% | 0.2 s/fold |
| Weekly naive (t−168) | 1719.6 | 2535.7 | 5.57% | 8.2% | — |
| Seasonal naive (t−24) | 1872.5 | 2944.3 | 5.98% | — | — |

**LightGBM is the production choice, despite not topping the table.** The LSTM's
15.4 MWh advantage is not statistically significant — per-fold differences are
−7.7, +11.0, −11.0, −54.0, giving a paired *t* = −1.12 (*p* = 0.34), with almost the
entire margin coming from a single fold. A 1.8% gain that fails a significance test
does not justify 15× the training time.

### Why the horizon matters

An earlier version of this project forecast one hour ahead and compared against a
persistence (t−1) baseline. Hourly load is so autocorrelated that this is close to a
free win, and persistence is not a legal baseline for day-ahead forecasting — at
forecast time, last hour's load is not yet known. Grid operators schedule generation
a day in advance, so that is the task modelled here.

| Task | Baseline | Baseline MAE | Baseline MAPE |
|---|---|---|---|
| Next hour (t+1) | Persistence (t−1) | 999.7 | 3.12% |
| **Day ahead (t+24)** | **Seasonal naive (t−24)** | **1872.5** | **5.98%** |

The honest baseline is 1.9× harder to beat.

---

## What the model relies on

Mean feature importance across folds:

| Feature | Importance |
|---|---|
| `lag_168` (same hour, last week) | 0.341 |
| `lag_336` (same hour, two weeks ago) | 0.183 |
| `lag_24` (same hour, yesterday) | 0.160 |
| `lag_169` | 0.073 |
| `lag_25` | 0.058 |
| `is_holiday` | 0.037 |

Roughly 68% of the model is week-lagged and fortnight-lagged demand. Cyclical
encodings (`hour_sin`, `dow_sin`, etc.) contribute under 1% each — the lag features
already carry time-of-day and day-of-week structure, making the trigonometric
encodings largely redundant here. This dependence on week-old demand is also the
model's central weakness, as the regime-shift section shows.

---

## Error analysis

### By demand period

Hours are assigned to periods by tercile of mean load, derived from the data.

| Period | MAE | MAPE | Mean load | Bias |
|---|---|---|---|---|
| Off-peak (00–07) | 553.3 | 2.07% | 28,368 | −71.9 |
| Transition (08–10, 12–13, 21–23) | 966.4 | **3.22%** | 33,430 | +76.9 |
| Peak (11, 14–20) | 1005.7 | 3.09% | 35,247 | +52.9 |

Transition hours carry the highest *relative* error — the morning and evening ramps
are the hardest part of the daily curve to place, even though peak hours carry more
absolute error.

The bias column is systematic rather than random: the model over-forecasts hours
07–16 (peaking at +155.6 MWh at 13:00) and under-forecasts overnight hours
(−129.1 MWh at 04:00). This is the signature of the 2020 demand shock — commercial
and industrial daytime load fell sharply while residential overnight load held —
being absorbed into an aggregate trained largely on pre-shock data. Overall bias is
+19.3 MWh, 0.06% of mean load, so the model is unbiased in aggregate.

### By day type

| Segment | MAE | MAPE |
|---|---|---|
| Thursday (best weekday) | 661.9 | 2.00% |
| Monday (worst weekday) | 1165.4 | 3.67% |
| Normal day | 794.3 | 2.57% |
| **Public holiday** | **2015.4** | **8.33%** |

**Holidays are the largest single weakness: 2.54× the normal-day error**, and this is
*with* `is_holiday` already among the six most important features. A binary flag
cannot represent Turkish public holidays adequately — the religious holidays
(Ramazan and Kurban Bayramı) follow the lunar calendar, so they do not recur on the
same dates year to year, and they span several days with different demand profiles
on each. Monday's weakness follows from the same root cause: with `lag_168` carrying
34% of the model, Monday's reference point sits on the far side of a weekend.

---

## Distribution shift: the 2020 demand shock

Fold 4 (April–June 2020) degraded sharply while the naive baselines did not. That
asymmetry — learned models collapsing while trivial ones hold — is the signature of
distribution shift rather than of a modelling bug.

| Model | Fold 2 MAE | Fold 4 MAE | Change |
|---|---|---|---|
| Random Forest | 511.8 | 1337.1 | **+161%** |
| LightGBM | 524.3 | 1284.6 | +145% |
| LSTM | 535.3 | 1230.6 | +130% |
| Seasonal naive (t−24) | 1845.3 | 1797.3 | −3% |
| Persistence (t−1) | 1030.4 | 861.5 | −16% |

Split at 2020-03-16, when COVID-19 restrictions began:

| | MAE | RMSE | MAPE |
|---|---|---|---|
| Before | 659.6 | 996.3 | 1.93% |
| After | 1272.7 | 1793.6 | 4.83% |
| Degradation | **1.93×** | 1.80× | 2.50× |

Underlying demand, compared week-on-week against 2019: weeks 1–11 ran +2.8%,
weeks 12–26 ran −9.5%, and week 22 fell **26.3% below** the previous year.

The mechanism is the feature importance table. A model that is 52% `lag_168` and
`lag_336` forecasts by reference to demand one and two weeks old. When the level
shifts within days, those references are stale. Persistence survived precisely
because `t−1` tracks the new regime immediately.

### Mitigation: retraining cadence

Same model, same features, same evaluation window. Only the deployment cadence
differs.

| Strategy | MAE | RMSE | MAPE |
|---|---|---|---|
| Train once | 1284.6 | 1830.7 | 5.02% |
| Retrain weekly (13 refits) | **1132.2** | 1658.9 | 4.40% |

**11.9% MAE reduction at zero modelling cost.** Reported honestly, though: the shock
opened a 613 MWh gap and weekly retraining closed 140 of it, about **23%**.
Retraining cadence is a mitigation, not a fix.

---

## Uncertainty quantification

Generation is not scheduled on a point forecast; reserve capacity is priced off the
upper bound. Quantile models were fitted at the 5th, 50th and 95th percentiles.

| Metric | Value |
|---|---|
| Empirical coverage (target 90%) | **65.7%** |
| Mean interval width | 2058 MWh (6.4% of mean load) |
| Pinball loss (q05 / q50 / q95) | 205.3 / 427.2 / 138.7 |

Coverage by fold: 67.6%, 67.5%, 72.1%, **55.7%**.

**These intervals are not fit for use.** The under-coverage is systematic across
every fold, not just the shock window, so it is not simply a consequence of COVID.
Quantile regression fitted on training residuals underestimates out-of-sample
uncertainty whenever the test period does not resemble the training period. The
correct remedy is conformalized quantile regression (CQR), which calibrates the
quantiles on a held-out block and provides a finite-sample coverage guarantee. This
is listed under known limitations rather than quietly dropped.

---

## Repository layout

```
.
├── src/
│   ├── data.py          # loading, cleaning, data-quality audit
│   ├── features.py      # leakage-safe feature engineering
│   ├── backtest.py      # rolling-origin backtesting, metrics, pinball loss
│   └── models.py        # model factories including the PyTorch LSTM
├── scripts/
│   └── run_backtest.py  # reproduce the headline table from the CLI
├── tests/
│   └── test_features.py # leakage tests
├── notebooks/
│   ├── 01_modelling.ipynb       # cleaning, features, baselines, backtest
│   └── 02_error_analysis.ipynb  # diagnostics, regime shift, intervals, LSTM
├── data/                # dataset (place the CSV here)
├── .gitignore
├── requirements.txt
└── README.md
```

## Reproducing the results

```bash
pip install -r requirements.txt

# Place the CSV in data/, then:
python scripts/run_backtest.py --data "data/Electricity Consumption 2015-2020.csv"

# Compare against the easier next-hour framing:
python scripts/run_backtest.py --data "data/..." --horizon 1

pytest tests/
```

All models use a fixed random seed. The backtest protocol (4 folds, 90-day test
windows, 24-hour gap) is defined in `src/backtest.py`.

---

## Methodology notes

**Leakage prevention.** Every feature derived from the demand series is shifted by
at least the forecast horizon before use — rolling statistics are computed on the
shifted series, never the raw one. `tests/test_features.py` asserts this by
reconstructing each lag column from the source series independently.

**No rows are dropped.** Deleting a row from a time series does not leave a gap; it
shifts every subsequent observation and corrupts every lag computed afterwards. Bad
values (two daylight-saving artefacts and one duplicate timestamp) are marked
missing, the series is reindexed onto a complete hourly grid, and 0.005% of values
are interpolated.

**Model sizing.** Leaving `min_samples_leaf` at its default grew a forest of
2,283,257 leaves averaging 1.01 samples each — memorisation, and a model several
hundred MB in size. Setting `min_samples_leaf=5` reduced this to 372,036 leaves,
**6.1× smaller, with slightly better fold MAE**.

**Fair model comparison.** An earlier version gave the Random Forest 26 lag features
while the LSTM saw only the raw scaled series, making the comparison meaningless.
Both model families here receive the same information set: the LSTM gets a 72-hour
sequence (every value at least 24 hours old) plus the same calendar and
seasonal-lag features joined at the head.

---

## Known limitations

1. **Holiday modelling is inadequate** — 2.54× error on public holidays. A binary
   flag cannot capture lunar-calendar religious holidays or multi-day holiday
   profiles. Next step: holiday-type categories, day-position-within-holiday, and
   bridge-day indicators.
2. **Prediction intervals are under-calibrated** — 65.7% empirical coverage against
   a 90% target. Next step: conformalized quantile regression.
3. **No exogenous variables.** Temperature is the single largest external driver of
   electricity demand and is absent here. Adding weather data would likely deliver
   a larger improvement than any further model tuning.
4. **Retraining cadence recovers only 23%** of the shock-induced degradation. Fully
   handling regime change would need drift detection and sample reweighting toward
   recent observations.
5. **Single-market data.** Results are specific to the Turkish grid, 2015–2020, and
   the COVID period makes the final fold unrepresentative of normal operation.

## Data source

Hourly Turkish electricity consumption, 2015–2020 (EPİAŞ / Turkish Electricity
Transmission Corporation). The dataset is committed under `data/` for reproducibility.
