# Day-Ahead Electricity Demand Forecasting

Day-ahead (24-hour horizon) hourly forecasts for the Turkish national grid,
2015–2020, packaged with a full MLOps stack: rolling-origin backtesting,
MLflow tracking and model registry, a Streamlit dashboard, a FastAPI serving
endpoint, and Evidently drift monitoring.

**Headline: LightGBM cuts MAE by 55% vs a seasonal-naive baseline (842 vs
1,872 MWh) across four folds, ties an LSTM statistically at 12× the training
cost, and a drift-triggered retraining policy recovers 16.5% of the
COVID-shock MAE degradation using only two retrains over six months.**

---

## Table of contents

1. [Problem statement](#problem-statement)
2. [Data](#data)
3. [Approach](#approach)
4. [Headline results](#headline-results)
5. [Error analysis](#error-analysis)
6. [The 2020 demand shock](#the-2020-demand-shock)
7. [Drift monitoring and drift-triggered retraining](#drift-monitoring-and-drift-triggered-retraining)
8. [Production stack](#production-stack)
9. [Repository layout](#repository-layout)
10. [Reproducing the results](#reproducing-the-results)
11. [Methodology notes](#methodology-notes)
12. [Known limitations](#known-limitations)
13. [Tech stack](#tech-stack)

---

## Problem statement

Grid operators must schedule generation capacity in advance of demand. Under-
scheduling triggers reserve activation or blackouts; over-scheduling burns
fuel that no one needed. The Turkish grid dispatches on a day-ahead auction,
so forecasts are required not for the next hour, but for hours **t+1 through
t+24** given only data available at time **t**.

That framing matters. An earlier version of this project forecast one hour
ahead against a persistence (t−1) baseline. Hourly load is autocorrelated
enough that this is close to a free win, and persistence is not a legal
baseline for day-ahead forecasting — at forecast time, last hour's load is
not yet known. Grid operators cannot use it. This project uses the honest
task.

| Task | Baseline | Baseline MAE | Baseline MAPE |
|---|---|---|---|
| Next hour (t+1) | Persistence (t−1) | 999.7 | 3.12% |
| **Day ahead (t+24)** | **Seasonal naive (t−24)** | **1,872.5** | **5.98%** |

The honest baseline is 1.9× harder to beat.

## Data

Hourly Turkish national grid load, 2015-12-31 to 2020-06-30 (EPİAŞ / Turkish
Electricity Transmission Corporation). 39,456 observations over 4.5 years.
The CSV is committed under `data/` for reproducibility.

**Data quality audit run at load time.** European number formatting (`.`
thousands, `,` decimal) is parsed. One duplicate timestamp and two
daylight-saving artefacts are marked missing rather than dropped — deleting a
row from a time series does not leave a hole; it shifts every subsequent
observation and silently corrupts every lag computed afterwards. The series
is reindexed onto a complete hourly grid and 0.005% of values are
interpolated.

## Approach

**Rolling-origin backtest**, 4 expanding folds, 90 days of test data each,
with a 24-hour gap between train and test so no training row is younger than
the earliest test point. Every reported number is a mean across the 4 folds.

**Leakage-safe feature engineering.** 61 features derived from the demand
series and calendar. Every feature is shifted by at least the forecast
horizon (24h). Rolling statistics are computed on the shifted series so a
rolling window can never include the value being predicted. `tests/test_features.py`
asserts this by reconstructing each lag column from the source series
independently.

Feature families:
- **Lags**: t−24 through t−48, plus t−72, t−168, t−336, t−720
- **Rolling statistics**: mean, std, min, max over 24h, 168h, 720h windows
- **Calendar**: hour, day-of-week, month, day-of-year (sin/cos encodings)
- **Holidays**: Turkish public holidays via the `holidays` library, plus holiday-eve

**Four models on identical folds:**

- **Ridge regression** — a linear model on the engineered features. Cheap. Shows how much of the signal is linear.
- **Random Forest** — sklearn `RandomForestRegressor`, tuned `min_samples_leaf=5`.
- **LightGBM** — the production choice. 1,200 trees, 63 leaves, 5% learning rate.
- **LSTM (PyTorch)** — 72-hour sequence + calendar features joined at the head, single-layer, 64 hidden units, early stopping on a held-out training tail.

**Fair comparison.** An earlier version gave the Random Forest 26 lag features
while the LSTM saw only the raw scaled series, making the comparison
meaningless. Here both model families receive the same information set: the
LSTM gets a 72-hour sequence plus the same calendar and seasonal-lag features
as the tree models.

## Headline results

Day-ahead forecasting, mean over 4 expanding-window folds of 90 days each,
with a 24-hour gap between train and test. MAE and RMSE in MWh.

| Model | MAE | RMSE | MAPE | vs. seasonal naive | Fit time |
|---|---|---|---|---|---|
| LSTM (PyTorch) | **792.3** | **1,161.6** | **2.60%** | 57.7% | 98.6 s/fold |
| Random Forest | 833.7 | 1,245.3 | 2.81% | 55.5% | 61.9 s/fold |
| LightGBM | 841.8 | 1,229.0 | 2.79% | 55.0% | **8.5 s/fold** |
| Ridge | 1,127.6 | 1,483.8 | 3.67% | 39.8% | 0.1 s/fold |
| Weekly naive (t−168) | 1,719.6 | 2,535.7 | 5.57% | 8.2% | — |
| Seasonal naive (t−24) | 1,872.5 | 2,944.3 | 5.98% | — | — |

**LightGBM is the production choice, despite not topping the table.** The
LSTM's 49.4 MWh MAE advantage is not statistically significant — a paired
t-test on per-fold MAE gives **t = −1.591, p = 0.210**. With four folds, the
difference doesn't clear the 0.05 bar. A 6% gain that fails a significance
test does not justify **12× the training time** and additional deployment
complexity. LightGBM also has the lowest fold-to-fold variance (MAE std 320
vs the LSTM's 259, and Random Forest's 353).

## Error analysis

### By demand period

Hours are assigned to periods by tercile of mean load, derived from the data.

| Period | MAE | MAPE | Mean load | Bias |
|---|---|---|---|---|
| Off-peak (00–07) | 553.3 | 2.07% | 28,368 | −71.9 |
| Transition (08–10, 12–13, 21–23) | 966.4 | **3.22%** | 33,430 | +76.9 |
| Peak (11, 14–20) | 1,005.7 | 3.09% | 35,247 | +52.9 |

Transition hours carry the highest *relative* error — morning and evening
ramps are the hardest part of the daily curve to place.

The bias column is systematic: the model over-forecasts hours 07–16 (peaking
at +155.6 MWh at 13:00) and under-forecasts overnight hours (−129.1 MWh at
04:00). This is the signature of the 2020 demand shock — commercial and
industrial daytime load fell while residential overnight load held — absorbed
into an aggregate trained largely on pre-shock data. Overall bias is +19.3
MWh, 0.06% of mean load: unbiased in aggregate.

### By day type

| Segment | MAE | MAPE |
|---|---|---|
| Thursday (best weekday) | 629.1 | 1.85% |
| Monday (worst weekday) | 1,076.4 | 3.25% |
| Normal day | 760.6 | 2.31% |
| Weekend | 876.0 | 3.20% |
| **Public holiday** | **2,015.4** | **8.33%** |

**Holidays are the largest single weakness: 2.65× the normal-day error**, and
this is *with* `is_holiday` already among the six most important features. A
binary flag cannot represent Turkish public holidays adequately — the
religious holidays (Ramazan and Kurban Bayramı) follow the lunar calendar, so
they do not recur on the same dates year to year, and they span several days
with different demand profiles on each. Monday's weakness follows from the
same root cause: with `lag_168` carrying 34% of the model, Monday's reference
point sits on the far side of a weekend.

## The 2020 demand shock

Fold 4 (April–June 2020) degraded sharply while the naive baselines did not.
That asymmetry — learned models collapsing while trivial ones hold — is the
signature of distribution shift, not a modelling bug.

| Model | Fold 2 MAE | Fold 4 MAE | Change |
|---|---|---|---|
| Random Forest | 511.8 | 1,337.1 | **+161%** |
| LightGBM | 524.3 | 1,284.6 | +145% |
| LSTM | 535.9 | 1,151.5 | +115% |
| Seasonal naive (t−24) | 1,845.3 | 1,797.3 | −3% |

Split at 2020-03-16, when COVID-19 restrictions began:

| | MAE | RMSE | MAPE |
|---|---|---|---|
| Before | 659.6 | 996.3 | 1.93% |
| After | 1,272.7 | 1,793.6 | 4.83% |
| Degradation | **1.93×** | 1.80× | **2.50×** |

**Underlying demand fell** in weeks 12–26 of 2020 by 9.5% on average vs 2019,
with week 22 falling **26.3%** below the previous year. A drop is counter-
intuitive at first: lockdown put more people at home using more appliances.
The resolution is that households are a minority of national load. Turkish
electricity consumption in 2020 split roughly 44.9% industrial, 26.7%
services, 23.5% residential and 4.4% agricultural, so commercial and
industrial demand together account for close to three quarters of the total.
Idled factories and closed offices outweighed the residential increase
several times over.

The mechanism is the feature importance table: a model that is over half
`lag_168` and `lag_336` forecasts by reference to demand one and two weeks
old. When the level shifts within days, those references are stale.

## Drift monitoring and drift-triggered retraining

The frozen model has no way to know it has gone out of date. This section
closes that loop.

**Reference: the same calendar window one year earlier.** Comparing to
last year's same period, instead of the whole training history, controls for
seasonality — winter-vs-summer differences don't get mistaken for regime
change.

**Signal: normalized Wasserstein distance per feature**, computed by
Evidently. Wasserstein is the actual distance between distributions, in units
of the reference standard deviation, so it stays interpretable across
features of different scales and across window sizes. Alert fires when the
worst single-feature distance crosses 10 std of the reference.

Bi-weekly replay of Jan–Jun 2020:

| Policy | Mean MAE | Median MAE | Worst MAE | Retrains |
|---|---|---|---|---|
| Frozen (no retrain) | 1,243.8 | 898.6 | 2,643.1 | 0 |
| **Drift-triggered** | **1,038.1** | **818.3** | **1,962.2** | **2** |
| **MAE reduction** | **−16.5%** | −9% | **−26%** | — |

The drift alert fires exactly where the model breaks. April 8 window:
max_distance jumps to 16.5, frozen MAE peaks at 2,643 MWh; the retrain drops
the next window's MAE from 2,010 → 1,495. May 20 fires again, catching a
second regime shift.

**Compare to fixed-cadence retraining** (from the shock analysis above):
weekly retraining recovers 11.9% of the loss but requires ~13 refits over the
same period. Drift-triggered retraining recovers more (16.5%) with 6× fewer
retrains. **The detector is doing work a fixed cadence can only approximate.**

## Production stack

Every artifact is a real service. Together they cover the full lifecycle from
experiment tracking to serving.

```
┌───────────────────┐   models:/electricity-forecaster@champion
│   FastAPI :8000   │◄──────────────────────────────┐
│   /forecast       │                               │
└─────────┬─────────┘                               │
          │ 24h JSON                                │
          ▼                                         │
   any HTTP client                                  │
                                                    │
┌───────────────────┐   sqlite:///mlflow.db  ┌──────┴──────┐
│  Streamlit :8501  │──────────────────────► │ MLflow :5000 │
│  6-page dashboard │   metrics/preds/model  │ UI + registry│
└───────────────────┘                        └──────────────┘
```

### MLflow tracking + model registry

Every backtest is logged as a parent run with one nested run per fold, so
per-fold metrics render in the MLflow UI without extra plumbing. Feature
importance and per-fold predictions are stored as artifacts. The production
LightGBM is registered as `electricity-forecaster` with alias `champion` — a
movable pointer so the API and dashboard can load "whichever version is
current" without knowing its number.

The paired LSTM-vs-LightGBM significance test runs as its own MLflow
"comparison" run, so the statistical argument for the production choice lives
alongside the models it compares.

### Streamlit dashboard

Six pages, all reading directly from the MLflow store (no local CSVs):

- **Overview** — headline table, per-fold MAE chart, LSTM vs LightGBM significance panel
- **Forecast explorer** — actual vs predicted for any (model, fold, date range)
- **Where it fails** — hourly MAE and signed bias, day-type breakdown
- **The 2020 shock** — before/after 2020-03-16 split, 2019 vs 2020 weekly demand
- **Champion model** — loads by alias, predicts a held-out 48h window fresh
- **Data quality** — the load-time audit and the full time series

### FastAPI serving

Three endpoints. Loads the champion by alias on first request, caches with
`lru_cache`.

- `GET /health` — service state, model version, model flavor, last observed timestamp
- `GET /forecast?hours=N` — next N hourly predictions (1–168), each with `datetime`, `hour_ahead`, `prediction_mwh`
- `POST /predict` — single-row prediction from a raw feature vector

`GET /docs` renders Swagger with Try-it-out buttons on every endpoint.

### Evidently drift monitoring

`notebooks/drift_demo.ipynb` replays Jan–Jun 2020 bi-weekly and compares
frozen model vs drift-triggered retraining. Section 7 above has the numbers.
The alert is a Wasserstein magnitude threshold, not a p-value, so it doesn't
break under the reference/current sample-size gap that trips K-S tests on
time-series drift.

## Repository layout

```
electricity-consumption-forecasting/
├── README.md                          # this file
├── requirements.txt
├── data/
│   └── Electricity Consumption 2015-2020.csv
│
├── src/                               # library — importable modules
│   ├── data.py                        # loading, cleaning, audit
│   ├── features.py                    # leakage-safe feature engineering
│   ├── backtest.py                    # rolling-origin backtest + metrics
│   ├── models.py                      # model factories inc. PyTorch LSTM
│   ├── analysis.py                    # error breakdowns, regime shift
│   ├── intervals.py                   # quantile GBM + CQR helper
│   ├── retraining.py                  # fixed-cadence retraining experiment
│   ├── lstm_backtest.py               # LSTM in the same interface as the trees
│   ├── mlflow_logging.py              # thin MLflow helpers
│   └── drift.py                       # Evidently drift + drift-triggered retrain
│
├── notebooks/
│   ├── run_pipeline.ipynb             # end-to-end walkthrough
│   ├── mlflow_pipeline.ipynb          # tracking + registry
│   └── drift_demo.ipynb               # Jan–Jun 2020 replay
│
├── scripts/
│   └── run_backtest.py                # CLI reproduction of the headline table
│
├── dashboards/
│   └── app.py                         # Streamlit, MLflow-backed
│
├── api/
│   ├── main.py                        # FastAPI /health, /forecast, /predict
│   └── schemas.py                     # Pydantic request/response models
│
├── tests/
│   └── test_features.py               # leakage tests
│
├── mlflow_store/                      # gitignored: SQLite backend + artifacts
└── outputs/                           # gitignored: local CSVs
```

## Reproducing the results

**Setup:**

```bash
pip install -r requirements.txt
```

**Reproduce the headline backtest table from the CLI:**

```bash
python scripts/run_backtest.py --data "data/Electricity Consumption 2015-2020.csv"
```

**Reproduce everything (including error analysis, intervals, LSTM,
retraining cadence) with charts:**

```bash
jupyter lab notebooks/run_pipeline.ipynb
```

**Run the leakage tests:**

```bash
pytest tests/
```

**Bring up the MLflow tracking store and register the champion:**

```bash
jupyter lab notebooks/mlflow_pipeline.ipynb
mlflow ui --backend-store-uri sqlite:///mlflow_store/mlflow.db --port 5000
```

**Serve the dashboard:**

```bash
streamlit run dashboards/app.py
```

**Serve the forecast API:**

```bash
uvicorn api.main:app --port 8000
curl "http://localhost:8000/forecast?hours=24"
```

**Run the drift replay:**

```bash
jupyter lab notebooks/drift_demo.ipynb
```

All models use a fixed random seed. The backtest protocol (4 folds, 90-day
test windows, 24-hour gap) is defined in `src/backtest.py`.

## Methodology notes

**Leakage prevention.** Every feature derived from the demand series is
shifted by at least the forecast horizon before use. Rolling statistics are
computed on the shifted series, never the raw one.
`tests/test_features.py` asserts this by reconstructing each lag column from
the source series independently and comparing element-wise.

**No rows are dropped.** Bad values (two daylight-saving artefacts and one
duplicate timestamp) are marked missing, the series is reindexed onto a
complete hourly grid, and 0.005% of values are interpolated. Deleting rows
would silently corrupt every downstream lag.

**Model sizing.** Leaving `min_samples_leaf` at its default grew a Random
Forest of 2,283,257 leaves averaging 1.01 samples each — memorisation, and a
model several hundred MB in size. Setting `min_samples_leaf=5` reduced this
to 372,036 leaves, **6.1× smaller, with slightly better fold MAE**.

**Fair model comparison.** Both tree models and the LSTM receive the same
information set. The LSTM gets a 72-hour sequence (every value at least 24
hours old) plus the same calendar and seasonal-lag features joined at the
head.

**Statistical rigour on model choice.** The LSTM–LightGBM difference is
reported as a paired t-statistic and p-value, not just point estimates. With
n=4 folds, the standard error is large; the decision goes to LightGBM on
training cost and fold stability, not raw MAE.

**Drift metric choice.** Wasserstein distance (Evidently's normalized
version) is preferred over Kolmogorov-Smirnov p-values because K-S with a
year-long reference and a two-week current window rejects the null almost
always due to sample-size asymmetry, not real drift. Wasserstein returns a
magnitude, not a hypothesis test, so it stays interpretable across window
sizes.

**Season-matched drift reference.** Each replay window is compared to the
same calendar window one year earlier. A four-year rolling reference would
have caught seasonality as "drift" and rung the alarm year-round.

## Known limitations

1. **Holiday modelling is inadequate** — 2.65× error on public holidays. A
   binary flag cannot capture lunar-calendar religious holidays or multi-day
   holiday profiles. Next step: holiday-type categories,
   day-position-within-holiday, and bridge-day indicators.
2. **Prediction intervals are under-calibrated** — 65.7% empirical coverage
   against a 90% target. The scaffolding for conformalized quantile
   regression (CQR) is in `src/intervals.py`; a conformal calibration block
   would restore nominal coverage.
3. **No exogenous variables.** Temperature is the single largest external
   driver of electricity demand and is absent here. Adding weather data would
   likely deliver a larger improvement than any further model tuning.
4. **Drift monitoring closes limitation #4 in part.** The drift-triggered
   retraining policy demonstrated here recovers 16.5% of the shock-induced
   MAE, versus 11.9% for fixed weekly retraining. Fully handling regime
   change would still need feature-level drift attribution (which feature
   moved and why) and sample reweighting toward the new distribution.
5. **Single-market data.** Results are specific to the Turkish grid,
   2015–2020, and the COVID period makes the final fold unrepresentative of
   normal operation.

## Tech stack

**Core**: Python 3.11, pandas, NumPy, scikit-learn
**Modelling**: LightGBM, PyTorch, statsmodels, holidays
**MLOps**: MLflow, Evidently, FastAPI, Streamlit, Uvicorn, Pydantic
**Testing**: pytest

---

## Data source

Hourly Turkish electricity consumption, 2015-12-31 to 2020-06-30
(EPİAŞ / Turkish Electricity Transmission Corporation). The dataset is
committed under `data/` for reproducibility.
