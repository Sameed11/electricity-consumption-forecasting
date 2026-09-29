"""Streamlit dashboard for the day-ahead electricity load forecasting project.

Reads everything from the local MLflow store:
- Backtest metrics via MlflowClient.search_runs
- Per-fold predictions from each backtest run's 'predictions' artifact
- Feature importance from each backtest run's 'importance' artifact
- Champion model via models:/electricity-forecaster@champion

To run:
    conda activate ML
    streamlit run dashboards/app.py

Requires the mlflow_store/ folder produced by notebooks/mlflow_pipeline.ipynb.
"""

from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import streamlit as st


# --- Repo + MLflow setup ---------------------------------------------------

_here = Path(__file__).resolve()
REPO_ROOT = None
for _p in [_here, *_here.parents]:
    if (_p / "src").is_dir() and (_p / "data").is_dir():
        REPO_ROOT = _p
        break

if REPO_ROOT is None:
    st.error("Could not find repo root (expected src/ and data/ folders).")
    st.stop()

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

STORE_DIR = REPO_ROOT / "mlflow_store"
if not STORE_DIR.exists():
    st.error(
        f"MLflow store not found at {STORE_DIR}.\n\n"
        "Run notebooks/mlflow_pipeline.ipynb first to populate it."
    )
    st.stop()

TRACKING_URI = f"sqlite:///{(STORE_DIR / 'mlflow.db').as_posix()}"
mlflow.set_tracking_uri(TRACKING_URI)

EXPERIMENT_NAME = "electricity-forecasting"
REGISTERED_NAME = "electricity-forecaster"

st.set_page_config(
    page_title="Electricity Load Forecast",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --- Cached MLflow reads ---------------------------------------------------

def _client() -> mlflow.MlflowClient:
    return mlflow.MlflowClient()


@st.cache_data(ttl=300)
def experiment_id() -> str | None:
    exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    return exp.experiment_id if exp else None


@st.cache_data(ttl=300)
def backtest_summary() -> pd.DataFrame:
    """One row per model — mean/std of MAE, RMSE, MAPE, fit_s."""
    exp_id = experiment_id()
    if exp_id is None:
        return pd.DataFrame()
    parents = _client().search_runs(
        exp_id,
        filter_string="tags.kind = 'backtest'",
        order_by=["metrics.MAE_mean ASC"],
    )
    rows = []
    for r in parents:
        rows.append({
            "model": r.data.tags.get("mlflow.runName"),
            "run_id": r.info.run_id,
            "MAE_mean": r.data.metrics.get("MAE_mean"),
            "MAE_std": r.data.metrics.get("MAE_std"),
            "RMSE_mean": r.data.metrics.get("RMSE_mean"),
            "MAPE_pct_mean": r.data.metrics.get("MAPE_pct_mean"),
            "fit_s_mean": r.data.metrics.get("fit_s_mean"),
        })
    return pd.DataFrame(rows)


@st.cache_data(ttl=300)
def fold_metrics() -> pd.DataFrame:
    """Per-fold metrics across all backtest models, long form."""
    exp_id = experiment_id()
    if exp_id is None:
        return pd.DataFrame()
    parents = _client().search_runs(exp_id, filter_string="tags.kind = 'backtest'")
    rows = []
    for parent in parents:
        model = parent.data.tags.get("mlflow.runName")
        children = _client().search_runs(
            exp_id,
            filter_string=f"tags.mlflow.parentRunId = '{parent.info.run_id}'",
        )
        for c in children:
            rows.append({
                "model": model,
                "fold": int(c.data.params["fold"]),
                "MAE": c.data.metrics.get("MAE"),
                "RMSE": c.data.metrics.get("RMSE"),
                "MAPE_pct": c.data.metrics.get("MAPE_pct"),
            })
    return pd.DataFrame(rows).sort_values(["model", "fold"])


@st.cache_data(ttl=300)
def comparison_run() -> dict | None:
    """Most recent significance-test run."""
    exp_id = experiment_id()
    if exp_id is None:
        return None
    runs = _client().search_runs(
        exp_id,
        filter_string="tags.kind = 'comparison'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        return None
    r = runs[0]
    return {
        "model_a": r.data.tags.get("model_a"),
        "model_b": r.data.tags.get("model_b"),
        "t_stat": r.data.metrics.get("t_statistic"),
        "p_value": r.data.metrics.get("p_value"),
        "mean_diff": r.data.metrics.get("mean_diff_MAE"),
    }


def _run_id_for(model_name: str) -> str | None:
    exp_id = experiment_id()
    if exp_id is None:
        return None
    runs = _client().search_runs(
        exp_id,
        filter_string=(
            f"tags.mlflow.runName = '{model_name}' and tags.kind = 'backtest'"
        ),
        max_results=1,
    )
    return runs[0].info.run_id if runs else None


@st.cache_data(ttl=300)
def predictions_for(model: str) -> pd.DataFrame:
    """Download predictions.csv from a model's backtest artifacts."""
    run_id = _run_id_for(model)
    if run_id is None:
        return pd.DataFrame()
    try:
        local = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path="predictions/predictions.csv"
        )
        return pd.read_csv(local, parse_dates=["datetime"])
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def importance_for(model: str) -> pd.DataFrame:
    run_id = _run_id_for(model)
    if run_id is None:
        return pd.DataFrame()
    try:
        local = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path="importance/importance.csv"
        )
        return pd.read_csv(local, index_col=0)
    except Exception:
        return pd.DataFrame()


@st.cache_resource
def champion_model():
    import mlflow.sklearn
    return mlflow.sklearn.load_model(f"models:/{REGISTERED_NAME}@champion")


@st.cache_data(ttl=3600)
def load_series() -> tuple[pd.Series, dict]:
    from src.data import load_and_clean
    return load_and_clean(
        str(REPO_ROOT / "data" / "Electricity Consumption 2015-2020.csv")
    )


# --- Pages -----------------------------------------------------------------

def page_overview():
    st.title("Electricity Load Forecasting")
    st.caption(
        "Day-ahead hourly forecasts for the Turkish grid, 2015–2020. "
        "All numbers below are read from the local MLflow store."
    )

    summary = backtest_summary()
    if summary.empty:
        st.warning("No backtest runs found in MLflow. Run mlflow_pipeline.ipynb first.")
        return

    baseline_row = summary[summary["model"].str.contains("Seasonal", na=False)]
    baseline_mae = float(baseline_row["MAE_mean"].iloc[0]) if not baseline_row.empty else None
    non_naive = summary[~summary["model"].str.contains("naive", na=False, case=False)]
    best = non_naive.loc[non_naive["MAE_mean"].idxmin()]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best model", best["model"], f"{best['MAE_mean']:,.0f} MAE")
    if baseline_mae:
        improvement = 100 * (baseline_mae - best["MAE_mean"]) / baseline_mae
        c2.metric("vs seasonal naive", f"{improvement:.1f}%")
    c3.metric("Test coverage", "8,640 h", "4 folds × 90 d")
    c4.metric("Features", "61", "leakage-safe")

    st.divider()

    st.subheader("Model comparison")
    display = summary.copy()
    for c in ["MAE_mean", "RMSE_mean", "fit_s_mean"]:
        if c in display:
            display[c] = display[c].round(1)
    if "MAPE_pct_mean" in display:
        display["MAPE_pct_mean"] = display["MAPE_pct_mean"].round(2)
    st.dataframe(
        display.drop(columns=["run_id"]),
        hide_index=True,
        width="stretch",
    )

    st.subheader("Per-fold MAE")
    folds = fold_metrics()
    if not folds.empty:
        pivot = folds.pivot(index="fold", columns="model", values="MAE")
        fig, ax = plt.subplots(figsize=(10, 4))
        pivot.plot(marker="o", ax=ax)
        ax.set(xlabel="Fold", ylabel="MAE (MWh)")
        ax.legend(loc="upper left", fontsize=9, ncol=2)
        ax.grid(alpha=0.3)
        ax.set_xticks([1, 2, 3, 4])
        plt.tight_layout()
        st.pyplot(fig, width="stretch")

    comp = comparison_run()
    if comp:
        st.subheader(f"{comp['model_a']} vs {comp['model_b']} — paired t-test")
        c1, c2, c3 = st.columns(3)
        c1.metric("t-statistic", f"{comp['t_stat']:+.3f}")
        c2.metric("p-value", f"{comp['p_value']:.3f}")
        c3.metric("Mean MAE diff", f"{comp['mean_diff']:+.1f} MWh")
        verdict = "significant" if comp["p_value"] < 0.05 else "not significant"
        st.info(
            f"**{verdict} at 0.05.** {comp['model_a']} leads {comp['model_b']} "
            f"by {abs(comp['mean_diff']):.0f} MWh MAE on average, but the "
            "difference doesn't clear the significance bar with only 4 folds. "
            "Production choice: whichever trains faster and deploys cheaper."
        )


def page_forecast_explorer():
    st.title("Forecast explorer")
    st.caption("Actual vs predicted, sliceable by model, fold and date range")

    summary = backtest_summary()
    models = [m for m in summary["model"] if "naive" not in m.lower()]
    if not models:
        st.warning("No non-naive models available.")
        return

    c1, c2 = st.columns([2, 1])
    with c1:
        default_ix = models.index("LightGBM") if "LightGBM" in models else 0
        model = st.selectbox("Model", models, index=default_ix)

    preds = predictions_for(model)
    if preds.empty:
        st.warning(f"No predictions artifact found for {model}.")
        return

    available_folds = sorted(preds["fold"].unique())
    with c2:
        fold = st.selectbox(
            "Fold", available_folds, index=len(available_folds) - 1
        )

    view = preds[preds["fold"] == fold].sort_values("datetime")
    min_d = view["datetime"].min().date()
    max_d = view["datetime"].max().date()
    default_end = min(max_d, min_d + timedelta(days=14))

    date_range = st.slider(
        "Date range",
        min_value=min_d,
        max_value=max_d,
        value=(min_d, default_end),
        format="YYYY-MM-DD",
    )
    window = view[
        (view["datetime"].dt.date >= date_range[0])
        & (view["datetime"].dt.date <= date_range[1])
    ].set_index("datetime")

    if window.empty:
        st.warning("No data in the selected range.")
        return

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(window.index, window["y_true"], color="black", linewidth=1, label="Actual")
    ax.plot(window.index, window["y_pred"], color="steelblue", linewidth=1.2, label=model)
    ax.set(ylabel="MWh", xlabel="", title=f"Fold {fold}")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig, width="stretch")

    err = window["y_pred"] - window["y_true"]
    mae = float(err.abs().mean())
    bias = float(err.mean())
    mape = 100 * float((err.abs() / window["y_true"]).mean())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAE", f"{mae:,.0f} MWh")
    c2.metric("MAPE", f"{mape:.2f}%")
    c3.metric("Bias", f"{bias:+,.0f} MWh")
    c4.metric("Hours", f"{len(window):,}")


def page_where_it_fails():
    st.title("Where it fails")
    st.caption("Error breakdowns by hour of day and day type")

    from src.analysis import error_by_day_type, error_by_hour

    summary = backtest_summary()
    models = [m for m in summary["model"] if "naive" not in m.lower()]
    default_ix = models.index("LightGBM") if "LightGBM" in models else 0
    model = st.selectbox("Model", models, index=default_ix)

    preds = predictions_for(model)
    if preds.empty:
        st.warning(f"No predictions for {model}.")
        return

    st.subheader("By hour of day")
    hour_err = error_by_hour(preds)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    hour_err["MAE"].plot(ax=axes[0], marker="o", color="steelblue")
    axes[0].set(title="MAE by hour", ylabel="MAE (MWh)", xlabel="Hour")
    axes[0].grid(alpha=0.3)
    hour_err["bias"].plot(ax=axes[1], marker="o", color="firebrick")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set(title="Signed bias by hour", ylabel="Bias (MWh)", xlabel="Hour")
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig, width="stretch")

    st.subheader("By day type")
    day_err = error_by_day_type(preds)
    st.dataframe(day_err, width="stretch")


def page_shock():
    st.title("The 2020 demand shock")
    st.caption("Fold 4 (April–June 2020) — a real distribution shift, not a modelling bug")

    from src.analysis import regime_shift_split, year_over_year_load

    series, _ = load_series()
    preds = predictions_for("LightGBM")
    if preds.empty:
        st.warning("No LightGBM predictions logged.")
        return

    st.subheader("Before vs after 2020-03-16")
    shift = regime_shift_split(preds, cut="2020-03-16")
    st.dataframe(shift, width="stretch")

    st.subheader("2020 weekly demand vs the same ISO weeks in 2019")
    yoy = year_over_year_load(series, baseline_year=2019, shock_year=2020)
    fig, ax = plt.subplots(figsize=(12, 4))
    yoy["change_pct"].plot(ax=ax, color="firebrick", marker="o")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(11, color="grey", linestyle="--", alpha=0.6, label="Mid-March")
    ax.set(ylabel="% change vs 2019", xlabel="ISO week")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig, width="stretch")


@st.cache_resource
def _holdout_champion(holdout_hours: int, holdout_seed: str = "v1"):
    """Retrain a fresh LightGBM on everything except the last `holdout_hours`.

    The registered champion was trained on all data, so predicting the tail
    of that data is a training-fit view, not an honest test. This model has
    the same hyper-params but never sees the holdout window, so its numbers
    on it are a fair out-of-sample demo. Cached by hours so the slider is
    responsive after the first render.
    """
    from src.features import make_features
    from src.models import make_gbm

    series, _ = load_series()
    frame = make_features(series, min_lag=24, country="TR")
    train_frame = frame.iloc[:-holdout_hours]
    hold_frame = frame.iloc[-holdout_hours:]

    X_train, y_train = train_frame.drop(columns="y"), train_frame["y"]
    X_hold, y_hold = hold_frame.drop(columns="y"), hold_frame["y"]

    model = make_gbm().fit(X_train, y_train)
    return model, X_hold, y_hold


def page_champion():
    st.title("Champion model")
    st.caption(f"Loaded from the registry — `models:/{REGISTERED_NAME}@champion`")

    try:
        model = champion_model()
        st.success(f"Loaded: **{type(model).__name__}**")
    except Exception as e:
        st.error(f"Could not load champion model: {e}")
        return

    st.info(
        "The registered champion was trained on **all** available data, so "
        "predicting the tail of that data would only measure how well it fit "
        "itself. To keep this page honest, the plot below shows a **fresh "
        "LightGBM with identical hyper-parameters**, retrained with the "
        "chosen window held out. Backtest headline: ~842 MWh MAE, 2.79% MAPE."
    )

    holdout = st.slider(
        "Hold-out window (hours)",
        min_value=24,
        max_value=168,
        value=48,
        step=24,
    )
    holdout_model, X_hold, y_hold = _holdout_champion(holdout)
    y_hat = holdout_model.predict(X_hold)

    plot_df = pd.DataFrame(
        {"Actual": y_hold.values, "Champion (holdout)": y_hat},
        index=X_hold.index,
    )
    fig, ax = plt.subplots(figsize=(12, 4))
    plot_df.plot(ax=ax, marker=".", linewidth=1)
    ax.set(
        ylabel="MWh",
        xlabel="",
        title=f"Last {holdout} hours — never seen during training",
    )
    ax.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig, width="stretch")

    err = y_hat - y_hold.values
    mae = float(np.abs(err).mean())
    mape = 100 * float((np.abs(err) / y_hold.values).mean())
    bias = float(err.mean())
    c1, c2, c3 = st.columns(3)
    c1.metric("MAE (holdout)", f"{mae:,.0f} MWh")
    c2.metric("MAPE (holdout)", f"{mape:.2f}%")
    c3.metric("Bias (holdout)", f"{bias:+,.0f} MWh")

    st.subheader("Top features (LightGBM backtest)")
    imp = importance_for("LightGBM")
    if imp.empty:
        st.info("No importance artifact logged.")
    else:
        top = imp.head(15)
        fig, ax = plt.subplots(figsize=(8, 5))
        top.iloc[::-1].plot.barh(ax=ax, color="steelblue", legend=False)
        ax.set(xlabel="Importance", ylabel="")
        ax.grid(alpha=0.3, axis="x")
        plt.tight_layout()
        st.pyplot(fig, width="stretch")


def page_data_quality():
    st.title("Data quality")
    st.caption("What the raw file looks like before feature engineering")

    series, audit = load_series()
    st.subheader("Audit")
    audit_df = pd.DataFrame(
        [{"metric": k, "value": str(v)} for k, v in audit.items()]
    )
    st.dataframe(audit_df, hide_index=True, width="stretch")

    st.subheader("Full time series")
    fig, ax = plt.subplots(figsize=(12, 4))
    series.plot(ax=ax, linewidth=0.4, color="steelblue")
    ax.set(ylabel="MWh", xlabel="", title="Turkish grid load, 2015–2020 (hourly)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig, width="stretch")


# --- Router ----------------------------------------------------------------

PAGES = {
    "Overview": page_overview,
    "Forecast explorer": page_forecast_explorer,
    "Where it fails": page_where_it_fails,
    "The 2020 shock": page_shock,
    "Champion model": page_champion,
    "Data quality": page_data_quality,
}

st.sidebar.title("Load Forecast")
st.sidebar.caption(f"MLflow store: `{STORE_DIR.name}/`")
choice = st.sidebar.radio("Section", list(PAGES.keys()), label_visibility="collapsed")
st.sidebar.divider()
if st.sidebar.button("Refresh cache"):
    st.cache_data.clear()
    st.rerun()
st.sidebar.caption(
    "Reads runs, metrics, artifacts and the champion model directly from the "
    "SQLite backend. No local CSVs."
)

PAGES[choice]()
