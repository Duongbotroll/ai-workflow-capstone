#!/usr/bin/env python
"""
monitoring.py - post-deployment performance monitoring.

Two checks are run for every country:

1. Model performance on production data
   cs-production (Aug-Dec 2019) was never seen in training.  For every day whose
   30-day target window is complete we compare the deployed model's forecast
   with the actual revenue, and report RMSE / MAE / MAPE next to the naive
   baseline.  A model is flagged when its error is worse than the training
   hold-out error by more than `tolerance`.

2. Input drift
   - an IsolationForest fitted on the training features gives the share of
     production feature rows that look like outliers
   - the Wasserstein distance between the training and production distribution
     of the target (30-day revenue) is compared with bootstrap samples of the
     training data to decide whether the shift is unusual

    python monitoring.py                 # all countries, writes reports/monitoring-report.csv
    python monitoring.py --country all   # one country + plot
"""

import os
import argparse

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_error

import cslib
from model import model_load, rmse, NaiveBaseline, REPORT_DIR



def _split_train_prod(key, all_data, train_ts):
    """split the combined feature matrix into training-period and production-period rows"""
    data = all_data[key]
    dates = pd.to_datetime(data["dates"])
    train_end = train_ts[key]["date"].max()
    last_complete = dates.max() - pd.Timedelta(days=30)  # target window must be complete

    train_mask = np.asarray(dates <= train_end - pd.Timedelta(days=30))
    prod_mask = np.asarray((dates > train_end) & (dates <= last_complete))
    X, y = data["X"], data["y"]
    return (X[train_mask], y[train_mask]), (X[prod_mask], y[prod_mask], data["dates"][prod_mask])


def get_monitoring_tools(X_train, y_train, n_bootstrap=200, random_state=42):
    """fit the drift detectors on training data"""
    iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=random_state)
    iso.fit(X_train)

    rng = np.random.default_rng(random_state)
    n = len(y_train)
    dists = []
    for _ in range(n_bootstrap):
        sample = rng.choice(y_train, size=n // 2, replace=True)
        dists.append(wasserstein_distance(y_train, sample))
    wasserstein_threshold = float(np.percentile(dists, 95))
    return {"outlier_model": iso, "wasserstein_threshold": wasserstein_threshold}


def monitor_country(key, all_data, all_models, train_ts, tolerance=0.25, holdout_rmse=None):
    """compute performance and drift metrics for one country"""
    (X_tr, y_tr), (X_pr, y_pr, d_pr) = _split_train_prod(key, all_data, train_ts)
    if len(y_pr) == 0:
        raise ValueError("no production rows with a complete 30-day target")

    payload = all_models[key]
    cols = payload["features"]
    y_hat = payload["model"].predict(X_pr[cols])
    y_base = NaiveBaseline().predict(X_pr)

    tools = get_monitoring_tools(X_tr[cols], y_tr)
    outlier_rate = float((tools["outlier_model"].predict(X_pr[cols]) == -1).mean())
    w_dist = float(wasserstein_distance(y_tr, y_pr))

    mask = y_pr > 0
    mape = float(np.mean(np.abs(y_pr[mask] - y_hat[mask]) / y_pr[mask]) * 100) if mask.any() else np.nan
    prod_rmse = rmse(y_pr, y_hat)

    report = {
        "country": key,
        "model": payload["name"],
        "n_production_days": int(len(y_pr)),
        "rmse": round(prod_rmse, 2),
        "mae": round(float(mean_absolute_error(y_pr, y_hat)), 2),
        "mape_pct": round(mape, 2),
        "baseline_rmse": round(rmse(y_pr, y_base), 2),
        "holdout_rmse": holdout_rmse,
        "outlier_rate": round(outlier_rate, 3),
        "wasserstein": round(w_dist, 2),
        "wasserstein_threshold": round(tools["wasserstein_threshold"], 2),
    }
    report["drift_alert"] = bool(w_dist > tools["wasserstein_threshold"] or outlier_rate > 0.2)
    report["performance_alert"] = bool(holdout_rmse is not None
                                       and prod_rmse > holdout_rmse * (1 + tolerance))
    series = pd.DataFrame({"date": pd.to_datetime(d_pr), "actual": y_pr,
                           "predicted": y_hat, "baseline": y_base})
    return report, series


def plot_production(series, key, out_path=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = out_path or os.path.join(REPORT_DIR, "figures", f"production_{key}.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(series["date"], series["actual"], color="#222", lw=2, label="actual")
    ax.plot(series["date"], series["predicted"], color="#2a78d6", lw=2, label="model")
    ax.plot(series["date"], series["baseline"], color="#9e9e9e", lw=1.5, ls="--", label="baseline")
    ax.set_title(f"Production monitoring – {key}: 30-day revenue forecast vs actual")
    ax.set_ylabel("revenue next 30 days")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def run_monitoring(countries=None, prefix="sl", plot=True):
    all_data, all_models = model_load(prefix=prefix)
    train_ts = cslib.fetch_ts(cslib.DATA_DIR)

    holdout = {}
    comp_path = os.path.join(REPORT_DIR, "model-comparison.csv")
    if os.path.exists(comp_path):
        comp = pd.read_csv(comp_path)
        holdout = comp[comp["selected"]].set_index("country")["rmse"].round(2).to_dict()

    countries = countries or sorted(all_models)
    reports = []
    for key in countries:
        rep, series = monitor_country(key, all_data, all_models, train_ts,
                                      holdout_rmse=holdout.get(key))
        reports.append(rep)
        if plot and (key in ("all", "united_kingdom") or len(countries) == 1):
            plot_production(series, key)

    df = pd.DataFrame(reports)
    os.makedirs(REPORT_DIR, exist_ok=True)
    if prefix == "sl":
        df.to_csv(os.path.join(REPORT_DIR, "monitoring-report.csv"), index=False)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="monitor deployed models on production data")
    ap.add_argument("--country", default=None)
    args = ap.parse_args()
    countries = [cslib.country_key(args.country)] if args.country else None
    print(run_monitoring(countries).to_string(index=False))
