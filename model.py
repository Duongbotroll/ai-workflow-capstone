#!/usr/bin/env python
"""
model.py - train, compare, save, load and predict.

For every country (top 10 by revenue + 'all') several regressors are compared
against a naive baseline on a time-ordered hold-out set.  The best model per
country is refit on all the data and saved to models/.

    python model.py            # full training (production models, prefix 'sl')
    python model.py --test     # quick training on a subset (prefix 'test')
"""

import os
import re
import time
import argparse
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import cslib
from logger import update_train_log, update_predict_log
from estimators import NaiveBaseline, BaselineResidual

warnings.filterwarnings("ignore", category=UserWarning)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.environ.get("AAVAIL_MODEL_DIR", os.path.join(THIS_DIR, "models"))
REPORT_DIR = os.path.join(THIS_DIR, "reports")

MODEL_VERSION = 0.1
MODEL_VERSION_NOTE = "best of baseline / Ridge / RandomForest / GradientBoosting (residual on baseline) per country"
TEST_COUNTRIES = ["all", "united_kingdom"]


def _candidates(test=False):
    """estimators and parameter grids to compare"""
    if test:
        grids = {
            "ridge": (Pipeline([("scale", StandardScaler()), ("reg", Ridge())]),
                      {"reg__alpha": [10.0]}),
            "random_forest": (RandomForestRegressor(random_state=42, n_jobs=-1),
                              {"n_estimators": [20], "max_depth": [5]}),
        }
    else:
        grids = {
            "ridge": (Pipeline([("scale", StandardScaler()), ("reg", Ridge())]),
                      {"reg__alpha": [0.1, 1.0, 10.0, 100.0, 1000.0]}),
            "random_forest": (RandomForestRegressor(random_state=42, n_jobs=-1),
                              {"n_estimators": [100, 200], "max_depth": [4, 8],
                               "min_samples_leaf": [1, 5]}),
            "gradient_boosting": (GradientBoostingRegressor(random_state=42),
                                  {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1],
                                   "max_depth": [2, 3]}),
        }
    # every candidate learns the residual on top of the naive baseline
    return {name: (BaselineResidual(est), {f"estimator__{k}": v for k, v in grid.items()})
            for name, (est, grid) in grids.items()}


def _time_split(n, test_frac=0.2, gap=30):
    """indices for a time-ordered split with a gap so target windows do not overlap"""
    n_test = int(n * test_frac)
    test_idx = np.arange(n - n_test, n)
    train_idx = np.arange(0, max(n - n_test - gap, 1))
    return train_idx, test_idx


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _model_path(prefix, key, version=MODEL_VERSION):
    version_str = re.sub(r"\.", "_", str(version))
    return os.path.join(MODEL_DIR, f"{prefix}-{key}-{version_str}.joblib")


def _train_one(key, df, test=False):
    """compare candidates for one country, refit the best, save it, return comparison rows"""
    start = time.time()
    X, y, dates = cslib.engineer_features(df, training=True)

    if test:  # use only the most recent ~60% of the data to keep tests fast
        cut = int(len(X) * 0.4)
        X, y, dates = X.iloc[cut:].reset_index(drop=True), y[cut:], dates[cut:]

    tr, te = _time_split(len(X))
    X_tr, y_tr, X_te, y_te = X.iloc[tr], y[tr], X.iloc[te], y[te]
    cv = TimeSeriesSplit(n_splits=3 if test else 4, gap=30)

    rows, fitted = [], {}
    base = NaiveBaseline().fit(X_tr, y_tr)
    p = base.predict(X_te)
    rows.append({"country": key, "model": "baseline", "rmse": rmse(y_te, p),
                 "mae": float(mean_absolute_error(y_te, p)), "params": "{}"})
    fitted["baseline"] = NaiveBaseline()

    for name, (est, grid) in _candidates(test).items():
        gs = GridSearchCV(est, grid, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1)
        gs.fit(X_tr, y_tr)
        p = gs.predict(X_te)
        rows.append({"country": key, "model": name, "rmse": rmse(y_te, p),
                     "mae": float(mean_absolute_error(y_te, p)), "params": str({k.replace("estimator__", ""): v
                                                  for k, v in gs.best_params_.items()})})
        fitted[name] = gs.best_estimator_

    best = min(rows, key=lambda r: r["rmse"])
    final = fitted[best["model"]]
    final.fit(X, y)  # refit on all available data

    prefix = "test" if test else "sl"
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump({"model": final, "name": best["model"], "features": list(X.columns),
                 "version": MODEL_VERSION, "train_end": dates[-1]},
                _model_path(prefix, key))

    for r in rows:
        r["selected"] = r["model"] == best["model"]

    runtime = f"{time.time() - start:.1f}s"
    update_train_log(key, str(X.shape), {"model": best["model"], "rmse": round(best["rmse"], 2)},
                     runtime, MODEL_VERSION, MODEL_VERSION_NOTE, test=test)
    return rows


def model_train(data_dir=cslib.DATA_DIR, test=False, countries=None):
    """
    train models for all countries (or a subset) and write the comparison table.
    test=True trains small models on a subset of countries, prefixed 'test-'.
    """
    ts_data = cslib.fetch_ts(data_dir)
    if countries is None:
        countries = TEST_COUNTRIES if test else list(ts_data.keys())

    rows = []
    for key in countries:
        key = cslib.country_key(key)
        if key not in ts_data:
            raise ValueError(f"unknown country '{key}'")
        rows.extend(_train_one(key, ts_data[key], test=test))

    results = pd.DataFrame(rows)
    out_dir = MODEL_DIR if test else REPORT_DIR
    os.makedirs(out_dir, exist_ok=True)
    fname = "test-model-comparison.csv" if test else "model-comparison.csv"
    results.to_csv(os.path.join(out_dir, fname), index=False)
    if not test:
        plot_model_comparison(results)
    return results


def plot_model_comparison(results, out_path=None):
    """bar chart: hold-out RMSE of every model relative to the baseline, per country"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = out_path or os.path.join(REPORT_DIR, "figures", "model_comparison.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    pivot = results.pivot(index="country", columns="model", values="rmse")
    rel = pivot.div(pivot["baseline"], axis=0)
    order = ["baseline"] + [c for c in rel.columns if c != "baseline"]
    rel = rel[order].loc[pivot["baseline"].sort_values(ascending=False).index]

    colors = {"baseline": "#9e9e9e", "ridge": "#2a78d6",
              "random_forest": "#eb6834", "gradient_boosting": "#1baf7a"}
    fig, ax = plt.subplots(figsize=(11, 5))
    width = 0.8 / len(order)
    x = np.arange(len(rel))
    for i, m in enumerate(order):
        ax.bar(x + i * width - 0.4 + width / 2, rel[m], width, label=m, color=colors.get(m))
    ax.axhline(1.0, color="#333", lw=1, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(rel.index, rotation=30, ha="right")
    ax.set_ylabel("hold-out RMSE / baseline RMSE  (lower is better)")
    ax.set_title("Model comparison vs naive baseline (30-day revenue forecast)")
    ax.legend(frameon=False, ncol=len(order))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def model_load(prefix="sl", data_dir=None, use_production=True):
    """
    load saved models and the feature matrices needed for prediction.
    By default training + production data are combined so that the most recent
    dates can be predicted.
    """
    if not os.path.isdir(MODEL_DIR):
        raise FileNotFoundError("model directory missing - train the models first")
    files = [f for f in os.listdir(MODEL_DIR) if f.startswith(prefix + "-") and f.endswith(".joblib")]
    if not files:
        raise FileNotFoundError(f"no '{prefix}' models found - train the models first")

    all_models = {}
    for f in files:
        key = re.sub(rf"^{prefix}-|-\d+_\d+\.joblib$", "", f)
        all_models[key] = joblib.load(os.path.join(MODEL_DIR, f))

    if data_dir is None and use_production and os.path.isdir(cslib.PROD_DATA_DIR):
        ts_data = cslib.fetch_combined_ts()
    else:
        ts_data = cslib.fetch_ts(data_dir or cslib.DATA_DIR)

    all_data = {}
    for key, df in ts_data.items():
        X, y, dates = cslib.engineer_features(df, training=False)
        all_data[key] = {"X": X, "y": y, "dates": dates}
    return all_data, all_models


def model_predict(country, year, month, day, all_models=None, all_data=None, test=False):
    """predict revenue for the 30 days following the given date"""
    start = time.time()
    if all_models is None or all_data is None:
        all_data, all_models = model_load(prefix="test" if test else "sl")

    key = cslib.country_key(country)
    if key not in all_models:
        raise ValueError(f"model for country '{country}' not available. "
                         f"Available: {sorted(all_models)}")

    target_date = f"{int(year)}-{int(month):02d}-{int(day):02d}"
    try:
        pd.Timestamp(target_date)
    except ValueError:
        raise ValueError(f"invalid date {target_date}")

    data = all_data[key]
    dates = data["dates"]
    if target_date not in dates:
        raise ValueError(f"date {target_date} not in range {dates[0]} - {dates[-1]}")

    idx = int(np.where(dates == target_date)[0][0])
    payload = all_models[key]
    query = data["X"].iloc[[idx]][payload["features"]]
    y_pred = float(payload["model"].predict(query)[0])

    runtime = f"{time.time() - start:.3f}s"
    update_predict_log(key, target_date, round(y_pred, 2), runtime, payload["version"], test=test)
    return {"country": key, "target_date": target_date, "y_pred": round(y_pred, 2),
            "model": payload["name"], "model_version": payload["version"]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="train AAVAIL revenue models")
    ap.add_argument("--test", action="store_true", help="quick training for tests")
    args = ap.parse_args()

    t0 = time.time()
    res = model_train(test=args.test)
    print(res.round(1).to_string(index=False))
    print(f"\ntraining finished in {time.time() - t0:.1f}s")

    all_data, all_models = model_load(prefix="test" if args.test else "sl")
    print(model_predict("all", 2018, 1, 5, all_models, all_data, test=args.test))
