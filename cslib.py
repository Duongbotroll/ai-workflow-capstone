#!/usr/bin/env python
"""
cslib.py - data ingestion and feature engineering for the AAVAIL revenue capstone.

Functions
---------
fetch_data(data_dir)          -> raw invoice DataFrame (handles inconsistent JSON schemas)
convert_to_ts(df, country)    -> daily time series (purchases, invoices, streams, views, revenue)
fetch_ts(data_dir, clean)     -> dict {country: daily ts} for top-10 countries + 'all' (cached as CSV)
engineer_features(df, training) -> X, y, dates  (target = revenue over the next 30 days)

Run as a script to (re)build the cached time series:
    python cslib.py --data-dir data/cs-train
"""

import os
import re
import json
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(THIS_DIR, "data", "cs-train")
PROD_DATA_DIR = os.path.join(THIS_DIR, "data", "cs-production")

REQUIRED_COLUMNS = ["country", "customer_id", "day", "invoice", "month",
                    "price", "stream_id", "times_viewed", "year"]
COLUMN_RENAMES = {"StreamID": "stream_id",
                  "TimesViewed": "times_viewed",
                  "total_price": "price"}
TOP_N_COUNTRIES = 10


def country_key(country):
    """normalise a country name into the key used for files and models"""
    return re.sub(r"\s+", "_", str(country).strip().lower())


def fetch_data(data_dir=DATA_DIR):
    """
    load all invoice JSON files in data_dir into a single, cleaned DataFrame.
    The source files use inconsistent column names - they are harmonised here.
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"data directory does not exist: {data_dir}")

    files = sorted(f for f in os.listdir(data_dir) if f.endswith(".json"))
    if not files:
        raise ValueError(f"no json files found in {data_dir}")

    frames = []
    for fname in files:
        df = pd.read_json(os.path.join(data_dir, fname), dtype=False)
        df = df.rename(columns=COLUMN_RENAMES)
        missing = set(REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"{fname} is missing columns: {sorted(missing)}")
        frames.append(df[REQUIRED_COLUMNS])

    df = pd.concat(frames, ignore_index=True)

    # types and derived fields
    for col in ["year", "month", "day"]:
        df[col] = df[col].astype(int)
    df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0.0)
    df["times_viewed"] = pd.to_numeric(df["times_viewed"], errors="coerce").fillna(0).astype(int)
    df["invoice"] = df["invoice"].astype(str).str.replace(r"\D+", "", regex=True)
    df["stream_id"] = df["stream_id"].astype(str)
    df["invoice_date"] = pd.to_datetime(dict(year=df["year"], month=df["month"], day=df["day"]))

    return df.sort_values("invoice_date").reset_index(drop=True)


def convert_to_ts(df_orig, country=None):
    """
    aggregate the invoice data into a continuous daily time series.
    country=None aggregates all countries.
    """
    if country is not None and country != "all":
        if country not in set(df_orig["country"]):
            raise ValueError(f"country not found: {country}")
        df = df_orig[df_orig["country"] == country]
    else:
        df = df_orig

    start = df_orig["invoice_date"].min()
    end = df_orig["invoice_date"].max()
    days = pd.date_range(start, end, freq="D")

    grouped = df.groupby("invoice_date")
    ts = pd.DataFrame({
        "purchases": grouped.size(),
        "unique_invoices": grouped["invoice"].nunique(),
        "unique_streams": grouped["stream_id"].nunique(),
        "total_views": grouped["times_viewed"].sum(),
        "revenue": grouped["price"].sum(),
    }).reindex(days, fill_value=0)

    ts.index.name = "date"
    ts = ts.reset_index()
    ts["year_month"] = ts["date"].dt.strftime("%Y-%m")
    return ts


def top_countries(df, n=TOP_N_COUNTRIES):
    """the n countries with the most revenue"""
    return df.groupby("country")["price"].sum().sort_values(ascending=False).index[:n].tolist()


def fetch_ts(data_dir=DATA_DIR, clean=False, countries=None):
    """
    return {country_key: daily ts}.  Results are cached in <data_dir>/ts-data
    so the (slow) JSON parsing only happens once.
    """
    ts_dir = os.path.join(data_dir, "ts-data")

    if clean and os.path.isdir(ts_dir):
        for f in os.listdir(ts_dir):
            os.remove(os.path.join(ts_dir, f))

    if os.path.isdir(ts_dir) and any(f.endswith(".csv") for f in os.listdir(ts_dir)):
        out = {}
        for f in sorted(os.listdir(ts_dir)):
            if f.startswith("ts-") and f.endswith(".csv"):
                key = f[3:-4]
                out[key] = pd.read_csv(os.path.join(ts_dir, f), parse_dates=["date"])
        return out

    df = fetch_data(data_dir)
    if countries is None:
        countries = top_countries(df)
    countries = [c for c in countries if c in set(df["country"])]

    os.makedirs(ts_dir, exist_ok=True)
    out = {"all": convert_to_ts(df)}
    for c in countries:
        out[country_key(c)] = convert_to_ts(df, country=c)
    for key, ts in out.items():
        ts.to_csv(os.path.join(ts_dir, f"ts-{key}.csv"), index=False)
    with open(os.path.join(ts_dir, "countries.json"), "w") as fh:
        json.dump({country_key(c): c for c in countries}, fh, indent=2)
    return out


def fetch_combined_ts(train_dir=DATA_DIR, prod_dir=PROD_DATA_DIR):
    """
    training + production data concatenated into one continuous series per country.
    Used for monitoring and for predicting dates beyond the training window.
    """
    train = fetch_ts(train_dir)
    with open(os.path.join(train_dir, "ts-data", "countries.json")) as fh:
        names = list(json.load(fh).values())
    prod = fetch_ts(prod_dir, countries=names)
    out = {}
    for key, ts in train.items():
        if key in prod:
            both = pd.concat([ts, prod[key]], ignore_index=True)
            both = both.groupby("date", as_index=False).agg({
                "purchases": "sum", "unique_invoices": "sum", "unique_streams": "sum",
                "total_views": "sum", "revenue": "sum"})
            days = pd.date_range(both["date"].min(), both["date"].max(), freq="D")
            both = both.set_index("date").reindex(days, fill_value=0)
            both.index.name = "date"
            both = both.reset_index()
            both["year_month"] = both["date"].dt.strftime("%Y-%m")
            out[key] = both
        else:
            out[key] = ts
    return out


def engineer_features(df, training=True):
    """
    build a supervised dataset from a daily time series.

    For every day t:
      features: revenue over the previous 7/14/28/35/54 days and mean
                invoices & views over the previous 30 days
                (a "same window last year" feature was tried and dropped: with
                under two years of history it is zero for most training rows)
      target:   revenue over the next 30 days (t+1 ... t+30)

    With training=True the last 30 days are dropped because their target window
    is incomplete.
    """
    df = df.sort_values("date").reset_index(drop=True)
    rev = df["revenue"].astype(float)

    feats = pd.DataFrame(index=df.index)
    for k in [7, 14, 28, 35, 54]:
        feats[f"previous_{k}"] = rev.rolling(k, min_periods=1).sum()

    # target: revenue over the next 30 days (t+1 ... t+30)
    next30 = rev[::-1].rolling(30, min_periods=1).sum()[::-1].shift(-1)

    feats["recent_invoices"] = df["unique_invoices"].rolling(30, min_periods=1).mean()
    feats["recent_views"] = df["total_views"].rolling(30, min_periods=1).mean()

    y = next30.fillna(0.0).values
    dates = df["date"].dt.strftime("%Y-%m-%d").values

    if training:
        keep = np.arange(len(df)) < len(df) - 30
        feats, y, dates = feats[keep], y[keep], dates[keep]

    return feats.reset_index(drop=True), y, dates


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ingest invoice data and build cached time series")
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--clean", action="store_true", help="rebuild the cache")
    args = ap.parse_args()

    ts_all = fetch_ts(args.data_dir, clean=args.clean)
    for k, v in ts_all.items():
        print(f"{k:<20} {v.shape[0]} days  revenue={v['revenue'].sum():,.0f}")
