#!/usr/bin/env python
"""
app.py - Flask API for the AAVAIL revenue forecast.

Endpoints
---------
GET  /                      health check + list of countries
POST /train                 {"mode": "test"|"prod"}  -> retrain models
GET|POST /predict           country, year, month, day, [mode]  -> 30-day revenue forecast
GET  /monitor               [country] -> production performance + drift report
GET  /logs/<filename>       download a log file

    python app.py            # serves on 0.0.0.0:8080
"""

import os
import re
import argparse

from flask import Flask, jsonify, request, send_from_directory

import cslib
from logger import LOG_DIR
from model import model_train, model_load, model_predict

app = Flask(__name__)
_cache = {}


def _load(prefix):
    """load models + data once per prefix and keep them in memory"""
    if prefix not in _cache:
        _cache[prefix] = model_load(prefix=prefix)
    return _cache[prefix]


def _params():
    if request.method == "POST" and request.is_json:
        return request.get_json(silent=True) or {}
    return request.args.to_dict()


@app.route("/", methods=["GET"])
def index():
    try:
        _, models = _load("sl")
        countries = sorted(models)
    except FileNotFoundError:
        countries = []
    return jsonify({"status": "ok", "service": "aavail-revenue-forecast", "countries": countries})


@app.route("/train", methods=["POST"])
def train():
    params = _params()
    test = str(params.get("mode", "test")).lower() == "test"
    results = model_train(test=test)
    _cache.pop("test" if test else "sl", None)
    best = results[results["selected"]][["country", "model", "rmse"]]
    return jsonify({"status": "trained", "mode": "test" if test else "prod",
                    "selected_models": best.round(2).to_dict(orient="records")})


@app.route("/predict", methods=["GET", "POST"])
def predict():
    params = _params()
    missing = [k for k in ("country", "year", "month", "day") if k not in params]
    if missing:
        return jsonify({"error": f"missing parameters: {missing}"}), 400

    test = str(params.get("mode", "prod")).lower() == "test"
    try:
        all_data, all_models = _load("test" if test else "sl")
        result = model_predict(params["country"], params["year"], params["month"], params["day"],
                               all_models=all_models, all_data=all_data, test=test)
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(result)


@app.route("/monitor", methods=["GET"])
def monitor():
    from monitoring import run_monitoring
    country = request.args.get("country")
    countries = [cslib.country_key(country)] if country else None
    try:
        report = run_monitoring(countries, plot=False)
    except (ValueError, KeyError) as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(report.to_dict(orient="records"))


@app.route("/logs/<filename>", methods=["GET"])
def logs(filename):
    if not re.fullmatch(r"(train|predict)-[\w\-]+\.log", filename):
        return jsonify({"error": "invalid log file name"}), 400
    if not os.path.exists(os.path.join(LOG_DIR, filename)):
        return jsonify({"error": f"log file not found: {filename}"}), 404
    return send_from_directory(LOG_DIR, filename, as_attachment=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--debug", action="store_true")
    ap.add_argument("-p", "--port", type=int, default=int(os.environ.get("PORT", 8080)))
    args = ap.parse_args()
    app.run(host="0.0.0.0", port=args.port, debug=args.debug)
