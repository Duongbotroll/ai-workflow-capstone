#!/usr/bin/env python
"""
logger.py - training and prediction logs (CSV format, one file per month).

Production logs:  logs/train-YYYY-M.log,      logs/predict-YYYY-M.log
Test logs:        logs/train-test.log,        logs/predict-test.log

Unit tests always pass test=True so they never touch the production logs.
"""

import os
import csv
import time
import uuid
from datetime import date

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.environ.get("AAVAIL_LOG_DIR", os.path.join(THIS_DIR, "logs"))

TRAIN_HEADER = ["unique_id", "timestamp", "country", "data_shape", "eval_test",
                "runtime", "model_version", "model_version_note", "test"]
PREDICT_HEADER = ["unique_id", "timestamp", "country", "target_date", "y_pred",
                  "runtime", "model_version", "test"]


def _log_path(kind, test):
    os.makedirs(LOG_DIR, exist_ok=True)
    if test:
        return os.path.join(LOG_DIR, f"{kind}-test.log")
    today = date.today()
    return os.path.join(LOG_DIR, f"{kind}-{today.year}-{today.month}.log")


def _write(path, header, row):
    new_file = not os.path.exists(path)
    with open(path, "a", newline="") as fh:
        writer = csv.writer(fh, delimiter=",", quoting=csv.QUOTE_MINIMAL)
        if new_file:
            writer.writerow(header)
        writer.writerow(row)


def update_train_log(country, data_shape, eval_test, runtime,
                     model_version, model_version_note, test=False):
    """append one line to the training log"""
    path = _log_path("train", test)
    row = [str(uuid.uuid4()), time.time(), country, data_shape, eval_test,
           runtime, model_version, model_version_note, test]
    _write(path, TRAIN_HEADER, row)
    return path


def update_predict_log(country, target_date, y_pred, runtime, model_version, test=False):
    """append one line to the prediction log"""
    path = _log_path("predict", test)
    row = [str(uuid.uuid4()), time.time(), country, target_date, y_pred,
           runtime, model_version, test]
    _write(path, PREDICT_HEADER, row)
    return path
