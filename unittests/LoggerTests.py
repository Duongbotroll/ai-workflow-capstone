#!/usr/bin/env python
"""logger tests - write to a temporary directory, never to the production logs"""

import os
import csv
import sys
import shutil
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logger  # noqa: E402
from logger import update_train_log, update_predict_log, TRAIN_HEADER, PREDICT_HEADER  # noqa: E402


class LoggerTest(unittest.TestCase):

    def setUp(self):
        self._orig = logger.LOG_DIR
        self.tmp = tempfile.mkdtemp()
        logger.LOG_DIR = self.tmp

    def tearDown(self):
        logger.LOG_DIR = self._orig
        shutil.rmtree(self.tmp, ignore_errors=True)

    @staticmethod
    def _read(path):
        with open(path) as fh:
            return list(csv.reader(fh))

    def test_01_train_log_created(self):
        path = update_train_log("all", "(100, 7)", {"rmse": 0.5}, "1.0s", 0.1, "note", test=True)
        self.assertTrue(os.path.exists(path))
        self.assertEqual(os.path.basename(path), "train-test.log")

    def test_02_train_log_content(self):
        path = update_train_log("all", "(100, 7)", {"rmse": 0.5}, "1.0s", 0.1, "note", test=True)
        rows = self._read(path)
        self.assertEqual(rows[0], TRAIN_HEADER)
        self.assertEqual(rows[-1][2], "all")
        self.assertIn("rmse", rows[-1][4])

    def test_03_predict_log_created(self):
        path = update_predict_log("all", "2018-01-05", 1234.5, "0.01s", 0.1, test=True)
        self.assertTrue(os.path.exists(path))
        self.assertEqual(os.path.basename(path), "predict-test.log")

    def test_04_predict_log_content_appends(self):
        update_predict_log("all", "2018-01-05", 1.0, "0.01s", 0.1, test=True)
        path = update_predict_log("france", "2018-01-06", 2.0, "0.01s", 0.1, test=True)
        rows = self._read(path)
        self.assertEqual(rows[0], PREDICT_HEADER)
        self.assertEqual(len(rows), 3)  # header + 2 entries
        self.assertEqual(rows[-1][2:5], ["france", "2018-01-06", "2.0"])

    def test_05_production_log_separate(self):
        test_path = update_predict_log("all", "2018-01-05", 1.0, "0.01s", 0.1, test=True)
        prod_path = update_predict_log("all", "2018-01-05", 1.0, "0.01s", 0.1, test=False)
        self.assertNotEqual(test_path, prod_path)
        self.assertNotIn("test", os.path.basename(prod_path))


if __name__ == "__main__":
    unittest.main()
