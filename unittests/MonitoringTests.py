#!/usr/bin/env python
"""monitoring tests - run against the isolated 'test-' models"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cslib  # noqa: E402
from model import model_train, model_load  # noqa: E402
from monitoring import get_monitoring_tools, monitor_country, run_monitoring  # noqa: E402


class MonitoringTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        model_train(test=True)
        cls.all_data, cls.all_models = model_load(prefix="test")
        cls.train_ts = cslib.fetch_ts(cslib.DATA_DIR)

    def test_01_monitoring_tools(self):
        X = self.all_data["all"]["X"].iloc[:300]
        y = self.all_data["all"]["y"][:300]
        tools = get_monitoring_tools(X, y, n_bootstrap=20)
        self.assertIn("outlier_model", tools)
        self.assertGreater(tools["wasserstein_threshold"], 0)

    def test_02_monitor_country(self):
        report, series = monitor_country("all", self.all_data, self.all_models, self.train_ts,
                                         holdout_rmse=1.0)
        for key in ("rmse", "mae", "baseline_rmse", "outlier_rate", "drift_alert", "performance_alert"):
            self.assertIn(key, report)
        self.assertGreater(report["n_production_days"], 0)
        self.assertTrue(np.isfinite(series["predicted"]).all())
        self.assertTrue(report["performance_alert"])  # rmse is far above 1.0

    def test_03_run_monitoring_test_prefix(self):
        df = run_monitoring(prefix="test", plot=False)
        self.assertEqual(set(df["country"]), set(self.all_models))


if __name__ == "__main__":
    unittest.main()
