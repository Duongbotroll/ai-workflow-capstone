#!/usr/bin/env python
"""model tests - train/load/predict with isolated 'test-' models"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import model  # noqa: E402
from model import model_train, model_load, model_predict, MODEL_DIR  # noqa: E402


class ModelTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.results = model_train(test=True)

    def test_01_train_saves_test_models(self):
        for key in model.TEST_COUNTRIES:
            path = os.path.join(MODEL_DIR, f"test-{key}-0_1.joblib")
            self.assertTrue(os.path.exists(path), path)

    def test_02_multiple_models_compared(self):
        models = set(self.results["model"])
        self.assertIn("baseline", models)
        self.assertGreaterEqual(len(models), 3)
        # exactly one selected model per country
        self.assertTrue((self.results.groupby("country")["selected"].sum() == 1).all())

    def test_03_load(self):
        all_data, all_models = model_load(prefix="test")
        self.assertIn("all", all_models)
        for payload in all_models.values():
            self.assertTrue(hasattr(payload["model"], "predict"))
        self.assertIn("X", all_data["all"])

    def test_04_predict_all_countries(self):
        res = model_predict("all", 2018, 1, 5, test=True)
        self.assertEqual(res["country"], "all")
        self.assertIsInstance(res["y_pred"], float)
        self.assertGreater(res["y_pred"], 0)

    def test_05_predict_single_country(self):
        res = model_predict("United Kingdom", "2019", "3", "15", test=True)
        self.assertEqual(res["country"], "united_kingdom")
        self.assertGreater(res["y_pred"], 0)

    def test_06_predict_production_period(self):
        res = model_predict("all", 2019, 10, 1, test=True)
        self.assertEqual(res["target_date"], "2019-10-01")

    def test_07_invalid_country(self):
        with self.assertRaises(ValueError):
            model_predict("atlantis", 2018, 1, 5, test=True)

    def test_08_invalid_date(self):
        with self.assertRaises(ValueError):
            model_predict("all", 2030, 1, 1, test=True)
        with self.assertRaises(ValueError):
            model_predict("all", 2018, 2, 30, test=True)


if __name__ == "__main__":
    unittest.main()
