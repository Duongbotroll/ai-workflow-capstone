#!/usr/bin/env python
"""API tests - use Flask's test client, so no running server is needed"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app  # noqa: E402


class ApiTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        app.config["TESTING"] = True
        cls.client = app.test_client()
        r = cls.client.post("/train", json={"mode": "test"})
        assert r.status_code == 200, r.data

    def test_01_index(self):
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json()["status"], "ok")

    def test_02_train(self):
        r = self.client.post("/train", json={"mode": "test"})
        self.assertEqual(r.status_code, 200)
        body = r.get_json()
        self.assertEqual(body["mode"], "test")
        self.assertTrue(len(body["selected_models"]) >= 1)

    def test_03_predict_all_countries(self):
        r = self.client.post("/predict", json={"country": "all", "year": "2018",
                                               "month": "01", "day": "05", "mode": "test"})
        self.assertEqual(r.status_code, 200)
        self.assertGreater(r.get_json()["y_pred"], 0)

    def test_04_predict_specific_country_get(self):
        r = self.client.get("/predict?country=united_kingdom&year=2019&month=2&day=1&mode=test")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json()["country"], "united_kingdom")

    def test_05_predict_missing_params(self):
        r = self.client.post("/predict", json={"country": "all"})
        self.assertEqual(r.status_code, 400)

    def test_06_predict_bad_country(self):
        r = self.client.post("/predict", json={"country": "atlantis", "year": 2018,
                                               "month": 1, "day": 5, "mode": "test"})
        self.assertEqual(r.status_code, 400)
        self.assertIn("error", r.get_json())

    def test_07_predict_bad_date(self):
        r = self.client.post("/predict", json={"country": "all", "year": 2031,
                                               "month": 1, "day": 5, "mode": "test"})
        self.assertEqual(r.status_code, 400)

    def test_08_logs(self):
        self.client.post("/predict", json={"country": "all", "year": 2018,
                                           "month": 1, "day": 5, "mode": "test"})
        for name in ("train-test.log", "predict-test.log"):
            r = self.client.get(f"/logs/{name}")
            self.assertEqual(r.status_code, 200, name)
            self.assertIn(b"unique_id", r.data)
            r.close()

    def test_09_logs_invalid_name(self):
        self.assertEqual(self.client.get("/logs/..%2Fapp.py").status_code, 404)
        self.assertEqual(self.client.get("/logs/secret.txt").status_code, 400)


if __name__ == "__main__":
    unittest.main()
