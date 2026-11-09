"""
estimators.py - custom estimators, kept in their own module so pickled models
load from any entry point (model.py, app.py, monitoring.py, tests).
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone


class NaiveBaseline(BaseEstimator, RegressorMixin):
    """revenue of the next 30 days = revenue of the last 28 days scaled to 30 days"""

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return np.asarray(X["previous_28"], dtype=float) * 30.0 / 28.0


class BaselineResidual(BaseEstimator, RegressorMixin):
    """
    wrap a regressor so it learns the *correction* to the naive baseline
    (y - baseline) instead of the raw revenue.  Tree models cannot extrapolate
    beyond the revenue levels seen in training; learning the residual fixes that.
    """

    def __init__(self, estimator=None):
        self.estimator = estimator

    def fit(self, X, y):
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, np.asarray(y, dtype=float) - NaiveBaseline().predict(X))
        return self

    def predict(self, X):
        return NaiveBaseline().predict(X) + self.estimator_.predict(X)
