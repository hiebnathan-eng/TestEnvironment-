"""Tests for feature engineering and the ridge model."""

import numpy as np
import pandas as pd

from census_forecaster.features import DemographicsModel, build_feature_matrix
from census_forecaster.model import RidgeModel


def test_feature_matrix_shape_and_dow():
    dates = pd.date_range("2025-01-06", periods=7, freq="D")  # Mon..Sun
    demo = DemographicsModel.from_frame(None)
    X, names = build_feature_matrix(dates, demo, set(), yearly_harmonics=3)
    assert X.shape[0] == 7
    assert X.shape[1] == len(names)
    # Sunday is the reference: its day-of-week dummies are all zero.
    dow_idx = [i for i, n in enumerate(names) if n.startswith("dow_")]
    sunday_row = X[6, dow_idx]
    assert np.allclose(sunday_row, 0.0)
    # Monday has exactly one active dummy.
    assert np.isclose(X[0, dow_idx].sum(), 1.0)


def test_demographics_extrapolate_future():
    demo = pd.DataFrame(
        {
            "year": [2023, 2024],
            "population": [100000, 102000],
            "median_age": [39.0, 39.5],
            "pct_over_65": [0.17, 0.175],
            "growth_rate": [0.02, 0.02],
        }
    )
    dm = DemographicsModel.from_frame(demo)
    future = pd.DatetimeIndex(["2026-07-01"])
    vals = dm.daily(future)
    # 2026 is beyond known data; population should extrapolate upward past 102000.
    assert vals["population"][0] > 102000


def test_ridge_recovers_linear_signal():
    rng = np.random.default_rng(0)
    n = 400
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 5.0 + 3.0 * x1 - 2.0 * x2 + rng.normal(0, 0.1, size=n)
    X = np.column_stack([x1, x2])
    model = RidgeModel(alpha=0.1).fit(X, y)
    coefs = model.coefficients(["x1", "x2"])
    # Intercept on the original scale should be ~5; signs of slopes correct.
    assert abs(coefs["intercept"] - 5.0) < 0.3
    assert coefs["x1"] > 0 and coefs["x2"] < 0


def test_ridge_predict_requires_fit():
    model = RidgeModel()
    try:
        model.predict(np.zeros((1, 2)))
    except RuntimeError:
        return
    raise AssertionError("predict() should raise before fit()")
