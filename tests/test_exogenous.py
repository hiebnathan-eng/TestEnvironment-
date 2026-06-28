"""Tests for weather/flu (exogenous) signals and pattern analysis."""

import numpy as np
import pandas as pd

from census_forecaster import config as C
from census_forecaster import data as data_mod
from census_forecaster.features import ExogenousData, ExogenousSeries
from census_forecaster.forecast import train
from census_forecaster.patterns import analyze


def test_anomaly_is_zero_for_unknown_future_dates():
    df = pd.DataFrame(
        {C.WEATHER_DATE: pd.date_range("2024-01-01", periods=60, freq="D"),
         C.WEATHER_TEMP: np.linspace(30, 50, 60)}
    )
    series = ExogenousSeries.from_frame(df, C.WEATHER_TEMP)
    future = pd.DatetimeIndex(["2030-06-15"])  # no data here
    assert series.anomaly(future)[0] == 0.0


def test_exogenous_data_feature_names_order():
    weather = pd.DataFrame(
        {C.WEATHER_DATE: pd.date_range("2024-01-01", periods=10), C.WEATHER_TEMP: range(10)}
    )
    flu = pd.DataFrame(
        {C.FLU_DATE: pd.date_range("2024-01-01", periods=10), C.FLU_INDEX: range(10)}
    )
    exog = ExogenousData.from_frames(weather, flu)
    _, names = exog.feature_columns(pd.date_range("2024-01-01", periods=3))
    assert names == ["temp_anomaly", "flu_anomaly"]


def test_no_exogenous_files_is_fine(cfg):
    """Model trains using census alone when weather/flu are absent."""
    for i in range(30):
        d = pd.Timestamp("2025-01-01") + pd.Timedelta(days=i)
        data_mod.append_census(cfg, d, 100 + (i % 7))
    bundle = train(cfg)
    assert "temp_anomaly" not in bundle.feature_names
    assert "flu_anomaly" not in bundle.feature_names


def test_model_learns_weather_and_flu_signs(seeded_cfg):
    """Sample data builds in 'colder->busier' and 'worse flu->busier'."""
    bundle = train(seeded_cfg)
    coefs = bundle.model.coefficients(bundle.feature_names)
    # temp_anomaly = temp - normal; colder (negative anomaly) raises census => negative weight.
    assert coefs["temp_anomaly"] < 0
    # flu_anomaly positive (worse than usual) raises census => positive weight.
    assert coefs["flu_anomaly"] > 0


def test_analyze_reports_patterns(seeded_cfg):
    result = analyze(seeded_cfg)
    assert "day_of_week_effect" in result
    assert "seasonal_peak_month" in result
    # Winter should be the busy season in the sample data.
    assert result["seasonal_peak_month"] in (11, 12, 1, 2)
    assert result["flu_sensitivity"] > 0
    assert result["weather_sensitivity"] < 0


def test_weather_data_round_trips(seeded_cfg):
    w = data_mod.load_weather(seeded_cfg)
    f = data_mod.load_flu(seeded_cfg)
    assert len(w) > 1000 and C.WEATHER_TEMP in w.columns
    assert len(f) > 1000 and C.FLU_INDEX in f.columns
