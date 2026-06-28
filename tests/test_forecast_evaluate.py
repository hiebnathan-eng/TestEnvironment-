"""End-to-end tests for forecasting, aggregation, and backtesting."""

import pandas as pd
import pytest

from census_forecaster.forecast import aggregate, forecast, train
from census_forecaster.evaluate import backtest


def test_forecast_daily_shape_and_bounds(seeded_cfg):
    daily, bundle = forecast(seeded_cfg, horizon_days=90)
    assert len(daily) == 90
    # Predictions sit within their own interval and stay non-negative.
    assert (daily["lower"] <= daily["predicted_census"]).all()
    assert (daily["predicted_census"] <= daily["upper"]).all()
    assert (daily["lower"] >= 0).all()
    # Forecast starts the day after the last observation.
    last = bundle.history["date"].max()
    assert daily["date"].iloc[0] == last + pd.Timedelta(days=1)


def test_monthly_and_yearly_aggregation(seeded_cfg):
    daily, bundle = forecast(seeded_cfg, horizon_days=400)
    monthly = aggregate(daily, bundle, "M")
    yearly = aggregate(daily, bundle, "Y")
    assert len(monthly) >= 12
    assert len(yearly) >= 1
    # Patient-days should roughly equal avg census * number of days.
    row = monthly.iloc[0]
    assert row["patient_days"] == pytest.approx(row["avg_census"] * row["days"], rel=0.05)


def test_seasonality_winter_above_summer(seeded_cfg):
    """The model should predict higher winter census than summer."""
    daily, _ = forecast(seeded_cfg, horizon_days=400, start="2026-01-01")
    daily = daily.set_index("date")
    jan = daily.loc["2026-01-01":"2026-01-31", "predicted_census"].mean()
    jul = daily.loc["2026-07-01":"2026-07-31", "predicted_census"].mean()
    assert jan > jul


def test_backtest_is_reasonable(seeded_cfg):
    result = backtest(seeded_cfg, holdout_days=30)
    assert result["holdout_days"] == 30
    # On clean synthetic data the average error should be modest.
    assert result["mae"] < 15
    assert result["mape_pct"] < 15


def test_train_requires_minimum_history(cfg):
    from census_forecaster import data as data_mod

    for i in range(5):
        data_mod.append_census(cfg, f"2025-01-0{i + 1}", 100 + i)
    with pytest.raises(ValueError):
        train(cfg)
