"""Produce day-to-day, month-to-month, and year-to-year forecasts.

The flow is: train on all available history → predict daily census across the
requested horizon (with uncertainty) → aggregate those daily predictions up to
months and years. Aggregating from a single daily model keeps the three
granularities mutually consistent (the monthly average is literally the mean of
that month's daily predictions).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C
from . import data as data_mod
from .config import Config
from .features import DemographicsModel, ExogenousData, build_feature_matrix
from .model import RidgeModel


@dataclass
class Bundle:
    """A trained model plus everything needed to featurise new dates."""

    model: RidgeModel
    demo_model: DemographicsModel
    holidays: set
    yearly_harmonics: int
    interval_z: float
    feature_names: list
    history: pd.DataFrame
    exog: ExogenousData


def train(cfg: Config) -> Bundle:
    """Fit the model on all census history currently on hand."""
    census = data_mod.load_census(cfg)
    if len(census) < 14:
        raise ValueError(
            f"Need at least 14 days of census history to train; have {len(census)}. "
            "Add more data (see `census-forecast add-census` / `import-census`)."
        )
    demo = data_mod.load_demographics(cfg)
    holidays = data_mod.load_holidays(cfg)
    demo_model = DemographicsModel.from_frame(demo)
    exog = ExogenousData.from_frames(
        data_mod.load_weather(cfg), data_mod.load_flu(cfg)
    )

    dates = pd.DatetimeIndex(census[C.CENSUS_DATE])
    X, names = build_feature_matrix(
        dates, demo_model, holidays, cfg.yearly_harmonics, exog
    )
    y = census[C.CENSUS_VALUE].to_numpy(dtype=float)

    model = RidgeModel(alpha=cfg.ridge_alpha).fit(X, y)
    return Bundle(
        model=model,
        demo_model=demo_model,
        holidays=holidays,
        yearly_harmonics=cfg.yearly_harmonics,
        interval_z=cfg.interval_z,
        feature_names=names,
        history=census,
        exog=exog,
    )


def predict_daily(bundle: Bundle, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Predict census for each date with a symmetric prediction interval."""
    X, _ = build_feature_matrix(
        dates,
        bundle.demo_model,
        bundle.holidays,
        bundle.yearly_harmonics,
        bundle.exog,
    )
    mean = bundle.model.predict(X)
    half = bundle.interval_z * bundle.model.resid_std_
    lower = np.clip(mean - half, 0.0, None)
    upper = mean + half
    return pd.DataFrame(
        {
            "date": dates,
            "predicted_census": np.round(mean, 1),
            "lower": np.round(lower, 1),
            "upper": np.round(upper, 1),
        }
    )


def forecast(
    cfg: Config, horizon_days: int, start: str | pd.Timestamp | None = None
) -> tuple[pd.DataFrame, Bundle]:
    """Train and return a daily forecast frame for the next `horizon_days`.

    By default the forecast begins the day after the last observed date.
    """
    bundle = train(cfg)
    if start is None:
        last = bundle.history[C.CENSUS_DATE].max()
        start_ts = (last + pd.Timedelta(days=1)).normalize()
    else:
        start_ts = pd.Timestamp(start).normalize()
    future = pd.date_range(start_ts, periods=horizon_days, freq="D")
    daily = predict_daily(bundle, future)
    return daily, bundle


def _interval_for_mean(resid_std: float, z: float, n_days: int) -> float:
    """Half-width of the interval for an average over `n_days`.

    Daily census is autocorrelated, so treating every day as independent would
    understate uncertainty. We approximate the effective sample size as the
    number of weeks (≈ independent blocks), which is deliberately conservative.
    """
    n_eff = max(n_days / 7.0, 1.0)
    return z * resid_std / np.sqrt(n_eff)


def aggregate(daily: pd.DataFrame, bundle: Bundle, freq: str) -> pd.DataFrame:
    """Roll a daily forecast up to month (`freq='M'`) or year (`freq='Y'`).

    Reports the average census, total patient-days (bed-days), the peak day, and
    an interval on the average. `freq` accepts 'M'/'month' or 'Y'/'year'.
    """
    key = freq.lower()
    if key in ("m", "month", "monthly"):
        period, label = "M", "month"
    elif key in ("y", "year", "yearly"):
        period, label = "Y", "year"
    else:
        raise ValueError("freq must be 'M'/'month' or 'Y'/'year'")

    d = daily.copy()
    d["bucket"] = pd.PeriodIndex(d["date"], freq=period)
    z, resid_std = bundle.interval_z, bundle.model.resid_std_

    rows = []
    for bucket, grp in d.groupby("bucket", sort=True):
        n = len(grp)
        avg = float(grp["predicted_census"].mean())
        half = _interval_for_mean(resid_std, z, n)
        rows.append(
            {
                label: str(bucket),
                "days": n,
                "avg_census": round(avg, 1),
                "avg_lower": round(max(avg - half, 0.0), 1),
                "avg_upper": round(avg + half, 1),
                "peak_census": round(float(grp["predicted_census"].max()), 1),
                "patient_days": round(float(grp["predicted_census"].sum()), 0),
            }
        )
    return pd.DataFrame(rows)
