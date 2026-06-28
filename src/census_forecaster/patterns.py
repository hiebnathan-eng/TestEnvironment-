"""Surface the patterns the model has learned, in plain language.

"Incorporating patterns" is only trustworthy if you can see what was found. This
module reports three things:

1. **Observed structure** in the raw history — day-of-week and month effects, and
   how strongly census tracks weather and flu.
2. **What the model weights** — the largest standardised coefficients, i.e. the
   features doing the most work in the predictions.
3. **Weather/flu sensitivity** — the sign and size of those effects, so "colder →
   busier" or "bad flu year → busier" is stated explicitly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from . import data as data_mod
from .config import Config
from .forecast import train

_DOW_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def analyze(cfg: Config) -> dict:
    """Return a dict describing learned and observed patterns."""
    census = data_mod.load_census(cfg)
    if len(census) < 14:
        raise ValueError(
            f"Need at least 14 days of history to analyse patterns; have {len(census)}."
        )
    bundle = train(cfg)
    dates = pd.DatetimeIndex(census[C.CENSUS_DATE])
    y = census[C.CENSUS_VALUE].to_numpy(dtype=float)
    overall = float(np.mean(y))

    result: dict = {"overall_mean_census": round(overall, 1)}

    # Day-of-week effect: average deviation from the overall mean.
    dow = dates.dayofweek.to_numpy()
    result["day_of_week_effect"] = {
        _DOW_NAMES[d]: round(float(np.mean(y[dow == d]) - overall), 1)
        for d in range(7)
        if np.any(dow == d)
    }

    # Month effect, to locate the seasonal peak and trough.
    month = dates.month.to_numpy()
    month_dev = {
        m: round(float(np.mean(y[month == m]) - overall), 1)
        for m in range(1, 13)
        if np.any(month == m)
    }
    result["month_effect"] = month_dev
    if month_dev:
        result["seasonal_peak_month"] = max(month_dev, key=month_dev.get)
        result["seasonal_trough_month"] = min(month_dev, key=month_dev.get)

    # Correlation of census with weather and flu over overlapping dates.
    weather = data_mod.load_weather(cfg)
    if len(weather):
        merged = census.merge(weather, on=C.CENSUS_DATE, how="inner")
        corr = _safe_corr(
            merged[C.CENSUS_VALUE].to_numpy(float),
            merged[C.WEATHER_TEMP].to_numpy(float),
        )
        if corr is not None:
            result["census_vs_temperature_corr"] = round(corr, 3)
    flu = data_mod.load_flu(cfg)
    if len(flu):
        merged = census.merge(flu, on=C.CENSUS_DATE, how="inner")
        corr = _safe_corr(
            merged[C.CENSUS_VALUE].to_numpy(float),
            merged[C.FLU_INDEX].to_numpy(float),
        )
        if corr is not None:
            result["census_vs_flu_corr"] = round(corr, 3)

    # Largest standardised model coefficients (the top drivers of the forecast).
    coefs = bundle.model.coefficients(bundle.feature_names)
    coefs.pop("intercept", None)
    ranked = sorted(coefs.items(), key=lambda kv: abs(kv[1]), reverse=True)
    result["top_drivers"] = [(name, round(val, 2)) for name, val in ranked[:8]]

    # Explicit weather/flu sensitivity (signed), if those features are present.
    if "temp_anomaly" in coefs:
        result["weather_sensitivity"] = round(coefs["temp_anomaly"], 2)
    if "flu_anomaly" in coefs:
        result["flu_sensitivity"] = round(coefs["flu_anomaly"], 2)

    return result
