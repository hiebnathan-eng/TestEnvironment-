"""Generate realistic synthetic data so the system works end-to-end immediately.

The generated census embeds the same structure the model is built to recover:
a demographic-driven base level, a winter-peaking seasonal cycle, weekday/weekend
differences, a gentle upward trend, holiday dips, and random noise. Replace these
files with your real exports (same columns) when you are ready — the generator is
only a stand-in.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .config import Config

_DAYS_PER_YEAR = 365.25


def generate(
    cfg: Config,
    years: int = 4,
    end: str = "2025-12-31",
    base_population: int = 120_000,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Write `census_log.csv` and `demographics.csv`, returning both frames."""
    rng = np.random.default_rng(seed)
    end_ts = pd.Timestamp(end).normalize()
    start_ts = (end_ts - pd.Timedelta(days=int(years * _DAYS_PER_YEAR))).normalize()
    dates = pd.date_range(start_ts, end_ts, freq="D")

    # --- Demographics: one row per year, gently growing and ageing. ----------
    yr_lo, yr_hi = start_ts.year, end_ts.year
    demo_rows = []
    growth = 0.015  # 1.5% annual population growth
    for i, year in enumerate(range(yr_lo, yr_hi + 1)):
        pop = base_population * ((1 + growth) ** i)
        demo_rows.append(
            {
                C.DEMO_YEAR: year,
                C.DEMO_POPULATION: round(pop),
                C.DEMO_MEDIAN_AGE: round(38.0 + 0.25 * i, 1),
                C.DEMO_PCT_OVER_65: round(0.16 + 0.004 * i, 4),
                C.DEMO_GROWTH_RATE: growth,
            }
        )
    demo = pd.DataFrame(demo_rows)

    # --- Census: structure the model is meant to learn. ----------------------
    doy = dates.dayofyear.to_numpy(dtype=float)
    dow = dates.dayofweek.to_numpy()
    t_years = (dates - dates[0]).days.to_numpy(dtype=float) / _DAYS_PER_YEAR

    # Population/elderly share interpolated to daily, scaling the base level.
    year_frac = dates.year.to_numpy(dtype=float) + (doy - 1) / _DAYS_PER_YEAR
    pop_by_year = demo[C.DEMO_POPULATION].to_numpy(dtype=float)
    pct65_by_year = demo[C.DEMO_PCT_OVER_65].to_numpy(dtype=float)
    yrs = demo[C.DEMO_YEAR].to_numpy(dtype=float)
    pop_daily = np.interp(year_frac, yrs, pop_by_year)
    pct65_daily = np.interp(year_frac, yrs, pct65_by_year)

    # --- Weather: seasonal temperature with per-year offsets and daily noise. -
    temp_normal = 55.0 - 25.0 * np.cos(2.0 * np.pi * (doy - 15) / _DAYS_PER_YEAR)
    year_lookup = {y: rng.normal(0.0, 3.0) for y in range(yr_lo, yr_hi + 1)}
    temp_year_offset = np.array([year_lookup[int(y)] for y in dates.year])
    temp = temp_normal + temp_year_offset + rng.normal(0.0, 4.0, size=len(dates))
    temp_anomaly = temp - temp_normal  # colder-than-normal is positive cold load

    # --- Flu: winter-peaking curve whose *severity* varies year to year. ------
    flu_curve = (0.5 + 0.5 * np.cos(2.0 * np.pi * (doy - 30) / _DAYS_PER_YEAR)) ** 3
    severities = {y: s for y, s in zip(range(yr_lo, yr_hi + 1), [0.7, 1.4, 1.0, 1.5, 0.85, 1.1, 0.9])}
    sev_daily = np.array([severities[int(y)] for y in dates.year])
    mean_sev = float(np.mean(list(severities.values())[: yr_hi - yr_lo + 1]))
    flu_index = np.clip(sev_daily * 8.0 * flu_curve + rng.normal(0.0, 0.4, len(dates)), 0, None)
    flu_normal = mean_sev * 8.0 * flu_curve
    flu_anomaly = flu_index - flu_normal  # worse-than-usual season is positive

    # Base census scales with population and skews up with the elderly share.
    base = 0.0011 * pop_daily * (1.0 + 1.5 * (pct65_daily - 0.16))

    # Winter-peaking seasonal swing (the *average* weather/flu effect lives here).
    seasonal = 18.0 * np.cos(2.0 * np.pi * (doy - 15) / _DAYS_PER_YEAR)

    # Weekend census runs lower (fewer elective admissions, weekend discharges).
    weekend = np.where(dow >= 5, -9.0, 0.0)

    # Gentle non-demographic upward drift.
    drift = 4.0 * t_years

    # Year/day-specific deviations the model can only learn from weather/flu data.
    cold_extra = -0.35 * temp_anomaly  # colder than normal -> more census
    flu_extra = 2.2 * flu_anomaly  # worse flu season -> more census

    noise = rng.normal(0.0, 5.0, size=len(dates))
    census = base + seasonal + weekend + drift + cold_extra + flu_extra + noise

    # Holiday dips on a handful of fixed-date holidays.
    holiday_md = {(1, 1), (7, 4), (11, 11), (12, 25)}
    md = list(zip(dates.month.to_numpy(), dates.day.to_numpy()))
    holiday_mask = np.array([(m, d) in holiday_md for m, d in md])
    census = np.where(holiday_mask, census - 12.0, census)

    census = np.clip(np.round(census), 0, None)

    census_df = pd.DataFrame(
        {
            C.CENSUS_DATE: dates,
            C.CENSUS_VALUE: census.astype(int),
            C.CENSUS_ADMISSIONS: pd.NA,
            C.CENSUS_DISCHARGES: pd.NA,
        }
    )

    # --- Holidays file the model can use as a predictor. ---------------------
    holiday_rows = []
    for d in dates[holiday_mask]:
        holiday_rows.append({C.HOLIDAY_DATE: d.date().isoformat(), C.HOLIDAY_NAME: "holiday"})
    holidays_df = pd.DataFrame(holiday_rows)

    # --- Weather and flu activity files (optional model inputs). --------------
    weather_df = pd.DataFrame(
        {C.WEATHER_DATE: dates, C.WEATHER_TEMP: np.round(temp, 1)}
    )
    flu_df = pd.DataFrame(
        {C.FLU_DATE: dates, C.FLU_INDEX: np.round(flu_index, 2)}
    )

    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    census_df.to_csv(cfg.census_path, index=False)
    demo.to_csv(cfg.demographics_path, index=False)
    holidays_df.to_csv(cfg.holidays_path, index=False)
    weather_df.to_csv(cfg.weather_path, index=False)
    flu_df.to_csv(cfg.flu_path, index=False)
    return census_df, demo
