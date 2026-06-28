"""Feature engineering.

Turns a set of calendar dates (plus demographics and holidays) into a numeric
design matrix the model can learn from. The features encode the three signals
that drive hospital census:

1. **Calendar rhythm** — day-of-week effects (weekends differ) and a smooth
   within-year seasonal cycle (Fourier terms) that captures, e.g., winter peaks.
2. **Long-run trend** — a linear time term for drift not explained by demographics.
3. **Community structure** — population size, median age, and the share of
   residents over 65, which scale and shift demand year to year.

All features are plain floats so the same builder serves both training (on
historical dates) and forecasting (on future dates).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C

# Reference epoch so the linear trend is numerically small and centred-ish.
_EPOCH = pd.Timestamp("2015-01-01")
_DAYS_PER_YEAR = 365.25


class ExogenousSeries:
    """An external daily signal (e.g. temperature, flu activity) as an *anomaly*.

    The recurring seasonal shape of weather and flu is already captured by the
    model's yearly seasonality, so feeding the raw value would be redundant. What
    carries new information is the **deviation from the seasonal normal** — a
    colder-than-usual day, a worse-than-usual flu week.

    We learn a day-of-year "climatology" (the typical value for each calendar
    day) from the supplied history, then expose ``anomaly(dates) = value - normal``.
    For any date with no supplied value (notably *future* dates with no forecast),
    the anomaly is 0 — i.e. the model assumes a typical day, which is the honest
    fallback when the future signal is unknown.
    """

    def __init__(self, by_date: dict, climatology: np.ndarray, global_mean: float):
        self._by_date = by_date
        self._clim = climatology  # indexed by day-of-year (1..366)
        self._global_mean = global_mean

    @classmethod
    def from_frame(cls, df: pd.DataFrame, value_col: str) -> "ExogenousSeries | None":
        if df is None or len(df) == 0:
            return None
        dates = pd.DatetimeIndex(df[C.WEATHER_DATE])
        vals = df[value_col].to_numpy(dtype=float)
        by_date = {d.normalize(): v for d, v in zip(dates, vals)}
        doy = dates.dayofyear.to_numpy()
        global_mean = float(vals.mean())
        clim = np.full(367, global_mean)  # 1..366; index 0 unused
        for d in range(1, 367):
            mask = doy == d
            if mask.any():
                clim[d] = float(vals[mask].mean())
        return cls(by_date, clim, global_mean)

    def anomaly(self, dates: pd.DatetimeIndex) -> np.ndarray:
        doy = dates.dayofyear.to_numpy()
        out = np.empty(len(dates))
        for i, d in enumerate(dates):
            normal = self._clim[doy[i]]
            value = self._by_date.get(d.normalize(), normal)
            out[i] = value - normal
        return out


class ExogenousData:
    """Bundle of named external signals appended as features when available.

    Order is fixed (``temp`` then ``flu``) so feature columns line up between
    training and prediction. Missing signals are simply omitted.
    """

    _ORDER = ("temp", "flu")

    def __init__(self, series: dict[str, ExogenousSeries]):
        self.series = series

    @classmethod
    def from_frames(
        cls, weather: pd.DataFrame | None, flu: pd.DataFrame | None
    ) -> "ExogenousData":
        series: dict[str, ExogenousSeries] = {}
        temp = ExogenousSeries.from_frame(weather, C.WEATHER_TEMP) if weather is not None else None
        if temp is not None:
            series["temp"] = temp
        flu_s = ExogenousSeries.from_frame(flu, C.FLU_INDEX) if flu is not None else None
        if flu_s is not None:
            series["flu"] = flu_s
        return cls(series)

    def feature_columns(
        self, dates: pd.DatetimeIndex
    ) -> tuple[list[np.ndarray], list[str]]:
        cols: list[np.ndarray] = []
        names: list[str] = []
        for key in self._ORDER:
            if key in self.series:
                cols.append(self.series[key].anomaly(dates))
                names.append(f"{key}_anomaly")
        return cols, names


@dataclass
class DemographicsModel:
    """Per-year demographics with linear extrapolation for future years.

    Built from the (possibly sparse) yearly demographics table. Provides smooth
    daily values by interpolating across a continuous "year" coordinate, and
    extrapolates beyond the known range using the trend of the last two points
    (population uses the stated growth rate when available).
    """

    years: np.ndarray
    population: np.ndarray
    median_age: np.ndarray
    pct_over_65: np.ndarray
    available: bool

    @classmethod
    def from_frame(cls, demo: pd.DataFrame) -> "DemographicsModel":
        if demo is None or len(demo) == 0:
            return cls(
                years=np.array([]),
                population=np.array([]),
                median_age=np.array([]),
                pct_over_65=np.array([]),
                available=False,
            )
        demo = demo.sort_values(C.DEMO_YEAR)
        return cls(
            years=demo[C.DEMO_YEAR].to_numpy(dtype=float),
            population=demo[C.DEMO_POPULATION].to_numpy(dtype=float),
            median_age=demo[C.DEMO_MEDIAN_AGE].to_numpy(dtype=float),
            pct_over_65=demo[C.DEMO_PCT_OVER_65].to_numpy(dtype=float),
            available=True,
        )

    def _interp(self, values: np.ndarray, cont_year: np.ndarray) -> np.ndarray:
        """Linear interpolation in continuous-year space with linear extrapolation."""
        if len(self.years) == 1:
            return np.full_like(cont_year, values[0], dtype=float)
        out = np.interp(cont_year, self.years, values)
        # np.interp clamps outside the range; replace clamped tails with a linear
        # extrapolation from the two nearest known points so trends keep going.
        lo, hi = self.years[0], self.years[-1]
        if np.any(cont_year < lo):
            slope = (values[1] - values[0]) / (self.years[1] - self.years[0])
            mask = cont_year < lo
            out[mask] = values[0] + slope * (cont_year[mask] - lo)
        if np.any(cont_year > hi):
            slope = (values[-1] - values[-2]) / (self.years[-1] - self.years[-2])
            mask = cont_year > hi
            out[mask] = values[-1] + slope * (cont_year[mask] - hi)
        return out

    def daily(self, dates: pd.DatetimeIndex) -> dict[str, np.ndarray]:
        """Return interpolated population/age/elderly-share for each date."""
        doy = dates.dayofyear.to_numpy(dtype=float)
        year = dates.year.to_numpy(dtype=float)
        cont_year = year + (doy - 1) / _DAYS_PER_YEAR
        return {
            C.DEMO_POPULATION: self._interp(self.population, cont_year),
            C.DEMO_MEDIAN_AGE: self._interp(self.median_age, cont_year),
            C.DEMO_PCT_OVER_65: self._interp(self.pct_over_65, cont_year),
        }


def build_feature_matrix(
    dates: pd.DatetimeIndex,
    demo_model: DemographicsModel,
    holidays: set[pd.Timestamp],
    yearly_harmonics: int,
    exog: "ExogenousData | None" = None,
) -> tuple[np.ndarray, list[str]]:
    """Build the (n_dates, n_features) design matrix and the feature names.

    The intercept is NOT included here — the model adds and handles it
    separately so it is not standardised or penalised. Optional external signals
    (weather, flu) are appended via ``exog`` when provided.
    """
    dates = pd.DatetimeIndex(dates)
    n = len(dates)
    cols: list[np.ndarray] = []
    names: list[str] = []

    # Linear trend, in years since the epoch.
    t_years = (dates - _EPOCH).days.to_numpy(dtype=float) / _DAYS_PER_YEAR
    cols.append(t_years)
    names.append("trend_years")

    # Day-of-week one-hot (Monday=0 .. Saturday=5; Sunday is the reference).
    dow = dates.dayofweek.to_numpy()
    for d, label in enumerate(["mon", "tue", "wed", "thu", "fri", "sat"]):
        cols.append((dow == d).astype(float))
        names.append(f"dow_{label}")

    # Within-year seasonality via Fourier terms.
    doy = dates.dayofyear.to_numpy(dtype=float)
    phase = 2.0 * np.pi * doy / _DAYS_PER_YEAR
    for k in range(1, yearly_harmonics + 1):
        cols.append(np.sin(k * phase))
        names.append(f"yseason_sin{k}")
        cols.append(np.cos(k * phase))
        names.append(f"yseason_cos{k}")

    # Holiday flag.
    if holidays:
        norm = {pd.Timestamp(h).normalize() for h in holidays}
        flag = np.array([1.0 if d.normalize() in norm else 0.0 for d in dates])
    else:
        flag = np.zeros(n)
    cols.append(flag)
    names.append("holiday")

    # Demographic covariates (only if we have demographic data).
    if demo_model.available:
        daily = demo_model.daily(dates)
        cols.append(daily[C.DEMO_POPULATION])
        names.append("population")
        cols.append(daily[C.DEMO_MEDIAN_AGE])
        names.append("median_age")
        cols.append(daily[C.DEMO_PCT_OVER_65])
        names.append("pct_over_65")

    # External signals (weather, flu) as deviations from the seasonal normal.
    if exog is not None:
        exog_cols, exog_names = exog.feature_columns(dates)
        cols.extend(exog_cols)
        names.extend(exog_names)

    X = np.column_stack(cols)
    return X, names
