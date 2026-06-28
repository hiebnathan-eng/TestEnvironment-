"""Data loading, validation, and *continuous* appending.

This module is the heart of the "keep adding information" workflow. Census
observations and demographic rows live in plain CSV files so they are easy to
inspect, edit, and back up. Append helpers de-duplicate by key (date / year) and
keep the most recent value, so re-importing overlapping data is safe.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config as C
from .config import Config


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_census(cfg: Config) -> pd.DataFrame:
    """Load the daily census log, sorted and de-duplicated by date.

    Returns an empty, correctly-typed frame if the file does not exist yet.
    """
    if not cfg.census_path.exists():
        return pd.DataFrame(columns=C.CENSUS_COLUMNS).astype(
            {C.CENSUS_VALUE: "float64"}
        )
    df = pd.read_csv(cfg.census_path, parse_dates=[C.CENSUS_DATE])
    missing = {C.CENSUS_DATE, C.CENSUS_VALUE} - set(df.columns)
    if missing:
        raise ValueError(
            f"{cfg.census_path} is missing required column(s): {sorted(missing)}"
        )
    for col in (C.CENSUS_ADMISSIONS, C.CENSUS_DISCHARGES):
        if col not in df.columns:
            df[col] = pd.NA
    df = df[C.CENSUS_COLUMNS]
    df = (
        df.dropna(subset=[C.CENSUS_DATE, C.CENSUS_VALUE])
        .drop_duplicates(subset=[C.CENSUS_DATE], keep="last")
        .sort_values(C.CENSUS_DATE)
        .reset_index(drop=True)
    )
    df[C.CENSUS_VALUE] = df[C.CENSUS_VALUE].astype(float)
    return df


def load_demographics(cfg: Config) -> pd.DataFrame:
    """Load yearly demographics, sorted and de-duplicated by year.

    Returns an empty frame if the file does not exist; the model then falls back
    to census history alone.
    """
    if not cfg.demographics_path.exists():
        return pd.DataFrame(columns=C.DEMO_COLUMNS)
    df = pd.read_csv(cfg.demographics_path)
    if C.DEMO_YEAR not in df.columns:
        raise ValueError(f"{cfg.demographics_path} must have a '{C.DEMO_YEAR}' column")
    for col in C.DEMO_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[C.DEMO_COLUMNS]
    df = (
        df.dropna(subset=[C.DEMO_YEAR])
        .drop_duplicates(subset=[C.DEMO_YEAR], keep="last")
        .sort_values(C.DEMO_YEAR)
        .reset_index(drop=True)
    )
    df[C.DEMO_YEAR] = df[C.DEMO_YEAR].astype(int)
    return df


def load_holidays(cfg: Config) -> set[pd.Timestamp]:
    """Load optional holiday dates as a set of normalised Timestamps."""
    if not cfg.holidays_path.exists():
        return set()
    df = pd.read_csv(cfg.holidays_path, parse_dates=[C.HOLIDAY_DATE])
    return {ts.normalize() for ts in df[C.HOLIDAY_DATE]}


def append_census(
    cfg: Config,
    date: str | pd.Timestamp,
    census: float,
    admissions: float | None = None,
    discharges: float | None = None,
) -> pd.DataFrame:
    """Append (or overwrite) a single day's census observation and persist it.

    Returns the full, updated census frame. Appending the same date twice keeps
    the latest value, so corrections are easy.
    """
    df = load_census(cfg)
    row = {
        C.CENSUS_DATE: pd.Timestamp(date).normalize(),
        C.CENSUS_VALUE: float(census),
        C.CENSUS_ADMISSIONS: admissions,
        C.CENSUS_DISCHARGES: discharges,
    }
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df = (
        df.drop_duplicates(subset=[C.CENSUS_DATE], keep="last")
        .sort_values(C.CENSUS_DATE)
        .reset_index(drop=True)
    )
    _ensure_dir(cfg.census_path)
    df.to_csv(cfg.census_path, index=False)
    return df


def append_demographics(
    cfg: Config,
    year: int,
    population: float,
    median_age: float,
    pct_over_65: float,
    growth_rate: float,
) -> pd.DataFrame:
    """Append (or overwrite) a year's demographic row and persist it."""
    df = load_demographics(cfg)
    row = {
        C.DEMO_YEAR: int(year),
        C.DEMO_POPULATION: float(population),
        C.DEMO_MEDIAN_AGE: float(median_age),
        C.DEMO_PCT_OVER_65: float(pct_over_65),
        C.DEMO_GROWTH_RATE: float(growth_rate),
    }
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df = (
        df.drop_duplicates(subset=[C.DEMO_YEAR], keep="last")
        .sort_values(C.DEMO_YEAR)
        .reset_index(drop=True)
    )
    _ensure_dir(cfg.demographics_path)
    df.to_csv(cfg.demographics_path, index=False)
    return df


def import_census_csv(cfg: Config, path: str | Path) -> pd.DataFrame:
    """Bulk-import census rows from an external CSV, merging into the log.

    The incoming file must have at least `date` and `census` columns. Rows are
    merged by date (incoming wins on conflict), then persisted.
    """
    incoming = pd.read_csv(path, parse_dates=[C.CENSUS_DATE])
    missing = {C.CENSUS_DATE, C.CENSUS_VALUE} - set(incoming.columns)
    if missing:
        raise ValueError(f"{path} is missing required column(s): {sorted(missing)}")
    for col in (C.CENSUS_ADMISSIONS, C.CENSUS_DISCHARGES):
        if col not in incoming.columns:
            incoming[col] = pd.NA
    existing = load_census(cfg)
    merged = pd.concat([existing, incoming[C.CENSUS_COLUMNS]], ignore_index=True)
    merged[C.CENSUS_DATE] = pd.to_datetime(merged[C.CENSUS_DATE]).dt.normalize()
    merged = (
        merged.dropna(subset=[C.CENSUS_DATE, C.CENSUS_VALUE])
        .drop_duplicates(subset=[C.CENSUS_DATE], keep="last")
        .sort_values(C.CENSUS_DATE)
        .reset_index(drop=True)
    )
    _ensure_dir(cfg.census_path)
    merged.to_csv(cfg.census_path, index=False)
    return merged


def data_summary(cfg: Config) -> dict:
    """Return a compact summary of what data is currently on hand."""
    census = load_census(cfg)
    demo = load_demographics(cfg)
    summary: dict = {
        "census_rows": int(len(census)),
        "demographics_rows": int(len(demo)),
    }
    if len(census):
        summary["census_start"] = census[C.CENSUS_DATE].min().date().isoformat()
        summary["census_end"] = census[C.CENSUS_DATE].max().date().isoformat()
        summary["census_mean"] = round(float(census[C.CENSUS_VALUE].mean()), 1)
        summary["census_min"] = float(census[C.CENSUS_VALUE].min())
        summary["census_max"] = float(census[C.CENSUS_VALUE].max())
    if len(demo):
        summary["demographics_years"] = (
            f"{int(demo[C.DEMO_YEAR].min())}-{int(demo[C.DEMO_YEAR].max())}"
        )
    return summary
