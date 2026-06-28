"""Configuration: file locations, column names, and model hyperparameters.

Centralising these means the rest of the code never hard-codes a path or a
magic number, and you can tune the model in one place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Repository-root-relative default data directory.
DEFAULT_DATA_DIR = Path("data")

# --- Census log schema -------------------------------------------------------
# One row per calendar day. `census` is the headcount of occupied beds (the
# quantity we predict). `admissions`/`discharges` are optional context columns
# that are accepted and stored but not yet used as predictors.
CENSUS_DATE = "date"
CENSUS_VALUE = "census"
CENSUS_ADMISSIONS = "admissions"
CENSUS_DISCHARGES = "discharges"
CENSUS_COLUMNS = [CENSUS_DATE, CENSUS_VALUE, CENSUS_ADMISSIONS, CENSUS_DISCHARGES]

# --- Demographics schema -----------------------------------------------------
# One row per calendar year describing the local catchment population. These let
# the model scale census with community size and ageing, which is what drives
# year-to-year structural change.
DEMO_YEAR = "year"
DEMO_POPULATION = "population"
DEMO_MEDIAN_AGE = "median_age"
DEMO_PCT_OVER_65 = "pct_over_65"
DEMO_GROWTH_RATE = "growth_rate"  # fractional annual population growth, e.g. 0.015
DEMO_COLUMNS = [
    DEMO_YEAR,
    DEMO_POPULATION,
    DEMO_MEDIAN_AGE,
    DEMO_PCT_OVER_65,
    DEMO_GROWTH_RATE,
]

# --- Holidays schema (optional) ---------------------------------------------
HOLIDAY_DATE = "date"
HOLIDAY_NAME = "name"
HOLIDAY_COLUMNS = [HOLIDAY_DATE, HOLIDAY_NAME]


@dataclass
class Config:
    """Runtime configuration for data locations and the model.

    Attributes:
        data_dir: Directory holding the CSV inputs and where outputs are written.
        yearly_harmonics: Number of Fourier harmonic pairs used to model the
            within-year (seasonal) cycle. More harmonics capture sharper seasonal
            shapes (e.g. a narrow winter peak) at the risk of overfitting.
        ridge_alpha: L2 regularisation strength. Larger values produce smoother,
            more conservative models; smaller values fit history more tightly.
        interval_z: Z-score for prediction intervals (1.96 ≈ 95%).
    """

    data_dir: Path = field(default_factory=lambda: DEFAULT_DATA_DIR)
    yearly_harmonics: int = 4
    ridge_alpha: float = 5.0
    interval_z: float = 1.96

    @property
    def census_path(self) -> Path:
        return self.data_dir / "census_log.csv"

    @property
    def demographics_path(self) -> Path:
        return self.data_dir / "demographics.csv"

    @property
    def holidays_path(self) -> Path:
        return self.data_dir / "holidays.csv"
