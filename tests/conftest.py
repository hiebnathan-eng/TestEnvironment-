"""Shared pytest fixtures."""

from pathlib import Path

import pytest

from census_forecaster.config import Config
from census_forecaster import sample_data


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    """A Config pointed at an isolated temp data directory."""
    return Config(data_dir=tmp_path / "data")


@pytest.fixture
def seeded_cfg(cfg: Config) -> Config:
    """A Config whose data dir is populated with 4 years of sample data."""
    sample_data.generate(cfg, years=4)
    return cfg
