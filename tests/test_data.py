"""Tests for data loading, validation, and the append/import workflow."""

import pandas as pd

from census_forecaster import config as C
from census_forecaster import data as data_mod


def test_load_empty_when_no_file(cfg):
    assert len(data_mod.load_census(cfg)) == 0
    assert len(data_mod.load_demographics(cfg)) == 0
    assert data_mod.load_holidays(cfg) == set()


def test_append_census_persists_and_sorts(cfg):
    data_mod.append_census(cfg, "2025-01-02", 100)
    data_mod.append_census(cfg, "2025-01-01", 90)
    df = data_mod.load_census(cfg)
    assert list(df[C.CENSUS_VALUE]) == [90.0, 100.0]  # sorted by date
    assert df[C.CENSUS_DATE].is_monotonic_increasing


def test_append_census_overwrites_same_date(cfg):
    data_mod.append_census(cfg, "2025-01-01", 90)
    data_mod.append_census(cfg, "2025-01-01", 95)  # correction
    df = data_mod.load_census(cfg)
    assert len(df) == 1
    assert df[C.CENSUS_VALUE].iloc[0] == 95.0


def test_append_demographics_overwrites_year(cfg):
    data_mod.append_demographics(cfg, 2025, 100000, 39.0, 0.17, 0.015)
    data_mod.append_demographics(cfg, 2025, 101000, 39.1, 0.171, 0.016)
    df = data_mod.load_demographics(cfg)
    assert len(df) == 1
    assert df[C.DEMO_POPULATION].iloc[0] == 101000


def test_import_census_csv_merges(cfg, tmp_path):
    data_mod.append_census(cfg, "2025-01-01", 90)
    incoming = tmp_path / "incoming.csv"
    pd.DataFrame(
        {C.CENSUS_DATE: ["2025-01-01", "2025-01-02"], C.CENSUS_VALUE: [99, 110]}
    ).to_csv(incoming, index=False)
    df = data_mod.import_census_csv(cfg, incoming)
    assert len(df) == 2
    # Incoming value wins on the conflicting date.
    assert df.loc[df[C.CENSUS_DATE] == pd.Timestamp("2025-01-01"), C.CENSUS_VALUE].iloc[0] == 99.0


def test_summary_reports_range(seeded_cfg):
    summary = data_mod.data_summary(seeded_cfg)
    assert summary["census_rows"] > 1000
    assert "census_start" in summary and "census_end" in summary
