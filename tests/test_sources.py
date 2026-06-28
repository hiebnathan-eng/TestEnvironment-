"""Offline tests for the live-data fetchers.

The network call is isolated in `_get_json`; everything here tests the pure
parsing/date logic with hand-built payloads, so no internet is required.
"""

import datetime

import pandas as pd
import pytest

from census_forecaster import config as C
from census_forecaster import data as data_mod
from census_forecaster import sources


def test_mmwr_week_start_known_values():
    # 2024 MMWR week 1 starts Sun 2023-12-31; week 2 starts 2024-01-07.
    assert sources.mmwr_week_start(2024, 2) == datetime.date(2024, 1, 7)
    # 2020 MMWR week 1 starts Sun 2019-12-29.
    assert sources.mmwr_week_start(2020, 1) == datetime.date(2019, 12, 29)


def test_parse_open_meteo_daily():
    payload = {
        "daily": {
            "time": ["2025-01-01", "2025-01-02", "2025-01-03"],
            "temperature_2m_mean": [30.5, None, 28.1],
        }
    }
    df = sources._parse_open_meteo_daily(payload)
    assert list(df.columns) == [C.WEATHER_DATE, C.WEATHER_TEMP]
    # The None temperature row is dropped.
    assert len(df) == 2
    assert df[C.WEATHER_TEMP].iloc[0] == 30.5


def test_parse_open_meteo_bad_payload_raises():
    with pytest.raises(sources.FetchError):
        sources._parse_open_meteo_daily({"unexpected": True})


def test_parse_delphi_fluview():
    payload = {
        "result": 1,
        "epidata": [
            {"epiweek": 202402, "wili": 4.2},
            {"epiweek": 202403, "wili": None, "ili": 3.9},  # falls back to ili
            {"epiweek": 202404, "wili": None, "ili": None},  # skipped
        ],
    }
    weekly = sources._parse_delphi_fluview(payload)
    assert len(weekly) == 2
    assert weekly[C.FLU_DATE].iloc[0] == pd.Timestamp("2024-01-07")
    assert weekly[C.FLU_INDEX].iloc[0] == 4.2


def test_parse_delphi_error_result_raises():
    with pytest.raises(sources.FetchError):
        sources._parse_delphi_fluview({"result": -2, "message": "no results"})


def test_weekly_to_daily_expands_seven_days():
    weekly = pd.DataFrame(
        {C.FLU_DATE: [pd.Timestamp("2024-01-07")], C.FLU_INDEX: [4.2]}
    )
    daily = sources.weekly_to_daily(weekly)
    assert len(daily) == 7
    assert daily[C.FLU_DATE].min() == pd.Timestamp("2024-01-07")
    assert daily[C.FLU_DATE].max() == pd.Timestamp("2024-01-13")
    assert (daily[C.FLU_INDEX] == 4.2).all()


def test_merge_fetched_frames_into_files(cfg):
    weather = pd.DataFrame(
        {C.WEATHER_DATE: pd.date_range("2025-01-01", periods=5), C.WEATHER_TEMP: range(5)}
    )
    flu = pd.DataFrame(
        {C.FLU_DATE: pd.date_range("2025-01-01", periods=5), C.FLU_INDEX: range(5)}
    )
    data_mod.merge_weather_frame(cfg, weather)
    data_mod.merge_flu_frame(cfg, flu)
    assert len(data_mod.load_weather(cfg)) == 5
    assert len(data_mod.load_flu(cfg)) == 5
    # Re-merging overlapping dates keeps it de-duplicated.
    data_mod.merge_weather_frame(cfg, weather)
    assert len(data_mod.load_weather(cfg)) == 5
