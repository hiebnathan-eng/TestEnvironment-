"""Live data fetchers for weather and flu activity (no API keys required).

Two free, public sources:

- **Weather** — `Open-Meteo <https://open-meteo.com>`_: historical daily mean
  temperature (ERA5 archive) and a 16-day forecast.
- **Flu** — the `Delphi Epidata <https://delphi.cmu.edu/>`_ mirror of CDC
  **ILINet**: weekly "% influenza-like illness", expanded to daily.

Each fetcher returns a tidy ``DataFrame`` in exactly the schema the rest of the
system expects (``date`` + value column), so the result can be merged straight
into ``weather.csv`` / ``flu_activity.csv``.

The HTTP calls go through the standard library (``urllib``), which honours the
``HTTPS_PROXY`` environment variable and the system CA configuration — so the
same code works on a normal machine and behind a corporate/agent proxy. **It
does require outbound HTTPS** to ``open-meteo.com`` and ``api.delphi.cmu.edu``;
in a locked-down environment those hosts may be blocked (you'll get a clear
:class:`FetchError`), in which case run the fetch where egress is allowed and
copy the CSVs over, or enter data manually.

The network call (`_get_json`) is deliberately separated from the JSON→DataFrame
parsing so the parsing is unit-tested without touching the network.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import ssl
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

from . import config as C

OPEN_METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
DELPHI_FLUVIEW = "https://api.delphi.cmu.edu/epidata/fluview/"


class FetchError(RuntimeError):
    """A live data fetch failed (network blocked, offline, or a bad response)."""


# --- HTTP -------------------------------------------------------------------


def _ssl_context() -> ssl.SSLContext:
    """Default TLS context, honouring an explicit CA bundle if one is configured."""
    cafile = os.environ.get("SSL_CERT_FILE") or os.environ.get("REQUESTS_CA_BUNDLE")
    if cafile and os.path.exists(cafile):
        return ssl.create_default_context(cafile=cafile)
    return ssl.create_default_context()


def _get_json(url: str, timeout: int = 30) -> dict:
    """GET a URL and parse JSON, translating network errors into FetchError.

    Uses the default urllib opener, which picks up `HTTPS_PROXY` from the
    environment automatically.
    """
    req = urllib.request.Request(url, headers={"User-Agent": "census-forecaster/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=_ssl_context()) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:  # server replied with an error status
        raise FetchError(f"HTTP {exc.code} from {url}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        host = urllib.parse.urlsplit(url).netloc
        raise FetchError(
            f"Could not reach {host} ({getattr(exc, 'reason', exc)}). "
            f"Outbound HTTPS to {host} may be blocked by this environment's network "
            "policy (see https://code.claude.com/docs/en/claude-code-on-the-web), "
            "or you are offline. Run this where egress is allowed, or enter data "
            "manually with add-weather/import-weather / add-flu/import-flu."
        ) from exc


# --- Weather (Open-Meteo) ---------------------------------------------------


def _parse_open_meteo_daily(payload: dict) -> pd.DataFrame:
    """Turn an Open-Meteo daily payload into a (date, temp_avg) frame."""
    daily = payload.get("daily")
    if not daily or "time" not in daily or "temperature_2m_mean" not in daily:
        raise FetchError(f"Unexpected Open-Meteo response: {list(payload)[:5]}")
    df = pd.DataFrame(
        {
            C.WEATHER_DATE: pd.to_datetime(daily["time"]),
            C.WEATHER_TEMP: pd.to_numeric(daily["temperature_2m_mean"], errors="coerce"),
        }
    ).dropna()
    return df.reset_index(drop=True)


def fetch_weather_history(
    lat: float, lon: float, start: str, end: str
) -> pd.DataFrame:
    """Fetch daily mean temperature for [start, end] from the Open-Meteo archive."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": str(pd.Timestamp(start).date()),
        "end_date": str(pd.Timestamp(end).date()),
        "daily": "temperature_2m_mean",
        "timezone": "auto",
    }
    url = f"{OPEN_METEO_ARCHIVE}?{urllib.parse.urlencode(params)}"
    return _parse_open_meteo_daily(_get_json(url))


def fetch_weather_forecast(lat: float, lon: float, days: int = 16) -> pd.DataFrame:
    """Fetch the daily mean temperature forecast (Open-Meteo, up to 16 days)."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_mean",
        "forecast_days": int(max(1, min(days, 16))),
        "timezone": "auto",
    }
    url = f"{OPEN_METEO_FORECAST}?{urllib.parse.urlencode(params)}"
    return _parse_open_meteo_daily(_get_json(url))


# --- Flu (Delphi Epidata / CDC ILINet) --------------------------------------


def mmwr_week_start(year: int, week: int) -> _dt.date:
    """Return the Sunday that starts the given MMWR (CDC epidemiological) week.

    MMWR weeks run Sunday–Saturday; week 1 is the first week with at least four
    days in the new year (equivalently, the week containing the first Wednesday).
    """
    jan1 = _dt.date(year, 1, 1)
    # Day-of-week with Sunday = 0.
    sun0 = (jan1.weekday() + 1) % 7
    sunday_on_or_before_jan1 = jan1 - _dt.timedelta(days=sun0)
    # If Jan 1 is Sun–Wed, its week already has >=4 January days => it is week 1.
    if sun0 <= 3:
        week1_start = sunday_on_or_before_jan1
    else:
        week1_start = sunday_on_or_before_jan1 + _dt.timedelta(days=7)
    return week1_start + _dt.timedelta(weeks=week - 1)


def _epiweek_to_date(epiweek: int) -> _dt.date:
    return mmwr_week_start(epiweek // 100, epiweek % 100)


def _parse_delphi_fluview(payload: dict) -> pd.DataFrame:
    """Turn a Delphi fluview payload into a *weekly* (date, flu_index) frame."""
    if payload.get("result") != 1:
        raise FetchError(f"Delphi Epidata error: {payload.get('message', payload)}")
    rows = []
    for entry in payload.get("epidata", []):
        value = entry.get("wili")
        if value is None:
            value = entry.get("ili")
        if value is None:
            continue
        rows.append(
            {
                C.FLU_DATE: pd.Timestamp(_epiweek_to_date(int(entry["epiweek"]))),
                C.FLU_INDEX: float(value),
            }
        )
    return pd.DataFrame(rows, columns=[C.FLU_DATE, C.FLU_INDEX])


def weekly_to_daily(weekly: pd.DataFrame, value_col: str = C.FLU_INDEX) -> pd.DataFrame:
    """Expand a weekly (date, value) frame to daily by repeating each week's value.

    Each row's date is treated as a week start; it becomes 7 consecutive daily
    rows. Useful for flu data, which is reported weekly.
    """
    out = []
    for _, row in weekly.iterrows():
        start = pd.Timestamp(row[C.FLU_DATE]).normalize()
        for k in range(7):
            out.append({C.FLU_DATE: start + pd.Timedelta(days=k), value_col: row[value_col]})
    df = pd.DataFrame(out, columns=[C.FLU_DATE, value_col])
    return (
        df.drop_duplicates(subset=[C.FLU_DATE], keep="last")
        .sort_values(C.FLU_DATE)
        .reset_index(drop=True)
    )


def fetch_flu(
    regions: str = "nat", start_epiweek: int | None = None, end_epiweek: int | None = None
) -> pd.DataFrame:
    """Fetch CDC ILINet flu activity (via Delphi) and expand it to a daily frame.

    `regions` is a Delphi region code, e.g. 'nat' (national) or an HHS region
    like 'hhs2'. Epiweeks are YYYYWW integers; defaults cover ~5 years.
    """
    if end_epiweek is None:
        today = _dt.date.today()
        end_epiweek = today.year * 100 + 52
    if start_epiweek is None:
        start_epiweek = (end_epiweek // 100 - 5) * 100 + 1
    params = {"regions": regions, "epiweeks": f"{start_epiweek}-{end_epiweek}"}
    url = f"{DELPHI_FLUVIEW}?{urllib.parse.urlencode(params)}"
    weekly = _parse_delphi_fluview(_get_json(url))
    return weekly_to_daily(weekly)
