# census-forecaster

A hospital **census prediction** system. It learns from your historical daily
census logs and local demographics, then forecasts future census at three
granularities — **day-to-day, month-to-month, and year-to-year** — each with an
uncertainty interval. It is built to be *fed continuously*: append new
observations over time and re-run, and the model retrains on everything it has.

> **What "census" means here:** the number of occupied beds (patients in house)
> on a given day. That daily number is what the model predicts; monthly and
> yearly views are aggregated from the daily forecast so all three stay
> consistent.

## How it works (in plain terms)

The model combines the signals that actually move hospital census:

- **Calendar rhythm** — weekends differ from weekdays, and there is a smooth
  within-year season (winter peaks, summer troughs) captured with Fourier terms.
- **Long-run trend** — gradual drift not explained by demographics.
- **Community structure** — local population size, median age, and the share of
  residents over 65, which scale and shift demand year over year.
- **Weather and flu (optional)** — when you supply daily temperature and/or a
  flu-activity index, the model learns how census responds to *unusual* weather
  and flu (see below).

### Weather and flu: how they help (and their limits)

The recurring flu season and the average winter-weather effect are *already*
captured by the seasonal cycle above. What the weather and flu inputs add is the
**deviation from normal** — a colder-than-usual cold snap, a worse-than-usual flu
season. The model learns those sensitivities (e.g. "colder → busier", "bad flu
year → busier") and applies them.

This sharpens the **near-term** forecast most, because that is where you actually
have the information (real weather forecasts run ~2 weeks; current flu activity is
known). Far into the future nobody knows the weather, so the model falls back to
"typical for that day" and reverts to the seasonal baseline — the honest default.
On the bundled sample data, adding weather + flu cut held-out error by ~25%.

It fits a **ridge regression** (a regularised linear model): stable with limited
data, fast to retrain, and interpretable — every coefficient says how much a
factor moves census. As your history grows you can swap in a richer model behind
the same interface without changing the rest of the system.

> **Accuracy depends on your data.** No model is accurate in the abstract — it
> gets accurate by learning from *your* real history. Use `evaluate` (below) to
> measure error on held-out days, and watch it improve as you add data.

## Requirements

- Python 3.10+
- Dependencies: `numpy`, `pandas`, `matplotlib` (installed below)

## Setup

```sh
python3 -m pip install -e .          # install the package + dependencies
# or, just the libraries:
python3 -m pip install numpy pandas matplotlib
```

This installs a `census-forecast` command. You can also run it without
installing via `PYTHONPATH=src python3 -m census_forecaster ...`.

## Quick start

```sh
census-forecast generate-sample              # create 4 years of example data
census-forecast info                         # summarise the data on hand
census-forecast evaluate --holdout-days 30   # measure accuracy on held-out days
census-forecast forecast --horizon-days 90 --granularity all \
    --output-dir data/forecasts --plot data/forecasts/forecast.png
```

`--granularity` accepts `daily`, `monthly`, `yearly`, or `all`.

## Feeding it your own data

Two CSV files under `data/` drive everything. Replace the sample files with your
real exports (same columns) or append incrementally:

**`data/census_log.csv`** — one row per day:

| column       | meaning                                  | required |
|--------------|------------------------------------------|----------|
| `date`       | `YYYY-MM-DD`                             | yes      |
| `census`     | occupied beds that day                   | yes      |
| `admissions` | admissions that day (stored, optional)   | no       |
| `discharges` | discharges that day (stored, optional)   | no       |

**`data/demographics.csv`** — one row per year:

| column        | meaning                                   | required |
|---------------|-------------------------------------------|----------|
| `year`        | calendar year                             | yes      |
| `population`  | catchment population                      | yes      |
| `median_age`  | median age of the community               | yes      |
| `pct_over_65` | fraction over 65 (e.g. `0.18`)            | yes      |
| `growth_rate` | fractional annual growth (e.g. `0.015`)   | yes      |

Add data from the command line (both append-or-overwrite by key):

```sh
census-forecast add-census --date 2026-01-15 --census 142 --admissions 20 --discharges 17
census-forecast add-demographics --year 2026 --population 128000 \
    --median-age 39.2 --pct-over-65 0.176 --growth-rate 0.016
census-forecast import-census my_hospital_export.csv   # bulk-merge a CSV by date
```

**Optional extra signals** (each a simple `date,value` CSV; supply what you have):

- `data/weather.csv` — columns `date,temp_avg` (daily average temperature; any
  consistent unit). To improve the *near-term* forecast, include the next ~2
  weeks of forecasted temperatures.
- `data/flu_activity.csv` — columns `date,flu_index` (e.g. CDC ILINet "% ILI" or
  a 0–10 activity level). Include the latest known activity to inform the
  short-range forecast.
- `data/holidays.csv` — columns `date,name`; lets the model treat holidays specially.

```sh
census-forecast add-weather --date 2026-01-15 --temp-avg 28.4
census-forecast add-flu     --date 2026-01-15 --flu-index 6.1
census-forecast import-weather noaa_export.csv      # bulk-merge by date
census-forecast import-flu     cdc_ilinet.csv
```

> Flu data is often weekly (e.g. CDC ILINet). Use `import-flu --weekly` to expand
> a weekly `date,flu_index` file to daily automatically:
>
> ```sh
> census-forecast import-flu cdc_weekly.csv --weekly   # each row -> 7 daily rows
> ```

### Fetching live data automatically

Instead of supplying files, you can pull weather and flu directly from free
public sources (no API key needed):

```sh
# Weather: Open-Meteo (history for your census range + a 16-day forecast)
census-forecast fetch-weather --lat 40.71 --lon -74.01

# Flu: CDC ILINet via the Delphi Epidata API (national, last ~5 years)
census-forecast fetch-flu --region nat
```

Both merge straight into your `weather.csv` / `flu_activity.csv`, so just re-run
`forecast` afterwards.

> **Network required.** These commands need outbound HTTPS to `open-meteo.com`
> and `api.delphi.cmu.edu`. Some managed/sandboxed environments (including Claude
> Code on the web under a restrictive network policy) block external hosts — you
> will get a clear error pointing here:
> https://code.claude.com/docs/en/claude-code-on-the-web . If so, run the fetch
> on a machine with internet and copy the CSVs over, or relax the environment's
> network policy. Manual `add-*` / `import-*` entry always works offline.

### Seeing what the model learned

```sh
census-forecast explain
```

Reports day-of-week effects, the seasonal peak/trough months, how census
correlates with temperature and flu, and the model's top drivers — so the
"patterns" it uses are visible rather than a black box.

## Measuring accuracy

```sh
census-forecast evaluate --holdout-days 30
```

Trains on all but the last 30 days, predicts those days, and reports **MAE**
(average miss in patients/day), **RMSE**, **MAPE** (average miss as a percent),
and **bias** (≈ 0 means no systematic over/under-prediction). Re-run it as you
add data to confirm the model is improving.

## Development

```sh
python3 -m pytest          # run the test suite
```

## Layout

- `src/census_forecaster/`
  - `config.py` — paths, column names, model hyperparameters
  - `data.py` — load / validate / append / import (the continuous-update workflow)
  - `sample_data.py` — realistic synthetic data generator (census, demographics, weather, flu)
  - `features.py` — calendar, seasonal, demographic, and weather/flu feature engineering
  - `model.py` — ridge regression with prediction intervals
  - `forecast.py` — daily prediction + monthly/yearly aggregation
  - `evaluate.py` — backtesting / accuracy metrics
  - `patterns.py` — the `explain` command: surfaces learned patterns
  - `sources.py` — live weather (Open-Meteo) and flu (CDC/Delphi) fetchers
  - `plot.py` — optional charting
  - `cli.py` — command-line interface
- `data/` — CSV inputs (sample data committed; generated forecasts are ignored)
- `tests/` — pytest suite
```
