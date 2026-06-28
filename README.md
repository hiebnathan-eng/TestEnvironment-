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

`data/holidays.csv` (optional, columns `date,name`) lets the model treat
holidays specially.

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
  - `sample_data.py` — realistic synthetic data generator
  - `features.py` — calendar, seasonal, and demographic feature engineering
  - `model.py` — ridge regression with prediction intervals
  - `forecast.py` — daily prediction + monthly/yearly aggregation
  - `evaluate.py` — backtesting / accuracy metrics
  - `plot.py` — optional charting
  - `cli.py` — command-line interface
- `data/` — CSV inputs (sample data committed; generated forecasts are ignored)
- `tests/` — pytest suite
```
