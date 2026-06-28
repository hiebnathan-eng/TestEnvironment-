# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

`census-forecaster` is a hospital **census prediction** system. It learns from
historical daily census logs, local demographics, and optional weather/flu
signals, then forecasts future census (occupied beds) at **daily, monthly, and
yearly** granularity, each with an uncertainty interval. It is designed for
*continuous improvement*: users append new observations over time and re-run; the
model retrains on all data on hand.

- **Language:** Python (3.10+).
- **Dependencies:** `numpy`, `pandas`, `matplotlib`. Tests use `pytest`.
- **No network or services** — everything runs locally against CSV files.

## Setup

```sh
python3 -m pip install -e .            # package + runtime deps
python3 -m pip install pytest          # for the test suite
```

If the package is not installed, run via `PYTHONPATH=src python3 -m census_forecaster …`.

## Common commands

- **Run (installed):** `census-forecast <subcommand>` (e.g. `census-forecast info`)
- **Run (in-tree):** `PYTHONPATH=src python3 -m census_forecaster <subcommand>`
- **Generate sample data:** `census-forecast generate-sample --years 4`
- **Forecast:** `census-forecast forecast --horizon-days 90 --granularity all`
- **Accuracy backtest:** `census-forecast evaluate --holdout-days 30`
- **Test (all):** `python3 -m pytest`
- **Test (single):** `python3 -m pytest tests/test_forecast_evaluate.py -k seasonality`

## Architecture

Logic lives in the `census_forecaster` package under `src/` (src layout). The
data flow is: **CSV data → features → model → daily forecast → aggregation**.

- `config.py` — single source of truth for file paths, CSV **column names**, and
  model hyperparameters (`Config`). Nothing else hard-codes a path or magic number.
- `data.py` — load/validate/append/import. Append helpers de-duplicate by key
  (date / year, keep-last) so re-importing overlapping data is safe. This module
  is the "keep adding information" workflow.
- `sample_data.py` — synthetic data generator. It deliberately embeds the
  structure the model is meant to recover (demographic base level, winter-peaking
  season, weekend dips, trend, holiday dips, noise).
- `features.py` — builds the numeric design matrix: linear trend, day-of-week
  one-hot (Sunday = reference), yearly Fourier seasonal terms, holiday flag,
  demographic covariates, and optional weather/flu signals. `DemographicsModel`
  interpolates yearly demographics to daily values and **extrapolates** future
  years. `ExogenousSeries`/`ExogenousData` turn weather/flu into **anomaly**
  features (value minus the learned day-of-year climatology); unknown/future
  dates get anomaly 0, so the forecast cleanly reverts to the seasonal baseline.
- `model.py` — `RidgeModel`: ridge regression with an **unpenalised intercept**
  and internal feature standardisation; exposes `fit`/`predict` and a residual
  std used for prediction intervals.
- `forecast.py` — `train()` fits on all history; `forecast()` returns a daily
  prediction frame; `aggregate()` rolls daily predictions up to month/year
  (average census, patient-days, peak). Aggregating from one daily model keeps
  the three granularities consistent.
- `evaluate.py` — `backtest()` holds out the most recent N days, trains on the
  rest, and reports MAE/RMSE/MAPE/bias.
- `patterns.py` — `analyze()` powers the `explain` command: observed day-of-week
  and month effects, census↔weather/flu correlations, and the model's top
  standardised coefficients.
- `plot.py` — optional matplotlib chart (uses the headless `Agg` backend).
- `cli.py` — argparse CLI; `main()` translates `ValueError`/`FileNotFoundError`
  into clean messages instead of tracebacks.

### Key design choices (the *why*)

- **Daily model is the source of truth.** Monthly/yearly numbers are aggregates
  of daily predictions, never separately fit — this guarantees consistency.
- **Ridge regression**, not a heavyweight forecaster: stable on short history,
  fast to retrain (the core of the add-data-and-rerun loop), and interpretable.
  The `fit`/`predict` interface is the seam to swap in a richer model later.
- **Aggregate intervals** approximate the effective sample size as the number of
  weeks (not days) to stay honest about census autocorrelation
  (`forecast._interval_for_mean`).
- **Weather/flu enter as anomalies, not raw values.** The average seasonal effect
  is already in the Fourier terms; the *deviation from normal* is the only new
  information, so raw values would be redundant/collinear. Future dates with no
  supplied signal get anomaly 0 (reverts to baseline) — this is why these signals
  help the near term, not the multi-year horizon. Sign convention: `temp_anomaly`
  is (temperature − normal), so a **negative** weight means colder → busier.

## Conventions

- Keep CSV column names in `config.py`; reference them via the `C.*` constants,
  never string literals scattered through the code.
- Add tests under `tests/` alongside new behaviour. Fixtures in `conftest.py`
  provide an isolated temp data dir (`cfg`) and a sample-populated one (`seeded_cfg`).
- Sample/input CSVs in `data/` are tracked; generated forecasts
  (`data/forecasts/`, PNGs) are git-ignored.
- Run `python3 -m pytest` before committing.

## Working agreements

- Keep this file accurate. When you change how the project is built, run, or
  tested — or the CSV schemas — update the relevant section here in the same change.
- Prefer documenting the *why* and the non-obvious over restating what is
  already clear from reading the code.
