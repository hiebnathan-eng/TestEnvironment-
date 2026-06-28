"""Command-line interface for the census forecaster.

Run ``census-forecast --help`` (or ``python -m census_forecaster --help``) to see
all commands. Typical first session::

    census-forecast generate-sample      # create example data to play with
    census-forecast info                 # see what data is on hand
    census-forecast evaluate             # how accurate is it on held-out days?
    census-forecast forecast --horizon-days 90 --granularity all

As real data arrives, keep feeding it in and re-running::

    census-forecast add-census --date 2026-01-15 --census 142
    census-forecast import-census my_export.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from . import data as data_mod
from . import sample_data
from .config import Config
from .evaluate import backtest
from .forecast import aggregate, forecast


def _cfg(args: argparse.Namespace) -> Config:
    return Config(
        data_dir=Path(args.data_dir),
        yearly_harmonics=args.harmonics,
        ridge_alpha=args.alpha,
    )


def _print_df(df: pd.DataFrame, max_rows: int = 40) -> None:
    with pd.option_context(
        "display.max_rows", max_rows, "display.width", 120, "display.max_columns", 20
    ):
        print(df.to_string(index=False))


def cmd_generate_sample(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    census, demo = sample_data.generate(cfg, years=args.years)
    print(
        f"Wrote {len(census)} census rows -> {cfg.census_path}\n"
        f"Wrote {len(demo)} demographic rows -> {cfg.demographics_path}\n"
        f"Wrote holidays -> {cfg.holidays_path}"
    )
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    summary = data_mod.data_summary(cfg)
    if summary["census_rows"] == 0:
        print("No census data yet. Run `census-forecast generate-sample` to start.")
        return 0
    print("Data summary")
    print("------------")
    for k, v in summary.items():
        print(f"{k:>20}: {v}")
    return 0


def cmd_add_census(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    df = data_mod.append_census(
        cfg, args.date, args.census, args.admissions, args.discharges
    )
    print(f"Recorded census={args.census} on {args.date}. Log now has {len(df)} days.")
    return 0


def cmd_add_demographics(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    df = data_mod.append_demographics(
        cfg,
        args.year,
        args.population,
        args.median_age,
        args.pct_over_65,
        args.growth_rate,
    )
    print(f"Recorded demographics for {args.year}. Table now has {len(df)} years.")
    return 0


def cmd_import_census(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    df = data_mod.import_census_csv(cfg, args.path)
    print(f"Imported {args.path}. Census log now has {len(df)} days.")
    return 0


def cmd_add_weather(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    df = data_mod.append_weather(cfg, args.date, args.temp_avg)
    print(f"Recorded temp={args.temp_avg} on {args.date}. Weather now has {len(df)} days.")
    return 0


def cmd_add_flu(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    df = data_mod.append_flu(cfg, args.date, args.flu_index)
    print(f"Recorded flu_index={args.flu_index} on {args.date}. Flu data now has {len(df)} days.")
    return 0


def cmd_import_weather(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    df = data_mod.import_weather_csv(cfg, args.path)
    print(f"Imported {args.path}. Weather now has {len(df)} days.")
    return 0


def cmd_import_flu(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    if args.weekly:
        from . import sources

        weekly = pd.read_csv(args.path, parse_dates=[data_mod.C.FLU_DATE])
        if data_mod.C.FLU_INDEX not in weekly.columns:
            print(f"Error: {args.path} must have 'date' and 'flu_index' columns")
            return 1
        daily = sources.weekly_to_daily(weekly)
        df = data_mod.merge_flu_frame(cfg, daily)
        print(f"Imported {args.path} (weekly -> daily). Flu data now has {len(df)} days.")
    else:
        df = data_mod.import_flu_csv(cfg, args.path)
        print(f"Imported {args.path}. Flu data now has {len(df)} days.")
    return 0


def cmd_fetch_weather(args: argparse.Namespace) -> int:
    import datetime

    from . import sources

    cfg = _cfg(args)
    frames = []
    try:
        if not args.no_history:
            census = data_mod.load_census(cfg)
            if args.start:
                start = args.start
            elif len(census):
                start = census[data_mod.C.CENSUS_DATE].min().date().isoformat()
            else:
                start = (datetime.date.today() - datetime.timedelta(days=365 * 4)).isoformat()
            end = args.end or datetime.date.today().isoformat()
            print(f"Fetching weather history {start} -> {end} ...")
            frames.append(sources.fetch_weather_history(args.lat, args.lon, start, end))
        if args.forecast_days > 0:
            print(f"Fetching {args.forecast_days}-day weather forecast ...")
            frames.append(sources.fetch_weather_forecast(args.lat, args.lon, args.forecast_days))
    except sources.FetchError as exc:
        print(f"Error: {exc}")
        return 1

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if combined.empty:
        print("Nothing fetched (try removing --no-history or setting --forecast-days).")
        return 1
    df = data_mod.merge_weather_frame(cfg, combined)
    print(f"Saved -> {cfg.weather_path}. Weather now has {len(df)} days.")
    return 0


def cmd_fetch_flu(args: argparse.Namespace) -> int:
    import datetime

    from . import sources

    cfg = _cfg(args)
    end_year = args.end_year or datetime.date.today().year
    start_year = args.start_year or (end_year - 5)
    try:
        print(f"Fetching flu activity for region '{args.region}', {start_year}-{end_year} ...")
        daily = sources.fetch_flu(
            regions=args.region,
            start_epiweek=start_year * 100 + 1,
            end_epiweek=end_year * 100 + 52,
        )
    except sources.FetchError as exc:
        print(f"Error: {exc}")
        return 1
    if daily.empty:
        print("No flu data returned for that region/range.")
        return 1
    df = data_mod.merge_flu_frame(cfg, daily)
    print(f"Saved -> {cfg.flu_path}. Flu data now has {len(df)} days.")
    return 0


def cmd_explain(args: argparse.Namespace) -> int:
    from .patterns import analyze

    cfg = _cfg(args)
    result = analyze(cfg)
    print("Patterns the model has learned")
    print("------------------------------")
    print(f"Overall mean census: {result['overall_mean_census']}")

    dow = result.get("day_of_week_effect", {})
    if dow:
        print("\nDay-of-week effect (vs. average):")
        for day, val in dow.items():
            print(f"  {day}: {val:+.1f}")

    if "seasonal_peak_month" in result:
        print(
            f"\nSeasonal peak month: {result['seasonal_peak_month']} "
            f"| trough month: {result['seasonal_trough_month']}"
        )

    if "census_vs_temperature_corr" in result:
        print(f"\nCensus vs. temperature correlation: {result['census_vs_temperature_corr']:+.3f}")
    if "census_vs_flu_corr" in result:
        print(f"Census vs. flu correlation:          {result['census_vs_flu_corr']:+.3f}")
    if "weather_sensitivity" in result:
        # Feature is (temperature - seasonal normal); a negative weight => colder days busier.
        sign = "colder->busier" if result["weather_sensitivity"] < 0 else "warmer->busier"
        print(f"Weather sensitivity (model): {result['weather_sensitivity']:+.2f}  ({sign})")
    if "flu_sensitivity" in result:
        print(f"Flu sensitivity (model):     {result['flu_sensitivity']:+.2f}  (higher flu -> busier)")

    print("\nTop drivers (standardised model weights):")
    for name, val in result["top_drivers"]:
        print(f"  {name:>16}: {val:+.2f}")
    return 0


def cmd_forecast(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    daily, bundle = forecast(cfg, horizon_days=args.horizon_days, start=args.start)
    gran = args.granularity

    outputs: dict[str, pd.DataFrame] = {}
    if gran in ("daily", "all"):
        outputs["daily"] = daily
    if gran in ("monthly", "all"):
        outputs["monthly"] = aggregate(daily, bundle, "M")
    if gran in ("yearly", "all"):
        outputs["yearly"] = aggregate(daily, bundle, "Y")

    for name, df in outputs.items():
        print(f"\n=== {name.upper()} forecast ===")
        _print_df(df)
        if args.output_dir:
            out = Path(args.output_dir)
            out.mkdir(parents=True, exist_ok=True)
            path = out / f"forecast_{name}.csv"
            df.to_csv(path, index=False)
            print(f"(saved -> {path})")

    if args.plot:
        from . import plot as plot_mod

        plot_path = Path(args.plot)
        plot_mod.plot_forecast(bundle.history, daily, plot_path)
        print(f"\nSaved chart -> {plot_path}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    result = backtest(cfg, holdout_days=args.holdout_days)
    print("Backtest accuracy (lower error is better)")
    print("-----------------------------------------")
    for k, v in result.items():
        print(f"{k:>15}: {v}")
    print(
        "\nMAE = avg miss in patients/day; MAPE = avg miss as % of census; "
        "bias ≈ 0 means no systematic over/under-prediction."
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="census-forecast",
        description="Hospital census prediction: daily, monthly, and yearly forecasts.",
    )
    p.add_argument("--data-dir", default="data", help="Directory for CSV data (default: data)")
    p.add_argument("--harmonics", type=int, default=4, help="Yearly seasonal Fourier harmonics")
    p.add_argument("--alpha", type=float, default=5.0, help="Ridge regularisation strength")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("generate-sample", help="Write realistic example data")
    sp.add_argument("--years", type=int, default=4, help="Years of history to generate")
    sp.set_defaults(func=cmd_generate_sample)

    sp = sub.add_parser("info", help="Summarise the data currently on hand")
    sp.set_defaults(func=cmd_info)

    sp = sub.add_parser("add-census", help="Append/overwrite one day's census")
    sp.add_argument("--date", required=True, help="YYYY-MM-DD")
    sp.add_argument("--census", type=float, required=True)
    sp.add_argument("--admissions", type=float, default=None)
    sp.add_argument("--discharges", type=float, default=None)
    sp.set_defaults(func=cmd_add_census)

    sp = sub.add_parser("add-demographics", help="Append/overwrite one year's demographics")
    sp.add_argument("--year", type=int, required=True)
    sp.add_argument("--population", type=float, required=True)
    sp.add_argument("--median-age", type=float, required=True)
    sp.add_argument("--pct-over-65", type=float, required=True, help="Fraction, e.g. 0.18")
    sp.add_argument("--growth-rate", type=float, required=True, help="Fraction, e.g. 0.015")
    sp.set_defaults(func=cmd_add_demographics)

    sp = sub.add_parser("import-census", help="Bulk-import a census CSV (merges by date)")
    sp.add_argument("path", help="CSV with at least date,census columns")
    sp.set_defaults(func=cmd_import_census)

    sp = sub.add_parser("add-weather", help="Append/overwrite one day's avg temperature")
    sp.add_argument("--date", required=True, help="YYYY-MM-DD")
    sp.add_argument("--temp-avg", type=float, required=True)
    sp.set_defaults(func=cmd_add_weather)

    sp = sub.add_parser("add-flu", help="Append/overwrite one day's flu activity index")
    sp.add_argument("--date", required=True, help="YYYY-MM-DD")
    sp.add_argument("--flu-index", type=float, required=True)
    sp.set_defaults(func=cmd_add_flu)

    sp = sub.add_parser("import-weather", help="Bulk-import weather CSV (date,temp_avg)")
    sp.add_argument("path")
    sp.set_defaults(func=cmd_import_weather)

    sp = sub.add_parser("import-flu", help="Bulk-import flu CSV (date,flu_index)")
    sp.add_argument("path")
    sp.add_argument(
        "--weekly",
        action="store_true",
        help="Treat each row's date as a week start and expand to 7 daily rows",
    )
    sp.set_defaults(func=cmd_import_flu)

    sp = sub.add_parser(
        "fetch-weather", help="Download weather from Open-Meteo (needs internet)"
    )
    sp.add_argument("--lat", type=float, required=True, help="Latitude")
    sp.add_argument("--lon", type=float, required=True, help="Longitude")
    sp.add_argument("--start", default=None, help="History start YYYY-MM-DD (default: census start)")
    sp.add_argument("--end", default=None, help="History end YYYY-MM-DD (default: today)")
    sp.add_argument("--forecast-days", type=int, default=16, help="Days of forecast to add (0=none)")
    sp.add_argument("--no-history", action="store_true", help="Skip history; fetch only the forecast")
    sp.set_defaults(func=cmd_fetch_weather)

    sp = sub.add_parser(
        "fetch-flu", help="Download CDC ILINet flu activity via Delphi (needs internet)"
    )
    sp.add_argument("--region", default="nat", help="Delphi region code, e.g. 'nat' or 'hhs2'")
    sp.add_argument("--start-year", type=int, default=None)
    sp.add_argument("--end-year", type=int, default=None)
    sp.set_defaults(func=cmd_fetch_flu)

    sp = sub.add_parser("explain", help="Report the patterns the model has learned")
    sp.set_defaults(func=cmd_explain)

    sp = sub.add_parser("forecast", help="Predict future census")
    sp.add_argument("--horizon-days", type=int, default=90, help="Days ahead to forecast")
    sp.add_argument(
        "--granularity",
        choices=["daily", "monthly", "yearly", "all"],
        default="daily",
    )
    sp.add_argument("--start", default=None, help="First forecast date (default: day after last data)")
    sp.add_argument("--output-dir", default=None, help="Directory to save forecast CSVs")
    sp.add_argument("--plot", default=None, help="Path to save a PNG chart")
    sp.set_defaults(func=cmd_forecast)

    sp = sub.add_parser("evaluate", help="Backtest accuracy on held-out days")
    sp.add_argument("--holdout-days", type=int, default=30)
    sp.set_defaults(func=cmd_evaluate)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, FileNotFoundError) as exc:
        # Expected, user-actionable problems: show a clean message, not a traceback.
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
