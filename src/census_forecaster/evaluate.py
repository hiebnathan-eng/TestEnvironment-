"""Backtesting: measure how accurate the model actually is on *your* data.

Forecast accuracy is an empirical property, not a promise. This module holds out
the most recent stretch of history, trains on the rest, predicts the held-out
days, and reports error metrics. Run it whenever you add data to see whether the
model is improving — that feedback loop is the point of the whole system.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from . import data as data_mod
from .config import Config
from .features import DemographicsModel, ExogenousData, build_feature_matrix
from .model import RidgeModel


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    err = predicted - actual
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    # Mean absolute percentage error, guarding against zero-census days.
    nonzero = actual != 0
    mape = (
        float(np.mean(np.abs(err[nonzero] / actual[nonzero])) * 100.0)
        if np.any(nonzero)
        else float("nan")
    )
    return {
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "mape_pct": round(mape, 2),
        "bias": round(float(np.mean(err)), 2),
    }


def backtest(cfg: Config, holdout_days: int = 30) -> dict:
    """Train on all but the last `holdout_days`, then score on those days.

    Returns a dict of error metrics plus the holdout window for context. Lower
    MAE/RMSE/MAPE is better; `bias` near zero means no systematic over/under-call.
    """
    census = data_mod.load_census(cfg)
    if len(census) < holdout_days + 14:
        raise ValueError(
            f"Need at least {holdout_days + 14} days for a {holdout_days}-day "
            f"backtest; have {len(census)}."
        )
    census = census.sort_values(C.CENSUS_DATE).reset_index(drop=True)
    train_df = census.iloc[:-holdout_days]
    test_df = census.iloc[-holdout_days:]

    demo = data_mod.load_demographics(cfg)
    holidays = data_mod.load_holidays(cfg)
    demo_model = DemographicsModel.from_frame(demo)
    exog = ExogenousData.from_frames(
        data_mod.load_weather(cfg), data_mod.load_flu(cfg)
    )

    X_tr, _ = build_feature_matrix(
        pd.DatetimeIndex(train_df[C.CENSUS_DATE]),
        demo_model,
        holidays,
        cfg.yearly_harmonics,
        exog,
    )
    y_tr = train_df[C.CENSUS_VALUE].to_numpy(dtype=float)
    model = RidgeModel(alpha=cfg.ridge_alpha).fit(X_tr, y_tr)

    X_te, _ = build_feature_matrix(
        pd.DatetimeIndex(test_df[C.CENSUS_DATE]),
        demo_model,
        holidays,
        cfg.yearly_harmonics,
        exog,
    )
    pred = model.predict(X_te)
    actual = test_df[C.CENSUS_VALUE].to_numpy(dtype=float)

    result = _metrics(actual, pred)
    result["holdout_days"] = holdout_days
    result["holdout_start"] = test_df[C.CENSUS_DATE].min().date().isoformat()
    result["holdout_end"] = test_df[C.CENSUS_DATE].max().date().isoformat()
    result["train_days"] = int(len(train_df))
    return result
