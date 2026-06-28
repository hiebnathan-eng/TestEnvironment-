"""Hospital census forecasting.

A small, extensible system that learns from your historical daily census logs
and local demographic data, then predicts future census at day, month, and year
granularity with uncertainty intervals.

The design goal is *continuous improvement*: you append new observations over
time (see :mod:`census_forecaster.data`) and re-run the forecast — the model
retrains on everything it has and gets sharper as real history accumulates.

Public entry points:

- :func:`census_forecaster.forecast.forecast` — produce predictions.
- :func:`census_forecaster.evaluate.backtest` — measure accuracy on held-out days.
- :mod:`census_forecaster.data` — load and append data.
"""

from .config import Config

__all__ = ["Config"]
__version__ = "0.1.0"
