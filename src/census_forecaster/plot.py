"""Optional charting of history + forecast (matplotlib).

Kept separate so the core model has no hard dependency on a plotting backend.
Uses the non-interactive "Agg" backend so it works in headless environments.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import config as C  # noqa: E402


def plot_forecast(history, daily_forecast, out_path: str | Path) -> Path:
    """Plot recent history and the daily forecast with its interval band."""
    out_path = Path(out_path)
    fig, ax = plt.subplots(figsize=(12, 5))

    # Show at most the last ~2 years of history for readability.
    hist = history.tail(730)
    ax.plot(
        hist[C.CENSUS_DATE],
        hist[C.CENSUS_VALUE],
        color="#444",
        lw=0.9,
        label="History",
    )

    ax.plot(
        daily_forecast["date"],
        daily_forecast["predicted_census"],
        color="#1f77b4",
        lw=1.4,
        label="Forecast",
    )
    ax.fill_between(
        daily_forecast["date"],
        daily_forecast["lower"],
        daily_forecast["upper"],
        color="#1f77b4",
        alpha=0.2,
        label="Prediction interval",
    )

    ax.set_title("Hospital census: history and forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Census (occupied beds)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
