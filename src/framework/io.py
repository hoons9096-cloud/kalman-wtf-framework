"""Data loader for the 3-column SH-format groundwater + rainfall files.

Reads files of the form

    YYYY-MM-DD    gw(m)    rainfall(mm/day)

(whitespace-separated; the date column is optional content but expected
to be present in field files).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class WellTimeSeries:
    """Loaded well time series."""
    name: str
    dates: np.ndarray         # datetime64[D]
    gw_m: np.ndarray          # observed groundwater level (m)
    rain_m: np.ndarray        # rainfall (m/day)  (mm/1000)

    @property
    def n_days(self) -> int:
        return len(self.gw_m)

    @property
    def annual_rain_mm(self) -> float:
        if self.n_days == 0:
            return 0.0
        return float(self.rain_m.sum() * 1000.0 / (self.n_days / 365.0))


def load_well(path: str | Path, name: str | None = None) -> WellTimeSeries:
    """Load a single well from a 3-column whitespace-separated file.

    Tolerates NaN values written as ``NaN`` and ignores trailing whitespace.
    """
    path = Path(path)
    if name is None:
        name = path.stem

    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python",
                     names=["date", "gw", "rain"])

    dates = pd.to_datetime(df["date"], errors="coerce").to_numpy()
    gw = pd.to_numeric(df["gw"], errors="coerce").to_numpy(dtype=float)
    rain_mm = pd.to_numeric(df["rain"], errors="coerce").to_numpy(dtype=float)

    return WellTimeSeries(
        name=name,
        dates=dates,
        gw_m=gw,
        rain_m=rain_mm / 1000.0,
    )


__all__ = ["WellTimeSeries", "load_well"]
