"""Pumping disturbance injection.

Adds anthropogenic pumping-induced drawdown and subsequent recovery to a
synthetic groundwater head series, while recording the exact timing,
duration, and magnitude of every event. The framework's pumping-detection
module (statistical outlier filter on first-difference dh/dt) should
recover these events; the framework's Kalman prediction step should
reconstruct the natural recession behind the masked events.

Drawdown / recovery shape
-------------------------
For each event of duration ``D`` days starting at day ``t0``, the head is
modified as

    h_obs(t) = h_true(t) − S * f(t − t0)

with ``f`` a piecewise function: a half-cosine ramp-down during the
``D``-day pumping period (reaching peak drawdown ``S`` at the end of the
event), followed by an exponential recovery with timescale ``tau_rec``
(default 5 days). This produces the asymmetric "sharp down / slow up"
pattern that distinguishes pumping signatures from natural recession.

Returned alongside the disturbed head series is an event-truth record
that downstream evaluation uses to score the pumping-detection step:

    {
        "events": [
            {"start_day": int, "duration": int,
             "max_drawdown_m": float, "recovery_tau_days": float},
            ...
        ]
    }
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PumpingEvent:
    start_day: int
    duration: int
    max_drawdown_m: float
    recovery_tau_days: float


@dataclass
class PumpingTruth:
    events: list[PumpingEvent] = field(default_factory=list)
    mask: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    """Per-day boolean mask: True where pumping signal is non-negligible."""


def _event_shape(
    n_days_after_start: int,
    duration: int,
    max_drawdown_m: float,
    recovery_tau_days: float,
) -> np.ndarray:
    """Construct the head-reduction shape for one event (length n_days_after_start)."""
    n = n_days_after_start
    out = np.zeros(n)
    if n == 0 or duration <= 0:
        return out

    t = np.arange(n)

    # Ramp-down: half-cosine, 0 → max_drawdown over `duration` days
    ramp_mask = t < duration
    ramp_phase = (t[ramp_mask] / duration) * (np.pi / 2.0)
    out[ramp_mask] = max_drawdown_m * np.sin(ramp_phase)

    # Recovery: exponential decay from max_drawdown
    rec_mask = t >= duration
    if rec_mask.any():
        rec_t = t[rec_mask] - duration
        out[rec_mask] = max_drawdown_m * np.exp(-rec_t / max(recovery_tau_days, 0.5))

    return out


def inject_pumping(
    h_true_m: np.ndarray,
    n_events_per_year: float = 10.0,
    duration_range: tuple[int, int] = (2, 7),
    drawdown_range_m: tuple[float, float] = (0.2, 0.5),
    recovery_tau_range_days: tuple[float, float] = (3.0, 8.0),
    seed: int | None = None,
) -> tuple[np.ndarray, PumpingTruth]:
    """Inject pumping disturbances into a true-head time series.

    Parameters
    ----------
    h_true_m : ndarray of shape (n_days,)
        Synthetic-truth groundwater head (natural dynamics only).
    n_events_per_year : float, default 10.0
        Mean event rate per 365 days (Poisson).
    duration_range : (int, int), default (2, 7)
        Uniform-random pumping duration in days (inclusive).
    drawdown_range_m : (float, float), default (0.2, 0.5)
        Uniform-random peak drawdown in metres.
    recovery_tau_range_days : (float, float), default (3.0, 8.0)
        Uniform-random recovery time-scale in days.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    h_with_pumping : ndarray of shape (n_days,)
        Head series with pumping signal subtracted (drawdown + recovery).
    truth : PumpingTruth
        Event record + per-day non-negligible-effect mask. Used by
        downstream evaluation to score pumping-detection precision/recall.
    """
    rng = np.random.default_rng(seed)
    n_days = len(h_true_m)

    expected = n_events_per_year * (n_days / 365.0)
    n_events = int(rng.poisson(expected))

    events: list[PumpingEvent] = []
    mask = np.zeros(n_days, dtype=bool)
    h_out = h_true_m.copy()

    for _ in range(n_events):
        start_day = int(rng.integers(0, n_days))
        duration = int(rng.integers(duration_range[0], duration_range[1] + 1))
        max_dd = float(rng.uniform(*drawdown_range_m))
        tau_rec = float(rng.uniform(*recovery_tau_range_days))

        ev = PumpingEvent(start_day, duration, max_dd, tau_rec)
        events.append(ev)

        # Apply event shape to h_out
        n_remaining = n_days - start_day
        shape = _event_shape(n_remaining, duration, max_dd, tau_rec)
        h_out[start_day:] -= shape

        # Update mask: True where signal magnitude > 0.02 m (2 cm threshold)
        nonneg = shape > 0.02
        mask[start_day:start_day + len(nonneg)] |= nonneg

    truth = PumpingTruth(events=events, mask=mask)
    return h_out, truth


__all__ = ["inject_pumping", "PumpingEvent", "PumpingTruth"]
