"""Event-based WTF recharge estimation (Phase A-3).

The continuous Nelder-Mead optimisation in the v1 framework (and in
optim_v2.py) minimises an aggregate RMSE objective across the whole
record. This objective is dominated by the long dry-season recession
phases and tends to be satisfied by a low-Sy / slow-recession solution
that under-estimates the recharge magnitude.

The event-based approach, following Healy (2010, §5.2) and Crosbie et
al. (2005), instead isolates *individual rainfall–response events* and
applies the WTF identity locally:

    Rch_event = Sy × Δh_event

where Δh_event is the *extrapolated* head rise — the height between
the post-event peak and the antecedent recession curve extrapolated to
the peak time. Extrapolation removes the contribution of the
background recession from the observed rise, isolating the
recharge-only component.

Per-event recharge is summed across all detected events to give the
total annual recharge. Sy can be supplied as (a) a fixed constant
(e.g. from a Bayesian posterior or BE-derived value), (b) the
operational Sy estimated by the continuous framework, or (c) a daily
varying series from `filpor_tr`.

This estimate is structurally different from the continuous one and
provides an independent recharge value. If they agree, the framework
is internally consistent; if they diverge, the discrepancy localises
the bias to either the objective function (continuous) or the event
detection / recession extrapolation (event-based).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RechargeEvent:
    start_day: int          # day index of rainfall onset
    peak_day: int           # day index of head peak
    pre_event_h: float      # head at start_day (m)
    peak_h: float           # head at peak_day (m)
    extrapolated_h: float   # antecedent recession extrapolated to peak_day
    delta_h: float          # peak_h - extrapolated_h (m)
    rainfall_mm: float      # cumulative rainfall in event window
    sy_used: float          # Sy value used for this event
    recharge_mm: float      # delta_h × sy × 1000

    @property
    def is_meaningful(self) -> bool:
        return self.delta_h > 0 and self.recharge_mm > 0


def detect_events(
    rain_mm: np.ndarray,
    head_m: np.ndarray,
    rain_threshold_mm: float = 5.0,
    min_event_gap_days: int = 5,
    peak_search_days: int = 10,
) -> list[tuple[int, int]]:
    """Detect (start_day, peak_day) pairs for candidate recharge events.

    A candidate event starts when daily rainfall first exceeds the
    threshold after at least `min_event_gap_days` of dry weather. The
    peak is the maximum head within `peak_search_days` after the event
    start.
    """
    n = len(rain_mm)
    events = []
    last_event_end = -min_event_gap_days

    for i in range(n - 1):
        if rain_mm[i] < rain_threshold_mm:
            continue
        if i - last_event_end < min_event_gap_days:
            continue

        end = min(i + peak_search_days, n - 1)
        peak_i = i
        peak_val = head_m[i] if not np.isnan(head_m[i]) else -np.inf
        for j in range(i + 1, end + 1):
            if np.isnan(head_m[j]):
                continue
            if head_m[j] > peak_val:
                peak_val = head_m[j]
                peak_i = j

        if peak_i > i and not np.isnan(head_m[i]):
            events.append((i, peak_i))
            last_event_end = peak_i

    return events


def fit_antecedent_recession(
    head_m: np.ndarray,
    event_start: int,
    lookback_days: int = 30,
    min_points: int = 5,
) -> tuple[float, float] | None:
    """Fit an exponential recession h(t) = h_base + a · exp(-k·t) to
    the lookback period before the event start. Returns (a, k) where t
    is measured backward from the event start, or None if the fit fails.
    """
    lo = max(0, event_start - lookback_days)
    h_pre = head_m[lo:event_start]
    if np.sum(~np.isnan(h_pre)) < min_points:
        return None

    days = np.arange(len(h_pre)) - len(h_pre)
    valid = ~np.isnan(h_pre)
    days = days[valid]
    h = h_pre[valid]

    if np.std(h) < 1e-6:
        return None

    h_base = float(np.min(h)) - 0.05
    h_above = h - h_base
    if np.any(h_above <= 0):
        return None

    log_h = np.log(h_above)
    A = np.column_stack([np.ones_like(days), days.astype(float)])
    try:
        sol, *_ = np.linalg.lstsq(A, log_h, rcond=None)
    except np.linalg.LinAlgError:
        return None
    log_a, k = sol
    return h_base + float(np.exp(log_a)), float(k)


def extrapolate_recession(
    head_m: np.ndarray,
    event_start: int,
    peak_day: int,
    lookback_days: int = 30,
) -> float | None:
    """Extrapolate the antecedent recession from event_start to peak_day.

    Returns the projected head at peak_day, or None if extrapolation
    fails. Uses a simple linear fit on the last 5 valid points if the
    exponential fit is unreliable.
    """
    lo = max(0, event_start - lookback_days)
    h_pre = head_m[lo:event_start]
    valid = ~np.isnan(h_pre)
    if np.sum(valid) < 3:
        return None

    days_pre = np.arange(len(h_pre))[valid].astype(float)
    h_valid = h_pre[valid]

    # Linear fit (robust and adequate for short windows)
    A = np.column_stack([np.ones_like(days_pre), days_pre])
    sol, *_ = np.linalg.lstsq(A, h_valid, rcond=None)
    intercept, slope = sol
    # Project to peak_day; days_pre values are 0..lookback_days-1 from lo
    days_to_peak = (peak_day - lo)
    return float(intercept + slope * days_to_peak)


def event_based_recharge(
    rain_mm: np.ndarray,
    head_m: np.ndarray,
    sy: float | np.ndarray,
    rain_threshold_mm: float = 5.0,
    min_event_gap_days: int = 5,
    peak_search_days: int = 10,
    lookback_days: int = 30,
) -> tuple[float, list[RechargeEvent]]:
    """Total event-based annual recharge (mm/yr) and list of events.

    Parameters
    ----------
    rain_mm : ndarray
        Daily rainfall in mm.
    head_m : ndarray
        Daily observed head in m.
    sy : float or ndarray
        Specific yield to use. Scalar → constant; ndarray → daily series.
    """
    if np.isscalar(sy):
        sy_series = np.full(len(head_m), float(sy))
    else:
        sy_series = np.asarray(sy, dtype=float)

    events_idx = detect_events(rain_mm, head_m,
                               rain_threshold_mm=rain_threshold_mm,
                               min_event_gap_days=min_event_gap_days,
                               peak_search_days=peak_search_days)

    events: list[RechargeEvent] = []
    for start, peak in events_idx:
        pre_h = float(head_m[start])
        peak_h = float(head_m[peak])
        extrap = extrapolate_recession(head_m, start, peak,
                                       lookback_days=lookback_days)
        if extrap is None:
            continue
        delta_h = peak_h - extrap
        rainfall = float(np.nansum(rain_mm[start:peak + 1]))
        sy_used = float(sy_series[peak])
        rch_mm = max(0.0, delta_h) * sy_used * 1000.0  # m → mm
        events.append(RechargeEvent(
            start_day=start, peak_day=peak,
            pre_event_h=pre_h, peak_h=peak_h,
            extrapolated_h=float(extrap),
            delta_h=float(delta_h),
            rainfall_mm=rainfall,
            sy_used=sy_used,
            recharge_mm=rch_mm,
        ))

    total_rch_mm = float(sum(e.recharge_mm for e in events))
    years = len(head_m) / 365.25
    annual_rch_mm = total_rch_mm / years if years > 0 else 0.0

    return annual_rch_mm, events


__all__ = [
    "RechargeEvent",
    "detect_events",
    "fit_antecedent_recession",
    "extrapolate_recession",
    "event_based_recharge",
]
