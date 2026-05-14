"""Time-varying specific yield via the van Genuchten retention curve.

Generates a synthetic ground-truth Sy(t) series that responds to a running
soil-moisture state. The vadose-zone moisture is updated by a simple bucket
balance (rainfall inflow, drainage to recharge, evapotranspiration loss),
and the *fillable porosity* — defined as the available pore volume that
would accept the next rainfall increment — is computed from the van Genuchten
moisture-retention curve evaluated at the current matric head.

The result is the *true* time-varying Sy that the framework's dynamic-Sy
module (also van Genuchten based) should approximately recover.

Reference
---------
van Genuchten, M.Th. (1980). A closed-form equation for predicting the
hydraulic conductivity of unsaturated soils. *Soil Sci. Soc. Am. J.* 44(5).
"""
from __future__ import annotations

import numpy as np


# Default van Genuchten parameters for a loam soil (Carsel & Parrish 1988)
DEFAULT_VG_PARAMS = {
    "theta_r": 0.078,    # residual water content
    "theta_s": 0.43,     # saturated water content
    "alpha":   3.6,      # m^-1 (inverse air-entry pressure)
    "n":       1.56,     # pore-size distribution
    "Ks":      0.25,     # m/day (saturated hydraulic conductivity)
}


def _van_genuchten_theta(h_m: np.ndarray, p: dict) -> np.ndarray:
    """van Genuchten moisture-retention curve: theta(h)."""
    theta_r, theta_s, alpha, n = p["theta_r"], p["theta_s"], p["alpha"], p["n"]
    m = 1.0 - 1.0 / n
    h_pos = np.maximum(np.asarray(h_m, dtype=float), 1e-9)
    return theta_r + (theta_s - theta_r) / (1.0 + (alpha * h_pos) ** n) ** m


def _van_genuchten_h_from_theta(theta: np.ndarray, p: dict) -> np.ndarray:
    """Invert van Genuchten: solve for h given theta. Vectorised, analytic."""
    theta_r, theta_s, alpha, n = p["theta_r"], p["theta_s"], p["alpha"], p["n"]
    m = 1.0 - 1.0 / n
    theta = np.clip(theta, theta_r + 1e-6, theta_s - 1e-6)
    Se = (theta - theta_r) / (theta_s - theta_r)
    # Inversion: |alpha h|^n = (Se^(-1/m) - 1)
    rhs = Se ** (-1.0 / m) - 1.0
    h = (np.maximum(rhs, 0.0)) ** (1.0 / n) / alpha
    return h


def generate_dynamic_sy(
    rainfall_mm: np.ndarray,
    recharge_mm: np.ndarray,
    eto_mm_per_day: float | np.ndarray = 3.0,
    vadose_depth_m: float = 2.0,
    vg_params: dict | None = None,
    initial_theta_frac: float = 0.6,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a time-varying specific yield Sy(t) and matching theta(t).

    Parameters
    ----------
    rainfall_mm : ndarray
        Daily rainfall input (mm/day).
    recharge_mm : ndarray
        Daily effective recharge that drains out of the vadose-zone
        bucket (mm/day). Same length as `rainfall_mm`.
    eto_mm_per_day : float or ndarray, default 3.0
        Reference evapotranspiration (mm/day). Scalar or daily series.
    vadose_depth_m : float, default 2.0
        Vadose-zone thickness (m). Controls bucket size.
    vg_params : dict or None
        van Genuchten parameters. Default is loam (Carsel & Parrish 1988).
    initial_theta_frac : float, default 0.6
        Initial moisture content as a fraction between theta_r and theta_s.

    Returns
    -------
    sy_t : ndarray of shape (n_days,)
        Daily time-varying specific yield (dimensionless).
    theta_t : ndarray of shape (n_days,)
        Daily volumetric water content of the vadose-zone bucket.
    """
    p = dict(DEFAULT_VG_PARAMS) if vg_params is None else dict(vg_params)
    n_days = len(rainfall_mm)

    if np.isscalar(eto_mm_per_day):
        eto_series = np.full(n_days, float(eto_mm_per_day))
    else:
        eto_series = np.asarray(eto_mm_per_day, dtype=float)

    # Bucket capacity in mm-water
    capacity_mm = (p["theta_s"] - p["theta_r"]) * vadose_depth_m * 1000.0

    theta_t = np.empty(n_days)
    sy_t = np.empty(n_days)

    # Initialise water content
    theta_now = p["theta_r"] + initial_theta_frac * (p["theta_s"] - p["theta_r"])

    for i in range(n_days):
        # Inflow: rainfall reaching vadose zone
        # Outflow: recharge to aquifer + evapotranspiration loss
        # (recharge is already the post-vadose value, so we just decrement
        #  the bucket by it for moisture accounting)
        delta_storage_mm = rainfall_mm[i] - recharge_mm[i] - eto_series[i]

        # Convert mm to delta-theta over the bucket depth
        # delta_theta = delta_mm / (vadose_depth_m * 1000)
        theta_now += delta_storage_mm / (vadose_depth_m * 1000.0)
        theta_now = np.clip(theta_now, p["theta_r"] + 1e-4, p["theta_s"] - 1e-4)

        theta_t[i] = theta_now

        # Sy = fillable porosity (theta_s - theta_now) — the volume that
        # would accept the next rainfall increment before saturation
        sy_t[i] = p["theta_s"] - theta_now

    return sy_t, theta_t


__all__ = ["generate_dynamic_sy", "DEFAULT_VG_PARAMS"]
