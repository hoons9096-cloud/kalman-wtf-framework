"""Soil-heterogeneous synthetic ensemble with a realistic texture–Sy coupling.

The baseline benchmark (`synthetic.scenarios`) applies a uniform
`sy_field_scale = 0.25` to every texture, so the *effective* specific yield
varies by only ~9 % across USDA classes — far below the Sy-prior
uncertainty. That makes a soil-texture-informed prior useless and is an
artefact of the generator, not of nature: real specific yields span roughly
an order of magnitude across textures (Johnson, 1967; Healy and Cook, 2002).

This module generates an ensemble of wells whose *true effective specific
yield is anchored to a literature texture table* (`SY_BY_TEXTURE`), while
retaining a realistic dynamic Sy *shape* from the van Genuchten retention
model. The aquifer head responds accordingly (high-Sy sands give small
rises; low-Sy clays give large rises), so a texture-keyed prior now has a
genuine, exploitable signal — exactly the quantity a soil survey supplies
independently of the hydrograph.

Each well records its texture, true annual recharge, and true effective
specific yield, enabling exact recovery scoring of lumped vs soil-weighted
vs EnKF estimators.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .rainfall_generator import generate_rainfall
from .vadose_lag_filter import apply_lag_filter
from .dynamic_sy import generate_dynamic_sy
from .aquifer_state import simulate_aquifer
from .pumping_disturbance import inject_pumping
from .observation import apply_observation_process
from framework.fillable_porosity import SOIL_DB


# Representative specific yield by USDA texture (after Johnson, 1967;
# Healy and Cook, 2002, Table 2). Keyed to the SOIL_DB index convention.
# These are the *true* effective yields the generator anchors to, and the
# same table (with uncertainty) serves as the soil-weighted prior.
SY_BY_TEXTURE: dict[int, float] = {
    1:  0.27,   # Sand
    3:  0.22,   # Loamy Sand
    2:  0.18,   # Sandy Loam
    12: 0.14,   # Loam
    4:  0.12,   # Silt Loam
    11: 0.10,   # Sandy Clay Loam
    5:  0.10,   # Silt
    10: 0.08,   # Clay Loam
    9:  0.07,   # Silty Clay Loam
    8:  0.06,   # Sandy Clay
    7:  0.05,   # Silty Clay
    6:  0.03,   # Clay
}

SY_TEXTURE_PRIOR_STD = 0.03   # uncertainty of the texture-keyed prior


@dataclass
class SoilWell:
    name: str
    sn: int
    texture: str
    rain_m: np.ndarray
    gw_m: np.ndarray             # observed head (noise/pumping/gaps)
    h_true_m: np.ndarray         # natural head (no noise)
    sy_true_series: np.ndarray
    annual_recharge_true_mm: float
    sy_eff_true: float           # u-weighted, recession-corrected effective Sy
    k_true: float
    annual_rain_mm: float


def _effective_true_sy(h_true: np.ndarray, recharge_mm: np.ndarray,
                       k: float, rain_m: np.ndarray,
                       h_base: float = 0.0, r_cutoff_m: float = 0.002,
                       rain_lag_window: int = 0) -> float:
    """True effective Sy = total recharge / recession-corrected positive
    head-input integral, evaluated on the *noise-free* true head with the
    *same lag-widened rain gate* the estimator uses (vadose-lagged recharge
    arrives after the rain), so that the true Sy combined with a clean-head
    U recovers unity by construction."""
    dh = np.diff(h_true)
    u = dh + k * (h_true[:-1] - h_base)
    rain = rain_m[:len(u)] > r_cutoff_m
    mask = rain.copy()
    for s in range(1, rain_lag_window + 1):
        mask[s:] |= rain[:-s]
    U = float(np.sum(np.where(mask, np.maximum(u, 0.0), 0.0)))
    R = float(np.nansum(recharge_mm)) / 1000.0
    return R / U if U > 0 else np.nan


def generate_soil_well(
    sn: int,
    seed: int = 0,
    n_days: int = 365 * 5,
    annual_rainfall_mm: float = 950.0,
    tau_lag_days: float = 14.0,
    recharge_fraction: float = 0.12,
    k_per_day: float = 0.005,
    sensor_noise_std_m: float = 0.02,
    n_pump_per_year: float = 10.0,
    shared_rain: np.ndarray | None = None,
) -> SoilWell:
    """Generate one well of USDA texture ``sn`` with effective Sy anchored to
    SY_BY_TEXTURE[sn]."""
    s = SOIL_DB[sn]
    sy_target = SY_BY_TEXTURE[sn]

    rain = shared_rain if shared_rain is not None else generate_rainfall(
        n_days=n_days, annual_total_mm=annual_rainfall_mm, seed=seed)
    recharge = apply_lag_filter(rainfall_mm=rain, tau_lag_days=tau_lag_days,
                                recharge_fraction=recharge_fraction,
                                field_capacity_mm=3.0)

    # dynamic Sy *shape* from this texture's van Genuchten retention,
    # then rescale so the time-mean equals the literature texture yield.
    vg = dict(theta_r=s["theta_r"], theta_s=s["theta_s"],
              alpha=s["alpha"], n=s["n"], Ks=s["Ks"])
    sy_shape, _ = generate_dynamic_sy(rainfall_mm=rain, recharge_mm=recharge,
                                      eto_mm_per_day=3.0, vadose_depth_m=2.0,
                                      vg_params=vg)
    sy_true = np.clip(sy_shape * (sy_target / sy_shape.mean()), 0.01, 0.6)

    h_true = simulate_aquifer(recharge_mm_per_day=recharge, sy_t=sy_true,
                              k_per_day=k_per_day, h_base_m=0.0,
                              h_initial_m=1.0, process_noise_std_m=0.001,
                              seed=seed + 1)
    h_pump, _ = inject_pumping(h_true_m=h_true,
                               n_events_per_year=n_pump_per_year,
                               duration_range=(2, 7),
                               drawdown_range_m=(0.2, 0.5),
                               recovery_tau_range_days=(3.0, 8.0),
                               seed=seed + 2)
    h_obs = apply_observation_process(h_signal_m=h_pump,
                                      sensor_noise_std_m=sensor_noise_std_m,
                                      outlier_prob=0.005, outlier_magnitude_m=0.5,
                                      gap_prob=0.02, seed=seed + 3)

    n_years = n_days / 365.25
    return SoilWell(
        name=f"{s['name'].replace(' ', '')}_{sn}", sn=sn, texture=s["name"],
        rain_m=rain, gw_m=h_obs, h_true_m=h_true, sy_true_series=sy_true,
        annual_recharge_true_mm=float(np.nansum(recharge) / 1000.0 * 1000.0 / n_years),
        sy_eff_true=_effective_true_sy(h_true, recharge, k_per_day, rain),
        k_true=k_per_day,
        annual_rain_mm=float(rain.sum()) * 1000.0 / n_years,
    )


def generate_ensemble(seed: int = 0, **kw) -> list[SoilWell]:
    """One well per USDA texture, sharing a common regional rainfall."""
    rain = generate_rainfall(n_days=kw.get("n_days", 365 * 5),
                             annual_total_mm=kw.get("annual_rainfall_mm", 950.0),
                             seed=seed)
    wells = []
    for i, sn in enumerate(SY_BY_TEXTURE):
        wells.append(generate_soil_well(sn=sn, seed=seed + 10 * i,
                                        shared_rain=rain, **kw))
    return wells


__all__ = ["SY_BY_TEXTURE", "SY_TEXTURE_PRIOR_STD", "SoilWell",
           "generate_soil_well", "generate_ensemble", "_effective_true_sy"]
