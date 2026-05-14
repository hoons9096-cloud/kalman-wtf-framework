"""Five named benchmark scenarios for the Kalman–WTF framework.

Each scenario defines a configuration that exercises one or more of the
framework's four claimed contributions:

1. Kalman noise/disturbance separation
2. Cross-correlation lag identification
3. Pumping-event isolation
4. Dynamic specific yield (van Genuchten)

Scenarios
---------
S1 — Baseline.
    14-day vadose lag, 10 pumping events/year, σ_obs = 0.02 m, dynamic Sy.
    Standard test case; framework should recover all four targets cleanly.

S2 — Long lag.
    30-day vadose lag, otherwise as S1.
    Stress test for lag identification at longer delays.

S3 — High pumping density.
    20 pumping events/year (≈ doubled).
    Stress test for pumping isolation under heavy disturbance.

S4 — High observation noise.
    σ_obs = 0.05 m, otherwise as S1.
    Stress test for Kalman noise suppression.

S5 — Fixed Sy.
    Constant Sy = 0.10, van Genuchten modulation disabled.
    Isolates the contribution of dynamic Sy by removing its truth.

All scenarios share: 5-year (1825-day) duration, Korean monsoon rainfall
(annual 950 mm), linear-reservoir aquifer with k = 0.005 / day and base
level h_base = 0 m, recharge fraction 0.12.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np

from .rainfall_generator import generate_rainfall
from .vadose_lag_filter import apply_lag_filter
from .dynamic_sy import generate_dynamic_sy
from .aquifer_state import simulate_aquifer
from .pumping_disturbance import inject_pumping, PumpingTruth
from .observation import apply_observation_process


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ScenarioConfig:
    name: str
    description: str
    n_days: int = 365 * 5
    annual_rainfall_mm: float = 950.0
    tau_lag_days: float = 14.0
    recharge_fraction: float = 0.12
    field_capacity_mm: float = 3.0

    # Aquifer
    k_per_day: float = 0.005
    h_base_m: float = 0.0
    h_initial_m: float = 1.0
    process_noise_std_m: float = 0.001

    # Sy
    dynamic_sy: bool = True
    fixed_sy: float = 0.10
    vadose_depth_m: float = 2.0
    eto_mm_per_day: float = 3.0

    # Pumping
    n_pumping_events_per_year: float = 10.0
    pump_duration_range: tuple = (2, 7)
    pump_drawdown_range_m: tuple = (0.2, 0.5)
    pump_recovery_tau_range_days: tuple = (3.0, 8.0)

    # Observation
    sensor_noise_std_m: float = 0.02
    outlier_prob: float = 0.005
    outlier_magnitude_m: float = 0.5
    gap_prob: float = 0.02


SCENARIOS: dict[str, ScenarioConfig] = {
    "S1": ScenarioConfig(
        name="S1",
        description="Baseline (14-day lag, 10 events/yr, σ=0.02 m, dynamic Sy)",
    ),
    "S2": ScenarioConfig(
        name="S2",
        description="Long lag (30-day vadose delay)",
        tau_lag_days=30.0,
    ),
    "S3": ScenarioConfig(
        name="S3",
        description="High pumping density (20 events/yr)",
        n_pumping_events_per_year=20.0,
    ),
    "S4": ScenarioConfig(
        name="S4",
        description="High observation noise (σ=0.05 m)",
        sensor_noise_std_m=0.05,
    ),
    "S5": ScenarioConfig(
        name="S5",
        description="Fixed Sy (van Genuchten modulation disabled)",
        dynamic_sy=False,
    ),
}


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------

@dataclass
class SyntheticWell:
    """Output of a single synthetic-well realisation."""
    config: ScenarioConfig
    seed: int

    # Forcing
    rainfall_mm: np.ndarray
    recharge_mm: np.ndarray  # true effective recharge (post-lag, pre-aquifer)

    # State truth
    sy_true: np.ndarray
    theta_true: np.ndarray
    h_true_m: np.ndarray            # natural head (no pumping, no noise)
    h_with_pumping_m: np.ndarray    # head with pumping signal subtracted
    h_observed_m: np.ndarray        # observed head (with noise + outliers + gaps)

    # Disturbance truth
    pumping_truth: PumpingTruth

    def annual_recharge_mm(self) -> float:
        """Mean annual recharge in mm (true value)."""
        n_years = len(self.recharge_mm) / 365.0
        return float(self.recharge_mm.sum() / n_years)

    def to_dict(self) -> dict[str, Any]:
        """Serialise the realisation (numpy arrays as lists) for JSON dump."""
        out = {
            "config": asdict(self.config),
            "seed": int(self.seed),
            "rainfall_mm": self.rainfall_mm.tolist(),
            "recharge_mm": self.recharge_mm.tolist(),
            "sy_true": self.sy_true.tolist(),
            "theta_true": self.theta_true.tolist(),
            "h_true_m": self.h_true_m.tolist(),
            "h_with_pumping_m": self.h_with_pumping_m.tolist(),
            "h_observed_m": self.h_observed_m.tolist(),
            "pumping_events": [
                {
                    "start_day": ev.start_day,
                    "duration": ev.duration,
                    "max_drawdown_m": ev.max_drawdown_m,
                    "recovery_tau_days": ev.recovery_tau_days,
                }
                for ev in self.pumping_truth.events
            ],
        }
        return out


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def generate_synthetic_well(
    scenario_id: str = "S1",
    seed: int = 0,
    config_override: dict | None = None,
) -> SyntheticWell:
    """Generate a complete synthetic-well realisation under a named scenario.

    Parameters
    ----------
    scenario_id : str, default "S1"
        One of {"S1", "S2", "S3", "S4", "S5"}. See module docstring.
    seed : int, default 0
        Master random seed. Each sub-process uses a deterministic
        offset for reproducibility.
    config_override : dict or None
        Optional dictionary of `ScenarioConfig` field overrides. Useful
        for ablation studies or custom sensitivity sweeps.

    Returns
    -------
    well : SyntheticWell
        Container with rainfall, true recharge, true Sy, true head,
        head-with-pumping, observed head, and the pumping event record.
        Every value the framework will attempt to recover has a
        corresponding truth field in this object.
    """
    if scenario_id not in SCENARIOS:
        raise ValueError(
            f"unknown scenario_id={scenario_id!r}; "
            f"known: {sorted(SCENARIOS)}"
        )
    cfg = SCENARIOS[scenario_id]
    if config_override:
        cfg_dict = asdict(cfg)
        cfg_dict.update(config_override)
        cfg = ScenarioConfig(**cfg_dict)

    # 1. Rainfall
    rainfall = generate_rainfall(
        n_days=cfg.n_days,
        annual_total_mm=cfg.annual_rainfall_mm,
        seed=seed,
    )

    # 2. Vadose-zone lag → effective recharge (pre-aquifer)
    recharge = apply_lag_filter(
        rainfall_mm=rainfall,
        tau_lag_days=cfg.tau_lag_days,
        recharge_fraction=cfg.recharge_fraction,
        field_capacity_mm=cfg.field_capacity_mm,
    )

    # 3. Specific yield (dynamic or fixed)
    if cfg.dynamic_sy:
        sy_true, theta_true = generate_dynamic_sy(
            rainfall_mm=rainfall,
            recharge_mm=recharge,
            eto_mm_per_day=cfg.eto_mm_per_day,
            vadose_depth_m=cfg.vadose_depth_m,
        )
    else:
        sy_true = np.full(cfg.n_days, cfg.fixed_sy)
        theta_true = np.full(cfg.n_days, np.nan)

    # 4. Aquifer state — natural head (no pumping, no noise)
    h_true = simulate_aquifer(
        recharge_mm_per_day=recharge,
        sy_t=sy_true,
        k_per_day=cfg.k_per_day,
        h_base_m=cfg.h_base_m,
        h_initial_m=cfg.h_initial_m,
        process_noise_std_m=cfg.process_noise_std_m,
        seed=seed + 1,
    )

    # 5. Pumping disturbance
    h_with_pumping, pumping_truth = inject_pumping(
        h_true_m=h_true,
        n_events_per_year=cfg.n_pumping_events_per_year,
        duration_range=cfg.pump_duration_range,
        drawdown_range_m=cfg.pump_drawdown_range_m,
        recovery_tau_range_days=cfg.pump_recovery_tau_range_days,
        seed=seed + 2,
    )

    # 6. Observation process — noise, outliers, gaps
    h_observed = apply_observation_process(
        h_signal_m=h_with_pumping,
        sensor_noise_std_m=cfg.sensor_noise_std_m,
        outlier_prob=cfg.outlier_prob,
        outlier_magnitude_m=cfg.outlier_magnitude_m,
        gap_prob=cfg.gap_prob,
        seed=seed + 3,
    )

    return SyntheticWell(
        config=cfg,
        seed=seed,
        rainfall_mm=rainfall,
        recharge_mm=recharge,
        sy_true=sy_true,
        theta_true=theta_true,
        h_true_m=h_true,
        h_with_pumping_m=h_with_pumping,
        h_observed_m=h_observed,
        pumping_truth=pumping_truth,
    )


__all__ = [
    "ScenarioConfig",
    "SCENARIOS",
    "SyntheticWell",
    "generate_synthetic_well",
]
