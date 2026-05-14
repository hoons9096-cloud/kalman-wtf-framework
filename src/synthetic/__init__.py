"""Synthetic-well benchmark generator for the Kalman-WTF framework."""

from .scenarios import (
    ScenarioConfig,
    SCENARIOS,
    SyntheticWell,
    generate_synthetic_well,
)
from .rainfall_generator import generate_rainfall
from .vadose_lag_filter import apply_lag_filter
from .dynamic_sy import generate_dynamic_sy, DEFAULT_VG_PARAMS
from .aquifer_state import simulate_aquifer
from .pumping_disturbance import inject_pumping, PumpingEvent, PumpingTruth
from .observation import apply_observation_process

__all__ = [
    "ScenarioConfig",
    "SCENARIOS",
    "SyntheticWell",
    "generate_synthetic_well",
    "generate_rainfall",
    "apply_lag_filter",
    "generate_dynamic_sy",
    "DEFAULT_VG_PARAMS",
    "simulate_aquifer",
    "inject_pumping",
    "PumpingEvent",
    "PumpingTruth",
    "apply_observation_process",
]
