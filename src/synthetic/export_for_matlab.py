"""Export synthetic-well realisations to the MATLAB framework's input format.

The MATLAB Filter-WTF GUI (and the headless `batch_runner.m`) reads
3-column whitespace-separated text files of the form

    YYYY-MM-DD    gw_obs(m)    rainfall(mm/day)

This module writes a synthetic well realisation in that exact format so that
the same MATLAB framework used for the field SH08–SH28 wells can be applied
to known-truth synthetic data. The truth values (lag, recharge series,
pumping events, Sy(t)) are kept in a sibling `_truth.json` file for
downstream accuracy scoring.

Usage
-----
    from synthetic import generate_synthetic_well
    from synthetic.export_for_matlab import export

    well = generate_synthetic_well("S1", seed=0)
    export(well, out_dir="matlab_inputs/", basename="SYN_S1")
    # writes matlab_inputs/SYN_S1.txt + SYN_S1_truth.json
"""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path

import numpy as np


def export(
    well,
    out_dir: str | Path = ".",
    basename: str | None = None,
    start_date: date = date(2026, 1, 1),
) -> tuple[Path, Path]:
    """Export a SyntheticWell realisation to MATLAB-compatible txt + truth json.

    Parameters
    ----------
    well : synthetic.scenarios.SyntheticWell
        The synthetic-well realisation to export.
    out_dir : str or Path
        Output directory (will be created if necessary).
    basename : str or None
        File basename without extension. Defaults to ``SYN_<scenario_id>``.
    start_date : datetime.date
        Date stamp for day 0. Each subsequent day increments by one calendar
        day. This anchors the MATLAB input format which expects dates.

    Returns
    -------
    (txt_path, truth_json_path) : tuple of Path
        Paths of the two written files.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if basename is None:
        basename = f"SYN_{well.config.name}"

    n = well.config.n_days

    # Use observed head (with noise/outliers/gaps); NaN entries become 'NaN'
    # in the txt — the MATLAB loader handles missing values via readtable.
    txt_path = out_dir / f"{basename}.txt"
    with txt_path.open("w") as f:
        for i in range(n):
            d = start_date + timedelta(days=i)
            gw = well.h_observed_m[i]
            rain = well.rainfall_mm[i]
            if np.isnan(gw):
                gw_str = "NaN"
            else:
                gw_str = f"{gw:.4f}"
            f.write(f"{d.isoformat()}\t{gw_str} {rain:.2f}\n")

    # Truth file (everything the framework should recover)
    truth = {
        "scenario": well.config.name,
        "description": well.config.description,
        "seed": int(well.seed),
        "config": asdict(well.config),
        "start_date": start_date.isoformat(),
        "n_days": n,
        # True forcing
        "rainfall_mm": well.rainfall_mm.tolist(),
        # True targets
        "recharge_true_mm": well.recharge_mm.tolist(),
        "annual_recharge_true_mm": float(well.annual_recharge_mm()),
        "tau_lag_true_days": float(well.config.tau_lag_days),
        "sy_true": well.sy_true.tolist(),
        "h_true_natural_m": well.h_true_m.tolist(),
        "h_with_pumping_m": well.h_with_pumping_m.tolist(),
        # Pumping ground truth
        "pumping_events": [
            {
                "start_day": ev.start_day,
                "duration": ev.duration,
                "max_drawdown_m": ev.max_drawdown_m,
                "recovery_tau_days": ev.recovery_tau_days,
            }
            for ev in well.pumping_truth.events
        ],
        "pumping_mask": well.pumping_truth.mask.astype(int).tolist(),
    }

    truth_path = out_dir / f"{basename}_truth.json"
    with truth_path.open("w") as f:
        json.dump(truth, f, indent=2)

    return txt_path, truth_path


def export_all_scenarios(
    out_dir: str | Path = "matlab_inputs",
    seed: int = 0,
) -> list[tuple[Path, Path]]:
    """Generate + export all five named scenarios (S1–S5).

    Returns
    -------
    paths : list of (txt_path, truth_json_path) for each scenario
    """
    from .scenarios import SCENARIOS, generate_synthetic_well

    out: list[tuple[Path, Path]] = []
    for sid in SCENARIOS:
        well = generate_synthetic_well(scenario_id=sid, seed=seed)
        out.append(export(well, out_dir=out_dir))
    return out


__all__ = ["export", "export_all_scenarios"]
