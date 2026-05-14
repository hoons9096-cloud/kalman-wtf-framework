# kalman-wtf-framework

A phase-aligned and disturbance-robust **Kalman-filter–based framework** for
groundwater recharge estimation from noisy water-table records.

This repository accompanies the manuscript

> **Choi, J. (2026).** *A phase-aligned and disturbance-robust framework for
> groundwater recharge estimation using Kalman filter–based data assimilation.*
> Submitted to *Hydrogeology Journal*.

## What this framework does

Conventional water-table-fluctuation (WTF) recharge estimators assume an
**instantaneous** groundwater response to rainfall, ignore observation noise,
and confound pumping-induced recovery with natural recharge. Under typical
Korean field conditions (heterogeneous wells, agricultural pumping, sensor
noise) these assumptions fail and the conventional method becomes unusable
on a large fraction of the monitoring network.

The framework released here addresses the failure mode directly:

1. **State-space reformulation + Kalman filter** — Groundwater head is
   modelled as a linear-reservoir state with explicit process and observation
   noise. The Kalman update separates physical dynamics from sensor noise and
   from anthropogenic disturbances.
2. **Cross-correlation lag identification** — Rainfall and head time series
   are phase-synchronised by an objectively identified lag time, restoring
   hydrological causality between forcing and response.
3. **Pumping-event isolation** — Statistical-threshold filtering on the
   first-difference series flags pumping-induced drawdowns; the Kalman
   prediction step reconstructs the natural recession that the pumping
   masked.
4. **Dynamic specific yield** — A van Genuchten–based vadose-zone module
   produces a time-varying Sy that responds to antecedent moisture state,
   replacing the conventional fixed-Sy assumption.

## Repository scope

The repository currently contains:

- A **synthetic-well benchmark generator** with five named scenarios (S1–S5),
  designed so that every quantity the framework attempts to recover (lag time,
  recharge series, pumping events, dynamic Sy) has a corresponding
  ground-truth field. This is the part used to *validate the framework's
  recovery accuracy on known-truth data* in the accompanying paper.
- A test suite (`tests/`) covering the synthetic generator's physical
  properties (rainfall seasonality, lag-filter first moment, base-level
  constraint, Sy bounds, etc.).
- An example notebook (`notebooks/`) walking through scenario generation
  and visualisation.

The framework itself (`src/framework/`) and the conventional-WTF baselines
(`src/baseline/`) are organised as placeholder modules at this commit and
will be populated alongside the manuscript revision; the synthetic benchmark
is the artefact that the framework will be evaluated against.

## Quick start

```bash
git clone https://github.com/hoons9096-cloud/kalman-wtf-framework.git
cd kalman-wtf-framework
pip install -r requirements.txt

# Generate a synthetic well under scenario S1
python -c "
import sys; sys.path.insert(0, 'src')
from synthetic import generate_synthetic_well
well = generate_synthetic_well(scenario_id='S1', seed=0)
print(f'Annual recharge (true): {well.annual_recharge_mm():.1f} mm/yr')
print(f'Pumping events: {len(well.pumping_truth.events)}')
"

# Run the test suite
python -m unittest discover tests -v
```

## Synthetic-well benchmark scenarios

Each scenario exercises one or more of the framework's claimed contributions
on a *known-truth* time series.

| Scenario | Description | Tests |
|---|---|---|
| **S1** | 14-day lag, 10 pumping events/yr, σ_obs = 0.02 m, dynamic Sy | Baseline (all four contributions) |
| **S2** | 30-day lag, otherwise S1 | Lag identification at longer delays |
| **S3** | 20 pumping events/yr, otherwise S1 | Pumping isolation under heavy disturbance |
| **S4** | σ_obs = 0.05 m, otherwise S1 | Kalman noise suppression under high noise |
| **S5** | Fixed Sy = 0.10, otherwise S1 | Ablation of the dynamic-Sy contribution |

All scenarios share a 5-year (1,825-day) duration, Korean monsoon rainfall
(annual ≈ 950 mm), a linear-reservoir aquifer with k = 0.005 / day and base
level h_base = 0 m, and a recharge fraction of 0.12.

A sample S1 realisation and its overview plot are included in `data/`.

## Project layout

```
kalman-wtf-framework/
├── src/
│   ├── synthetic/                  # synthetic-well benchmark generator
│   │   ├── rainfall_generator.py
│   │   ├── vadose_lag_filter.py
│   │   ├── dynamic_sy.py
│   │   ├── aquifer_state.py
│   │   ├── pumping_disturbance.py
│   │   ├── observation.py
│   │   └── scenarios.py            # high-level API
│   ├── framework/                  # phase-aligned Kalman-WTF framework
│   └── baseline/                   # conventional-WTF baselines for comparison
├── tests/
│   └── test_synthetic.py
├── notebooks/                      # walkthroughs
├── data/                           # sample synthetic-well realisations
├── requirements.txt
└── LICENSE
```

## Citation

```bibtex
@article{Choi2026KalmanWTF,
  author  = {Choi, Junghoon},
  title   = {A phase-aligned and disturbance-robust framework for
             groundwater recharge estimation using Kalman filter-based
             data assimilation},
  journal = {Hydrogeology Journal},
  year    = {2026},
  note    = {Submitted}
}
```

## Contact

**Junghoon Choi, Ph.D.**
Founder & CEO, GEOINNOVATION Co., Ltd.
Daegu, Republic of Korea
ORCID: [0009-0002-0509-8089](https://orcid.org/0009-0002-0509-8089)

GEOINNOVATION Co., Ltd. is a hydrogeology consulting firm specialising in
groundwater impact assessment, dam-effect analysis, drought-response
planning, and basin-scale recharge mapping. The framework provided here is
the per-well operational pre-processing step that produces the input to the
watershed-scale envelope framework presented in a companion submission
(*Hybrid Recharge*; Choi 2026, *J. Hydrol.*, submitted).

## License

MIT — see [LICENSE](LICENSE).
