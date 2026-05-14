# kalman-wtf-framework

A **lag- and bias-aware Kalman-filter framework** for groundwater
recharge inference from shallow water-table records.

This repository accompanies the manuscript

> **Choi, J. (2026).** *A lag- and bias-aware Kalman-filter framework for
> groundwater recharge inference from shallow water-table records.*
> Submitted to *Hydrogeology Journal*.

## What this framework does

Conventional water-table-fluctuation (WTF) recharge estimators assume
an **instantaneous** groundwater response to rainfall, ignore
observation noise, and confound pumping-induced recovery with natural
recharge. The framework released here addresses these failure modes
and quantifies the residual lab-to-field specific-yield bias against a
controlled synthetic-truth benchmark:

1. **State-space reformulation + Kalman filter** — Groundwater head is
   modelled as a linear-reservoir state with explicit process and
   observation noise. The Kalman update separates physical dynamics
   from sensor noise and anthropogenic disturbances.
2. **Cross-correlation lag identification** — Rainfall and head time
   series are phase-synchronised by an objectively identified lag time
   (τ_cc), reported separately from the head-fit RMSE optimum
   (τ_rmse) — see §5.1.1 of the manuscript.
3. **Pumping-event isolation** — Statistical-threshold filtering on
   the first-difference series flags pumping-induced drawdowns; the
   Kalman prediction step reconstructs the natural recession that the
   pumping masked.
4. **Dynamic specific yield** — A van Genuchten–based vadose-zone
   module produces a time-varying Sy responsive to antecedent
   moisture state, replacing the conventional fixed-Sy assumption.
5. **Synthetic-truth benchmark** — Five purpose-built scenarios
   (S1–S5) provide a controlled quantification of the framework's
   recharge recovery under known-truth forcing.

The intended contribution is positioned as a *bias-aware
interpretation framework for WTF*, not as a black-box recharge
calculator. See §1.3 and §5 of the manuscript for the positioning
discussion.

## Repository contents

```
kalman-wtf-framework/
├── src/
│   ├── framework/                  # Kalman + WTF core (Python port of MATLAB)
│   │   ├── van_genuchten.py        # vg.m — retention curve
│   │   ├── fillable_porosity.py    # filpor / filpor_tr (lru_cache memoised)
│   │   ├── pumping_detection.py    # statistical outlier filter
│   │   ├── kalman_wtf.py           # state-space Kalman + linear-reservoir
│   │   ├── optim.py                # Nelder–Mead lag/k/z optimisation
│   │   └── io.py                   # 3-column txt loader
│   └── synthetic/                  # synthetic-well benchmark generator
│       ├── rainfall_generator.py
│       ├── vadose_lag_filter.py
│       ├── dynamic_sy.py
│       ├── aquifer_state.py
│       ├── pumping_disturbance.py
│       ├── observation.py
│       ├── scenarios.py            # high-level API
│       └── export_for_matlab.py
├── tests/                          # 28 unit tests (synthetic + framework)
├── notebooks/                      # figure-generation scripts
│   ├── make_paper_figures.py       # Figs 1–5
│   ├── make_site_map.py            # Fig 6
│   ├── make_plausibility_plot.py   # Fig 7
│   └── run_ablation.py             # Table 6 ablation
├── requirements.txt
└── LICENSE  (MIT)
```

The MATLAB reference implementation (used for the field analyses in
§4) is archived separately at `~/Dropbox/GW/Hybrid_final/` (not
mirrored here pending peer review). The Python port reproduces the
MATLAB outputs to within numerical tolerance — see `tests/test_framework.py`.

## Quick start (reproduce the paper)

```bash
git clone https://github.com/hoons9096-cloud/kalman-wtf-framework.git
cd kalman-wtf-framework
pip install -r requirements.txt

# Run the test suite (28 tests, ~30 s)
python -m pytest tests/ -v

# Generate the 5 synthetic-well scenarios + their MATLAB-format inputs
python -c "
import sys; sys.path.insert(0, 'src')
from synthetic.export_for_matlab import export_all_scenarios
export_all_scenarios(out_dir='data/matlab_inputs', seed=0)
"

# Regenerate the publication figures (1–7)
python notebooks/make_paper_figures.py
python notebooks/make_site_map.py
python notebooks/make_plausibility_plot.py

# Run the SH-22 ablation (Table 6)
python notebooks/run_ablation.py
```

## Synthetic-well benchmark scenarios

Each scenario exercises one or more framework components on a
*known-truth* time series.

| Scenario | Description | Tests |
|---|---|---|
| **S1** | 14-day lag, 10 pumping events/yr, σ_obs = 0.02 m, dynamic Sy | Baseline (all four contributions) |
| **S2** | 30-day lag, otherwise S1 | Lag identification at longer delays |
| **S3** | 20 pumping events/yr, otherwise S1 | Pumping isolation under heavy disturbance |
| **S4** | σ_obs = 0.05 m, otherwise S1 | Kalman noise suppression under high noise |
| **S5** | Fixed Sy = 0.10, otherwise S1 | Ablation of the dynamic-Sy contribution |

All scenarios share a 5-year (1,825-day) duration, Korean monsoon
rainfall (annual ≈ 950 mm), a linear-reservoir aquifer with
k = −0.005 day⁻¹, and a true annual recharge of ~ 79 mm yr⁻¹.

## Field data

The five Siheung-si monitoring wells (SH-08, SH-11, SH-22, SH-23,
SH-28) analysed in §4 of the manuscript are provided by the Siheung-si
municipal government. The raw daily head + rainfall records used in
the paper are available on request pending the municipality's data-use
agreement; the loader (`src/framework/io.py`) reads the standard
3-column whitespace-separated text format used by the MATLAB GUI.

## Field-private artefacts

To keep the repository consistent with the manuscript's peer-review
policy, the following files are *generated* by the scripts above but
not committed:

- `paper/` (LaTeX/Markdown sources, DOCX, BibTeX)
- `notebooks/figures/*.png` (publication figures)
- `data/batch_results.csv`, `data/ablation_sh22.csv`,
  `data/matlab_inputs/SYN_S*.{txt,_truth.json}`

These are listed in `.gitignore` and are reproducible from the
released source code with the commands above.

## Companion submission

A complementary watershed-scale envelope framework — applied to the
Yeongcheon / Geumho catchments with chloride mass-balance references
— is presented in a separate submission:

> **Choi, J. (2026).** *Bias-aware WTF at the watershed envelope
> scale.* Submitted to *Journal of Hydrology*. Code:
> [hoons9096-cloud/hybrid-recharge](https://github.com/hoons9096-cloud/hybrid-recharge).

The two papers are positioned as complementary in scope (single-well
daily inference vs. watershed-annual envelope), with non-overlapping
datasets, methods, and validation strategies.

## Citation

```bibtex
@article{Choi2026KalmanWTF,
  author  = {Choi, Junghoon},
  title   = {A lag- and bias-aware Kalman-filter framework for groundwater
             recharge inference from shallow water-table records},
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
Email: hoons9096@gmail.com

GEOINNOVATION Co., Ltd. is a hydrogeology consulting firm specialising
in groundwater impact assessment, dam-effect analysis,
drought-response planning, and basin-scale recharge mapping.

## License

MIT — see [LICENSE](LICENSE).
