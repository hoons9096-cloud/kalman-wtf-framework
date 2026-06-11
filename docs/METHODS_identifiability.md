# Recharge from water-table fluctuations: an identifiability analysis

*Working methods + results note for the next manuscript. Single source
of truth for the formulation, the honest numbers, and the figures.*

## 1. Problem

The water-table-fluctuation (WTF) method estimates recharge as
`R = Sy · Δh`. Head data constrain the head rises `Δh` but only weakly
constrain the specific yield `Sy`. We make this precise and show that
three quantities all called "specific yield" differ systematically, so
the recharge magnitude is **non-identifiable from head data alone**.

On the synthetic benchmark (scenario S1) the three coexisting Sy are:

| quantity | value | meaning |
|---|---|---|
| drainable porosity `θs−θr` | 0.29–0.43 | laboratory |
| near-water-table fillable porosity (van Genuchten, daily rise) | ~0.00–0.05 | capillary-fringe limited |
| operational / apparent Sy (`Σrch / Σ positive Δh`) | 0.12 | what WTF actually uses; recession-inflated |

The generator's mean *daily* Sy is 0.071, yet the apparent operational
Sy is 0.119 (1.69× larger) purely because recession erodes the positive
rises (1216 recession days vs 608 rise days in S1).

## 2. Formulation (free-Sy, recession-corrected)

We drop the van-Genuchten fillable-porosity engine (`filpor_tr`, which
returns its floor value on ~99.8 % of days for realistic head rises) and
write the linear-reservoir head dynamics

```
h(t+1) − h_base = (1 − k)(h(t) − h_base) + u(t),   u(t) = R(t)/Sy
```

Two identifiability facts:

1. **`k` is identifiable** from dry-period decline alone (Sy-independent).
   Because the daily recession step (`k·deficit`) is comparable to the
   observation noise for slow systems, we estimate `k` from **multi-day
   log-linear fits** of the declining tail of each sustained dry spell,
   not day-to-day differencing. Recovered `k` = 0.0047–0.0065 vs true
   0.005.

2. Given `k`, the **head-equivalent recharge input**
   `u(t) = Δh + k(h−h_base)` is recovered, and its positive sum
   `U = Σ max(u,0)` is identifiable. Recharge is then

   ```
   R_annual = Sy · U · 1000 / n_years      [mm/yr]
   ```

So recharge = (identifiable head-rise integral `U`) × (non-identifiable
`Sy`). **Head data fix the line `R = Sy·U`, not the point on it**
(Figure 1). This removes the old "right recharge from wrong Sy"
compensating-error artefact: recharge accuracy and Sy accuracy are now
consistent.

Code: `src/framework_v2/free_sy_inversion.py`.

## 3. Identifying the non-identifiable Sy (dual constraint)

Two *orthogonal, head-free* priors pin the position on the line
(Figure 2):

- **A — literature Sy prior:** `Sy ~ N(0.07, 0.03²)` (vertical band).
- **B — catchment water balance:** `R ~ N(c·P, (σc·P)²)`,
  recharge coefficient `c ~ 0.12 ± 0.05` (horizontal band).

Each maps to a Gaussian statement about Sy; their precision-weighted
combination gives `Sy_joint` and `R_joint = Sy_joint · U'`. The
`consistency_sigma = |Sy_A − Sy_B| / √(σ_A² + σ_B²)` gap is a built-in
cross-validation: small ⇒ the two independent constraints agree.

Code: `src/framework_v2/water_balance.py`.

## 4. Results (honest, single fixed pipeline)

One configuration (`smooth_window=7`, `Sy~0.07±0.03`, `c~0.12±0.05`),
untuned, applied to all five scenarios — no exclusions, no per-scenario
search (`notebooks/v2/honest_benchmark.py`):

| scenario | k_est | U (mm/yr) | Sy* (truth-consistent) | recov (Sy prior) | Sy_joint | recov (joint) | cons σ |
|---|---|---|---|---|---|---|---|
| S1 | 0.0048 | 958 | 0.083 | 0.85 | 0.083 | 1.01 | 0.8 |
| S2 | 0.0047 | 908 | 0.087 | 0.80 | 0.084 | 0.96 | 0.9 |
| S3 | 0.0059 | 1326 | 0.060 | 1.17 | 0.077 | 1.28 | 0.3 |
| S4 | 0.0065 | 1094 | 0.072 | 0.97 | 0.081 | 1.12 | 0.6 |
| S5 | 0.0053 | 781 | 0.101 | 0.69 | 0.085 | 0.84 | 1.1 |

- Sy-prior alone: **recovery 0.90 ± 0.18**
- dual-constraint: **recovery 1.04 ± 0.17**, all `consistency_sigma < 2σ`.

## 5. Honest limitations (these are findings, not defects)

- **Recharge ∝ Sy_prior exactly** (Figure 3A): the head cannot pin Sy.
  The estimate is only as good as the Sy / water-balance priors.
- **`U` is processing-sensitive** (Figure 3B): the noise-smoothing
  window shifts the truth-consistent Sy* by ~±30 %. WTF recharge thus
  carries two uncertainty sources — Sy prior (~±40 %) and `U` processing
  (~±30 %) — so a careful WTF recharge is uncertain to roughly a factor
  of two before any external constraint.
- The dual-constraint estimate works because the two priors **bracket**
  the truth; if both were biased the same way it would not help —
  `consistency_sigma` is the safeguard that flags this.

## 6. Thesis & target

> WTF recharge is non-identifiable from head data: the head fixes only
> the recharge–Sy line. We formalise the estimate as `R = Sy·U` with a
> data-driven, recession-corrected `U`, quantify the equifinality, and
> identify the recharge scale with two orthogonal head-free constraints
> (literature Sy prior + catchment water balance) whose mutual
> consistency cross-validates the result.

Target: a methods/identifiability paper for *Journal of Hydrology* /
*Hydrogeology Journal* / *Hydrological Processes*.

## 7. Reproduce

```
python -c "import sys;sys.path.insert(0,'src');\
from synthetic.export_for_matlab import export_all_scenarios as e;e('data/matlab_inputs')"
python notebooks/v2/honest_benchmark.py
pytest -q          # 40 tests
```

Figures: `notebooks/figures/fig1_identifiability_S1.png`,
`fig2_dual_constraint_S1.png`, `fig3_sensitivity.png` (regenerated by the
snippets in the project history; data CSVs stay local per `.gitignore`).
