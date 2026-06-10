"""L-curve selection of the Sy-prior strength λ (Hansen 1992).

Motivation: tuning λ against synthetic-truth recovery (as in the first
v2 iteration, λ = 10) re-introduces exactly the circularity that the
HJ Associate Editor criticised — the regularisation strength would be
calibrated on the quantity it is later claimed to recover. The
L-curve criterion selects λ from the observed data alone:

  For each candidate λ, run the optimisation and record
      x(λ) = head-fit RMSE          (data misfit)
      y(λ) = |Sy_op − μ| / σ        (prior misfit, in prior σ units)

  Plotted on log axes the points trace an "L"; the corner — the point
  of maximum curvature — balances the two misfits and is the standard
  objective choice in Tikhonov-style regularisation (Hansen 1992;
  Aster et al. 2018, §4.4).

The corner is located with the triangle method (Castellanos et al.
2002), which is robust for the short, monotone λ sweeps used here.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from .optim_v2 import OptimV2Result, run_optimization_v2
from .sy_prior import DEFAULT_SY_PRIOR_MEAN, DEFAULT_SY_PRIOR_STD

# Default λ sweep: log-spaced, spanning "data-dominated" to
# "prior-dominated" regimes.
DEFAULT_LAMBDA_GRID = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)


@dataclass
class LCurveResult:
    lambda_grid: tuple[float, ...]
    rmse: list[float] = field(default_factory=list)         # x(λ)
    prior_misfit: list[float] = field(default_factory=list)  # y(λ)
    results: list[OptimV2Result] = field(default_factory=list)
    corner_index: int = -1

    @property
    def lambda_corner(self) -> float:
        return self.lambda_grid[self.corner_index]

    @property
    def result_at_corner(self) -> OptimV2Result:
        return self.results[self.corner_index]


def _triangle_corner(x: np.ndarray, y: np.ndarray) -> int:
    """Locate the L-curve corner by maximum-area triangle (Castellanos
    et al. 2002) on log-transformed coordinates.

    Returns the index of the corner point. Endpoints are excluded.
    """
    lx = np.log10(np.maximum(x, 1e-12))
    ly = np.log10(np.maximum(y, 1e-12))
    n = len(lx)
    if n < 3:
        return n // 2

    # Normalise to [0, 1] so the two axes weigh equally
    lx = (lx - lx.min()) / max(lx.max() - lx.min(), 1e-12)
    ly = (ly - ly.min()) / max(ly.max() - ly.min(), 1e-12)

    best_i, best_area = 1, -1.0
    p0 = np.array([lx[0], ly[0]])
    p1 = np.array([lx[-1], ly[-1]])
    for i in range(1, n - 1):
        p = np.array([lx[i], ly[i]])
        # Twice the triangle area via the cross product
        area = abs((p1[0] - p0[0]) * (p[1] - p0[1])
                   - (p[0] - p0[0]) * (p1[1] - p0[1]))
        if area > best_area:
            best_area = area
            best_i = i
    return best_i


def lcurve_select_lambda(
    po_m: np.ndarray,
    ho_m: np.ndarray,
    sn: int,
    lambda_grid: tuple[float, ...] = DEFAULT_LAMBDA_GRID,
    prior_mean: float = DEFAULT_SY_PRIOR_MEAN,
    prior_std: float = DEFAULT_SY_PRIOR_STD,
    **optim_kwargs,
) -> LCurveResult:
    """Sweep λ, trace the L-curve, and return the corner selection.

    Parameters
    ----------
    po_m, ho_m : ndarray
        Rainfall (m) and observed head (m).
    sn : int
        USDA soil class for this sweep (run per-sn or at the v1-selected
        sn; the corner λ is only weakly sn-dependent in testing).
    lambda_grid : tuple of float
        Candidate prior strengths, ascending.
    **optim_kwargs
        Forwarded to `run_optimization_v2` (lag_step, nm_maxiter, ...).

    Returns
    -------
    LCurveResult with the per-λ trace and the corner index.
    """
    out = LCurveResult(lambda_grid=tuple(lambda_grid))

    for lam in lambda_grid:
        r = run_optimization_v2(
            po_m=po_m, ho_m=ho_m, sn=sn,
            use_sy_prior=True, prior_strength=lam,
            **optim_kwargs,
        )
        out.results.append(r)
        out.rmse.append(r.rmse_pure)
        out.prior_misfit.append(
            abs(r.sy_operational - prior_mean) / prior_std)

    out.corner_index = _triangle_corner(
        np.asarray(out.rmse), np.asarray(out.prior_misfit))
    return out


def select_lambda_discrepancy(
    lc: LCurveResult,
    rmse_tolerance: float = 0.05,
) -> int:
    """Discrepancy-style λ selection: the largest λ whose RMSE stays
    within (1 + rmse_tolerance) of the best achievable RMSE.

    More robust than the triangle corner on the two curve shapes seen
    in practice: (a) flat curves (prior agrees with data — λ barely
    matters, pick the regularised end) and (b) hinge curves (prior
    conflicts with data — RMSE rises sharply past a threshold λ; stop
    just before the hinge).

    Returns the index into lc.lambda_grid.
    """
    rmse = np.asarray(lc.rmse)
    rmse_min = float(rmse.min())
    ceiling = rmse_min * (1.0 + rmse_tolerance)
    ok = np.where(rmse <= ceiling)[0]
    return int(ok.max()) if len(ok) else int(np.argmin(rmse))


__all__ = ["LCurveResult", "lcurve_select_lambda", "DEFAULT_LAMBDA_GRID"]
