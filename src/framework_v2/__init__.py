"""kalman-wtf-framework v2 — Phase A accuracy improvements.

v2 introduces three changes vs the v1 paper version:

  1. **CCF-fixed lag**: lag is computed once from the rainfall–head
     cross-correlation peak and held fixed during Nelder-Mead. The v1
     framework let the optimiser absorb the lag into (k, z), producing
     τ_rmse = 0 for every Siheung well and undermining the titular
     contribution.

  2. **Bayesian Sy prior**: a penalty term draws the optimised effective
     Sy toward published field-effective values (Healy & Cook 2002),
     preventing the head-fit RMSE objective from collapsing to the
     degenerate near-zero-recharge solutions that affect 10 of 12 USDA
     texture classes at SH-22.

  3. **Event-based WTF**: in addition to the continuous head-trajectory
     optimisation of v1, recharge is also computed by isolating
     individual rainfall–response events and extrapolating the
     antecedent recession to the post-event peak (Healy 2010, §5.2).
     Event-based and continuous estimates are reported side-by-side and
     should converge if the framework is internally consistent.

These changes are kept entirely separate from src/framework/ so the v1
paper results remain reproducible by running the original modules.
"""

__all__: list[str] = []
