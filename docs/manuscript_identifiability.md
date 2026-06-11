# Recharge is the product of an identifiable head integral and a non-identifiable specific yield: pumping-robust estimation and water-balance–constrained identification for the water-table fluctuation method

**J. Choi¹**, and co-authors

¹ *(affiliation to be completed)*

Corresponding author: hoons9096@gmail.com

> **Manuscript status.** Working draft for submission to a hydrology/hydrogeology
> methods journal (target: *Journal of Hydrology*, *Hydrogeology Journal*, or
> *Water Resources Research*). All numerical results were produced by the
> open Python implementation accompanying this paper (`framework_v2`).
> Bibliographic details in the reference list follow standard author–year
> associations and **should be verified against a citation database before
> submission.**

---

## Abstract

The water-table fluctuation (WTF) method estimates diffuse groundwater
recharge as the product of a specific yield and an observed rise of the
water table, \(R = S_y\,\Delta h\). Despite six decades of use, the method
carries a structural weakness that is rarely stated explicitly: the head
record constrains the rise term but is almost wholly uninformative about
the specific yield, so the recharge magnitude is **non-identifiable from
head data alone**. We formalise this statement. Writing the unconfined
aquifer as a single linear-reservoir state-space system, we show that the
maximum-likelihood estimate of recharge factorises exactly as
\(R = S_y\,U\), where \(U\) — the cumulative, recession-corrected,
head-equivalent recharge input — is a structurally identifiable functional
of the head and rainfall records, whereas \(S_y\) enters the likelihood
only as a multiplicative scale on which the Fisher information vanishes
along the recharge-input direction. The familiar "operational" specific
yield \(S_{y}^{\mathrm{op}}=\sum R/\sum(\Delta h)^{+}\) is shown to be a
recession-biased apparent quantity, systematically larger than the
column-effective specific yield by a factor controlled by the ratio of
recession to recharge time, and distinct again from the near-water-table
fillable porosity, which is capillary-fringe limited and an order of
magnitude smaller at daily resolution. We resolve the three-way ambiguity
by (i) estimating the recession constant from a multi-day log-linear fit
of dry-spell tails — necessary because, for slow systems, the daily
recession decrement is dominated by observation noise; (ii) recovering
\(U\) robustly, by reconstructing the head along the recession baseline
through pumping excursions — which we show are the *dominant* bias in the
input integral, inflating it by ~2.5-fold when pumping recovery is
mis-read as recharge; and (iii)
identifying \(S_y\) by fusing two orthogonal, head-free constraints — a
field-effective specific-yield prior and a catchment water-balance recharge
prior — in a precision-weighted Bayesian update whose internal consistency
provides a built-in cross-validation. On a five-scenario synthetic
benchmark with known truth, a single fixed pipeline recovers annual
recharge with a mean ratio of 1.04 ± 0.17. On a separate
soil-heterogeneous ensemble spanning an order of magnitude in true
specific yield, we isolate the input estimator and show that the
recession-baseline reconstruction removes the dominant pumping bias,
cutting the recovery error of the head-input integral from
RMSE 2.14 (a moving-average estimator; mean recovery 2.8) to RMSE 0.30
(mean 1.16) — a sevenfold reduction — at the cost of a characterised
~15 % conservative low bias. Applied to five alluvial
monitoring wells in Siheung, Republic of Korea, the method yields
recharge ratios of 10.9–14.2 % of precipitation (mean 12.3 %), tightly
clustered within independent catchment water-balance bounds and mutually
consistent to within 1.1 σ across all wells — contracting the 7–27 %
spread produced by an unconstrained calibration. The contribution is not a
"more accurate" WTF estimator but an *honest* one: a method that separates
what the data can and cannot determine, attaches the irreducible
prior-dependence of the recharge scale to an explicit, falsifiable external
constraint, and reports the recharge as a posterior distribution rather
than a point value.

**Keywords:** groundwater recharge; water-table fluctuation; specific
yield; identifiability; equifinality; Bayesian inverse problem;
regularisation; recession analysis; water balance; Kalman filter.

---

## 1. Introduction

### 1.1 The water-table fluctuation method and its appeal

Diffuse recharge — the downward flux of water that crosses the water table
and replenishes an unconfined aquifer — is among the most consequential and
least observable fluxes in the terrestrial water cycle (Scanlon et al.,
2002; Healy, 2010). Of the many techniques devised to estimate it, the
water-table fluctuation (WTF) method occupies a privileged position because
of its disarming simplicity and its reliance on the single most widely
collected hydrogeological observation, the groundwater hydrograph (Healy
and Cook, 2002; Crosbie et al., 2005; Cuthbert, 2010). In its canonical
form the method equates the recharge accumulated during a rise event to the
product of the specific yield \(S_y\) and the magnitude of the water-table
rise measured above the antecedent recession,
\[
  R \;=\; S_y \, \Delta h .
  \tag{1}
\]
The method is non-invasive, inexpensive, integrates over a representative
aquifer volume, and — unlike unsaturated-zone or tracer approaches —
responds directly to the quantity of interest at the relevant control
surface. These virtues explain its persistence from Meinzer (1923) and
White (1932) through the modern syntheses of Healy and Cook (2002), Healy
(2010), and Cuthbert (2014), and its routine application across
physiographic settings (Sophocleous, 1991; Moon et al., 2004; Risser et
al., 2005; Cao et al., 2016; Obuobie et al., 2012).

### 1.2 The specific-yield problem

The simplicity of (1) is deceptive. The recharge estimate inherits the full
uncertainty of \(S_y\), and \(S_y\) is notoriously difficult to constrain
(Johnson, 1967; Healy and Cook, 2002; Sophocleous, 2002). Three distinct
difficulties compound:

1. **\(S_y\) is not a single number.** The term denotes, depending on
   author and context, the drainable porosity of laboratory columns, the
   field-effective yield inferred from pumping tests, the time- and
   depth-dependent fillable porosity of the vadose profile, or the apparent
   ratio \(\sum R / \sum(\Delta h)^{+}\) extracted operationally from a
   hydrograph. These quantities differ systematically (Childs, 1960;
   Nachabe, 2002; Crosbie et al., 2005; Acharya et al., 2012). The
   near-water-table fillable porosity in particular collapses toward zero
   as the capillary fringe is approached, because the pore space immediately
   above the water table is nearly saturated (Gillham, 1984; Nachabe, 2002;
   Sophocleous, 1985).

2. **\(S_y\) is scale- and time-dependent.** The yield realised over a rapid
   event differs from that realised over a slow seasonal rise because
   drainage and capillarity have different characteristic times (Nachabe,
   2002; Crosbie et al., 2005; Acharya et al., 2012; Cuthbert, 2010).

3. **\(S_y\) is, to first order, invisible to the hydrograph.** This is the
   point we develop formally below. Because (1) couples \(R\) and \(S_y\)
   only through their product, a head record that perfectly determines the
   rise term leaves the factorisation of \(R = S_y\,\Delta h\) entirely
   open. Any inference that "calibrates" \(S_y\) against the same head data
   used to compute \(\Delta h\) is therefore either uninformative or
   circular.

The third difficulty is an instance of the **equifinality** that pervades
hydrological inverse problems (Beven and Binley, 1992; Beven, 2006;
Carrera and Neuman, 1986). It is widely acknowledged in passing — Healy and
Cook (2002, §4) note that \(S_y\) is "the largest source of uncertainty" —
but it is rarely treated as the *structural* property it is, with the
consequence that the literature is replete with WTF recharge values whose
quoted precision exceeds what the data can support.

### 1.2.1 Prior algorithmic variants of the WTF method

The literature has produced a sequence of increasingly sophisticated WTF
implementations, each addressing one facet of the problem while leaving the
specific-yield ambiguity intact. The discrete-event ("storm") variant
isolates individual rises above an extrapolated antecedent recession
(Healy and Cook, 2002; Nimmo et al., 2015; Heppner and Nimmo, 2005),
thereby removing the recession contamination of the rise term but still
multiplying by an externally assumed \(S_y\). The master-recession-curve
approach (Heppner and Nimmo, 2005; Posavec et al., 2006) and the
continuous time-series approach of Crosbie et al. (2005) and Cuthbert
(2010) embed the rise extraction in a calibrated reservoir model, improving
the rise estimate and the recession description but, again, treating
\(S_y\) as a multiplicative constant supplied by the user. Coupled
soil-water-balance/WTF schemes (Sophocleous, 1991; Obuobie et al., 2012)
import \(S_y\) implicitly through a calibrated soil model. Across this
lineage the recurring pattern is clear: ingenuity is spent on the
*identifiable* rise term, while the *non-identifiable* yield is either fixed
by fiat or absorbed into a calibration that the head data cannot constrain.
Our analysis explains why this division of labour is not a coincidence but a
structural necessity, and it relocates the yield to the one place from which
it can be determined — an external mass balance.

### 1.3 Regularisation, priors, and the risk of circularity

A natural response to an ill-posed inverse problem is regularisation: to
augment the data-misfit objective with a penalty that expresses prior
knowledge (Tikhonov and Arsenin, 1977; Hansen, 1998; Aster et al., 2018).
In the WTF context this means penalising departures of the inferred \(S_y\)
from a field-plausible range (Healy and Cook, 2002; Moon et al., 2004). The
difficulty is calibrating the regularisation strength. If it is tuned to
maximise recovery of a known synthetic truth, the procedure re-imports the
circularity it was meant to avoid; if it is fixed arbitrarily, the result
is sensitive to an unjustified choice. The discrepancy principle and the
L-curve criterion (Morozov, 1966; Hansen, 1992; Hansen and O'Leary, 1993)
select the regularisation strength from the data alone, but — as we show —
even a data-selected penalty cannot manufacture information about \(S_y\)
that the head record does not contain. The honest conclusion is that the
recharge *scale* must come from outside the hydrograph.

### 1.4 Contributions

This paper makes the equifinality of the WTF method precise and turns it
into a constructive method. Specifically:

- **(C1) A factorisation theorem (Section 2.3–2.4).** We prove that, under
  the linear-reservoir state-space model, the recharge estimate
  factorises as \(R = S_y\,U\) with \(U\) structurally identifiable and
  \(S_y\) lying in the null space of the Fisher information restricted to
  the recharge-input direction. The head record fixes the *line*
  \(\{(S_y, R): R = S_y U\}\) but not the *point* on it.

- **(C2) A taxonomy of specific yields (Section 2.2, 2.5).** We quantify
  the systematic gaps between drainable porosity, near-water-table fillable
  porosity, the column-effective yield, and the operational/apparent yield,
  and show that the last is recession-inflated by a factor
  \(1 + \mathcal{O}(\tau_r/\tau_R)\).

- **(C3) A recession-corrected, noise-aware estimator of \(U\)
  (Section 2.5–2.6).** We derive the recession-corrected head increment and
  show that the recession constant must be estimated from a multi-day
  log-linear fit, because for slow systems the one-step recession decrement
  is below the observation-noise floor; the effective number of independent
  observations is reduced by serial correlation by more than an order of
  magnitude.

- **(C-pump) Pumping recovery is the dominant input bias, and a
  recession-baseline reconstruction removes it (Section 2.5.1, 4.2).**
  We show on a soil-heterogeneous benchmark that mis-reading pumping
  recovery as recharge inflates the head-input integral \(U\) by ~2.5×
  — the leading error, ahead of observation noise (~1.3×) — and that a
  reconstruction exploiting the recharge/pumping asymmetry of the
  episodic master recession (Nimmo et al., 2015) cuts the input-recovery
  RMSE sevenfold.

- **(C4) A dual-constraint identification of \(S_y\) (Section 2.7–2.9).**
  We fuse a field-effective \(S_y\) prior with a catchment water-balance
  recharge prior in a precision-weighted Bayesian update, propagate the
  resulting uncertainty multiplicatively into \(R\), and use the
  consistency between the two constraints as a posterior predictive check
  (Gelman et al., 2013).

- **(C5) Honest validation (Sections 4–5).** On a synthetic benchmark we
  report a *single fixed pipeline*, with no per-case tuning and no scenario
  exclusion, and we expose rather than hide the residual processing
  sensitivity of \(U\). On real wells we report recharge as a distribution
  with an explicit, falsifiable consistency statistic.

The remainder of the paper is organised as follows. Section 2 develops the
theory. Section 3 describes the synthetic benchmark, the field site, and
the Python implementation. Section 4 presents the identifiability
demonstration, synthetic recovery, sensitivity analysis, and field
application. Section 5 discusses implications and limitations. Section 6
concludes. Appendices A–D collect the longer derivations.

---

## 2. Theory

Throughout, time is discretised at a daily step \(\Delta t = 1\,\mathrm{d}\)
and indexed by \(t = 0,1,\dots,N-1\). Heads \(h_t\) are in metres above an
arbitrary datum; rainfall \(p_t\) is in metres of water per day. We write
\((\cdot)^{+} = \max(\cdot,0)\).

### 2.1 The linear-reservoir state-space model

We adopt the lumped linear-reservoir representation of an unconfined
aquifer cell that underlies both the classical recession theory of
Maillet (1905) and Dewandel et al. (2003) and the WTF practice of Healy and
Cook (2002). The water table above a hydraulic base level \(h_b\) (the local
discharge elevation) relaxes exponentially toward \(h_b\) at a constant
specific rate \(k>0\) (per day) and is forced by a recharge-driven head
input \(u_t\):
\[
  h_{t+1} - h_b \;=\; (1-k)\,(h_t - h_b) \;+\; u_t \;+\; w_t,
  \qquad
  y_t \;=\; h_t \;+\; v_t .
  \tag{2}
\]
Here \(u_t = R_t / S_y\) is the head-equivalent recharge input (metres of
head added by the day-\(t\) recharge flux \(R_t\), in metres of water,
divided by the specific yield), \(w_t \sim \mathcal N(0,\sigma_w^2)\) is the
process noise representing model error and unresolved fluxes, and
\(v_t \sim \mathcal N(0,\sigma_v^2)\) is the observation noise of the
logger. Equation (2) is the discrete-time analogue of the continuous
linear-reservoir equation
\(\dot h = u(t)/\Delta t - k\,(h - h_b)\); the substitution
\(1-k \leftrightarrow e^{-k}\) is immaterial for \(k \ll 1\) and we use the
former to match the forward generator of Section 3.1. The state-space form
(2) is exactly the process equation of the Kalman filter (Kalman, 1960;
Jazwinski, 1970) used in the parent framework, and it is the minimal model
that simultaneously supports recession analysis and WTF recharge inference.

Two structural facts about (2) drive everything that follows. First, on
**rain-free days** the recharge input vanishes, \(u_t = 0\), so the head
obeys a pure first-order recession. Second, the recharge input enters
**additively and linearly**, so the cumulative input is a linear functional
of the head trajectory once \(k\) is known.

### 2.2 Three specific yields

Before analysing identifiability we must disambiguate \(S_y\). Let
\(\theta(z)\) denote the volumetric water content of the vadose profile at
height \(z\) above the water table, and \(\theta_s\) the saturated content.
We distinguish:

- **Drainable porosity** \(S_y^{\mathrm{dr}} = \theta_s - \theta_r\), the
  laboratory end-member (Johnson, 1967). For the alluvial materials of
  Section 3 this is \(0.29\)–\(0.43\).

- **Near-water-table fillable porosity** \(S_y^{\mathrm{fp}}(\Delta h)\),
  the air-filled pore volume per unit rise actually available to a rise of
  magnitude \(\Delta h\):
  \[
    S_y^{\mathrm{fp}}(\Delta h) \;=\;
    \frac{1}{\Delta h}\int_0^{\Delta h}\!\big(\theta_s - \theta(z)\big)\,dz .
    \tag{3}
  \]
  Because \(\theta(z)\to\theta_s\) as \(z\to 0^{+}\) (the capillary fringe
  is saturated; Gillham, 1984), the integrand vanishes at the lower limit
  and \(S_y^{\mathrm{fp}}(\Delta h)\to 0\) as \(\Delta h\to 0\). For
  van Genuchten (1980) media with the parameters of Section 3 and daily
  rises of a few centimetres, (3) is \(\mathcal O(10^{-2})\) or smaller —
  an order of magnitude below the drainable porosity. This is the physical
  origin of the failure of fillable-porosity engines at daily resolution
  (Appendix D).

- **Operational (apparent) specific yield**
  \(S_y^{\mathrm{op}} = \sum_t R_t / \sum_t(\Delta h_t)^{+}\), the ratio
  that a practitioner extracts by dividing total inferred recharge by the
  sum of positive head increments. This is the quantity that the WTF
  identity (1) actually manipulates.

The three are **not interchangeable**. Appendix B shows, and Section 4.1
confirms numerically, that for a hydrograph with recession the operational
yield exceeds the column-effective yield by a factor
\[
  \frac{S_y^{\mathrm{op}}}{\bar S_y}
  \;=\; 1 + \mathcal{O}\!\left(\frac{\tau_r}{\tau_R}\right),
  \qquad \tau_r = 1/k,
  \tag{4}
\]
where \(\tau_R\) is the mean inter-event recharge time. Equation (4) is the
formal statement that *the apparent yield is recession-biased*: the more
the water table recedes between recharge pulses, the more the positive-rise
denominator is eroded and the larger the apparent yield appears. Conflating
\(S_y^{\mathrm{op}}\) with a soil property — as fillable-porosity
parameterisations implicitly do — therefore introduces an
\(\mathcal O(\tau_r/\tau_R)\) bias that no amount of head fitting can
remove.

### 2.3 Structural identifiability: the factorisation theorem

We now make precise the sense in which \(S_y\) is invisible to the
hydrograph. Collect the unknowns as \(\boldsymbol\vartheta = (k, h_b, S_y,
\{u_t\})\) and consider the noise-free version of (2), i.e. the
input–output map from \(\boldsymbol\vartheta\) to the head trajectory
\(\{h_t\}\).

> **Proposition 1 (Factorisation).** *Let \(\{h_t\}\) be generated by (2)
> with \(w_t \equiv 0\). The head trajectory determines \(k\), \(h_b\), and
> the head-input sequence \(\{u_t\}\) uniquely (given an identifying dry
> spell, Proposition 2), but determines the pair \((S_y, \{R_t\})\) only up
> to the one-parameter family*
> \[
>   \mathcal F \;=\;
>   \Big\{\, (S_y, \{R_t\}) \;:\; R_t = S_y\,u_t,\; S_y>0 \,\Big\}.
>   \tag{5}
> \]
> *In particular the cumulative recharge \(R = \sum_t R_t = S_y\,U\) with
> \(U = \sum_t u_t\) is determined only as a function of the free scale
> \(S_y\); the head trajectory is invariant along \(\mathcal F\).*

*Proof.* From (2) with \(w_t\equiv 0\), \(u_t = (h_{t+1}-h_b) -
(1-k)(h_t-h_b)\). Thus once \((k,h_b)\) are known, \(\{u_t\}\) is a
deterministic, linear functional of \(\{h_t\}\) and is uniquely determined.
The map \(\{u_t\}\mapsto\{h_t\}\) does not involve \(S_y\) at all: \(S_y\)
appears only in the *definition* \(u_t = R_t/S_y\) relating the head input
to the water flux. Hence any \((S_y,\{R_t\})\) with \(R_t = S_y u_t\)
reproduces \(\{u_t\}\) and therefore \(\{h_t\}\) identically. Conversely two
admissible pairs that reproduce the same \(\{h_t\}\) must share \(\{u_t\}\)
and hence satisfy \(R_t/S_y = R'_t/S'_y\); since the daily flux partition is
otherwise unconstrained, both lie in \(\mathcal F\). \(\;\square\)

Proposition 1 is the exact formal content of contribution (C1): **the head
fixes the line \(R = S_y U\), not the point on it.** Identifiability of
\((k,h_b,\{u_t\})\) is established next.

### 2.4 The likelihood, Fisher information, and the equifinality ridge

Re-introduce noise. Under (2) with Gaussian \(v_t\) and a flat or weak
process noise, the negative log-likelihood of the parameters given the head
record is, up to constants,
\[
  -\log\mathcal L(\boldsymbol\vartheta)
  \;=\;
  \frac{1}{2\sigma_v^{2}}\sum_{t}\big(h_t(\boldsymbol\vartheta) - y_t\big)^2
  \;+\; \text{const},
  \tag{6}
\]
where \(h_t(\boldsymbol\vartheta)\) is the forward solution of (2). Because,
by Proposition 1, \(h_t\) is invariant along the family \(\mathcal F\)
(equivalently, invariant to \(S_y\) at fixed \(\{u_t\}\)), the directional
derivative of (6) along the \(S_y\) direction vanishes:
\[
  \frac{\partial}{\partial S_y}\,h_t(\boldsymbol\vartheta)\Big|_{\{u_t\}\,\mathrm{fixed}} = 0
  \quad\Longrightarrow\quad
  \mathcal I_{S_yS_y}
  \;=\;
  \mathbb E\!\left[\Big(\tfrac{\partial \log\mathcal L}{\partial S_y}\Big)^2\right] = 0 ,
  \tag{7}
\]
i.e. the Fisher information for \(S_y\) along the recharge-input direction
is **exactly zero**. The Cramér–Rao bound (Rao, 1945; Cramér, 1946) is
therefore unbounded: no unbiased estimator of \(S_y\) from the head record
has finite variance. The likelihood surface possesses an exact flat ridge —
the equifinality of Beven (2006) in its sharpest, structurally exact form,
rather than the approximate, noise-induced flatness usually invoked. The
practical corollary is decisive: **a data-only objective, however cleverly
regularised by L-curve or discrepancy rules, cannot determine the recharge
scale.** Regularisation can only select *where on the ridge* a prior places
its mode; it cannot make the ridge informative.

This result reframes the entire enterprise. The question is not "how do we
fit \(S_y\)?" but "from what *external* information do we fix the scale, and
how honestly do we propagate its uncertainty?" Sections 2.7–2.9 answer it.

### 2.4.1 The full Fisher information matrix and its rank deficiency

Equation (7) treats the \(S_y\) direction in isolation. We now exhibit the
complete information geometry. Reparameterise the recharge inputs through
the scale as \(R_t = S_y u_t\) and collect the estimable quantities as
\(\boldsymbol\phi = (k, h_b, u_1,\dots,u_{N-1})^{\!\top}\) together with the
scale \(S_y\). The forward model (2) gives the sensitivity rows
\[
  \frac{\partial h_{t+1}}{\partial u_\tau} = \mathbb 1[\tau\le t]\,(1-k)^{\,t-\tau},
  \qquad
  \frac{\partial h_{t+1}}{\partial k}
   = -\!\sum_{\tau\le t}(t-\tau)(1-k)^{\,t-\tau-1}(h_\tau-h_b),
\]
\[
  \frac{\partial h_{t+1}}{\partial h_b} = 1-(1-k)\sum(\cdots),
  \qquad
  \frac{\partial h_{t+1}}{\partial S_y} \equiv 0 .
\]
The Gaussian Fisher information is
\(\mathcal I_{ij} = \sigma_v^{-2}\sum_t \partial_i h_t\,\partial_j h_t\).
Because the final column (and row) is identically zero, the full
information matrix has the block structure
\[
  \mathcal I(\,\boldsymbol\phi, S_y) =
  \begin{pmatrix} \mathcal I_{\phi\phi} & \mathbf 0 \\[2pt] \mathbf 0^{\!\top} & 0 \end{pmatrix},
  \qquad \operatorname{rank}\mathcal I = \dim\boldsymbol\phi ,
  \tag{7$'$}
\]
so \(\mathcal I\) is **singular with a one-dimensional null space spanned by
\(\partial/\partial S_y\)**. By the theory of locally identifiable
parameters (Rothenberg, 1971; Bellman and Åström, 1970), a parameter is
locally identifiable iff the information matrix is non-singular in its
direction; \(S_y\) fails this test exactly, while \(\boldsymbol\phi\) — and
in particular every \(u_t\) and the recession constant \(k\) — is locally
identifiable provided \(\mathcal I_{\phi\phi}\succ 0\), which holds whenever
the record contains at least one resolved recession (so that the \(k\)- and
\(h_b\)-columns are non-degenerate) and the rainfall forcing is not
collinear in time. Equation (7$'$) is the multi-parameter generalisation of
(7): no reparameterisation, prior-free penalty, or change of optimiser can
populate the null space, because rank deficiency of the information matrix
is invariant under smooth reparameterisation. This is the precise sense in
which the recharge scale is *structurally*, not merely *practically*,
non-identifiable (Walter and Pronzato, 1997; Raue et al., 2009).

### 2.4.2 Profile likelihood and the geometry of the ridge

The same conclusion is visible in the profile likelihood. Profiling out
\(\boldsymbol\phi\) at fixed \(S_y\),
\(\ell_{\mathrm p}(S_y)=\min_{\boldsymbol\phi}[-\log\mathcal L]\), one finds
\(\ell_{\mathrm p}(S_y)\equiv\ell_{\mathrm p}(S_y')\) for all admissible
scales, because the minimising \(\boldsymbol\phi\) simply rescales the
recovered inputs \(u_t\) to hold \(R_t=S_y u_t\) fixed at the head-matching
values. The profile likelihood is therefore *exactly flat* — a horizontal
ridge of infinite length in \(\log S_y\), bounded only by the positivity and
physical-range constraints. Confidence intervals built from the profile
likelihood (the standard remedy for non-quadratic likelihoods; Raue et al.,
2009) are consequently \((0,\infty)\): the data alone license any positive
yield. This is the information-geometric portrait of equifinality, and it
motivates the move to an informative, *external* prior in Section 2.7.

### 2.5 Recession-constrained recovery of the head input \(U\)

We first secure the identifiable half of the problem. By Proposition 1,
\(U\) is recoverable once \((k,h_b)\) are known via
\[
  u_t \;=\; (h_{t+1}-h_b) - (1-k)(h_t-h_b)
        \;=\; \Delta h_t + k\,(h_t - h_b),
  \qquad
  U \;=\; \sum_{t\in\mathcal R}\big(u_t\big)^{+},
  \tag{8}
\]
where \(\mathcal R = \{t : p_t > p_c\}\) restricts the sum to days with
non-negligible rainfall (cut-off \(p_c\)), so that recession-driven sampling
noise on dry days does not enter \(U\). The second term \(k(h_t-h_b)\) is
the **recession correction**: it adds back the head that drainage removed,
so that \(u_t\) measures the recharge-driven component of the rise above the
extrapolated recession, not the raw increment. This is the continuous,
per-step analogue of the antecedent-recession extrapolation advocated by
Healy and Cook (2002), Crosbie et al. (2005), and Nimmo et al. (2015), and
it is precisely the correction that removes the recession bias (4): \(U\)
defined by (8) is the recession-free recharge-input integral, so the
yield that reproduces the true recharge from \(U\) is the column-effective
\(\bar S_y\), not the inflated \(S_y^{\mathrm{op}}\) (Appendix B).

### 2.5.1 Pumping recovery as the dominant input bias, and its removal

The recession-corrected input (8) is unbiased on clean data, but real
hydrographs are contaminated by groundwater abstraction. A pumping
episode is a drawdown followed by a recovery back toward the natural
trajectory. The drawdown itself contributes no positive input, but the
**recovery is a genuine upward movement of the water table that the
estimator (8) counts as recharge**. Because abstraction is uncorrelated
with rainfall, a fraction of recoveries coincides with the rain-gated
window and enters \(U\); the bias is largest for high-\(S_y\) materials,
whose genuine recharge rises are small (\(R/S_y\)) and therefore most
easily swamped by a 0.2–0.5 m pumping recovery. We show in Section 4.2
that this is the *dominant* error in \(U\) — a ~2.5-fold inflation —
exceeding the observation-noise contribution (~1.3-fold). Separating
pumping from recharge is the classic open difficulty of WTF analysis
(Healy and Cook, 2002; Cuthbert, 2010, 2014).

We resolve it by exploiting a physical asymmetry made explicit by the
episodic-master-recession concept (Nimmo et al., 2015; Heppner and Nimmo,
2005): **genuine recharge raises the recession baseline permanently —
the head subsequently recedes from a higher level — whereas a pumping
episode returns to the pre-existing baseline, leaving the recession
envelope unchanged.** Tracking the deficit baseline \(b_t\) under the
recession dynamics,
\[
  b_{t+1} =
  \begin{cases}
    x_{t+1}, & x_{t+1} \ge (1-k)\,b_t \quad(\text{recharge: baseline rises})\\[2pt]
    (1-k)\,b_t, & x_{t+1} < (1-k)\,b_t \quad(\text{drawdown: hold recession})
  \end{cases}
  \tag{8$'$}
\]
where \(x_t = h_t - h_b\), and forming the reconstructed deficit
\(\hat x_t = b_t\), any below-baseline excursion — the pumping drawdown
*and its recovery* — is replaced by the continuing recession line, so its
spurious rise never enters \(U\). On recharge-only data \(\hat x_t = x_t\)
and the estimator is unchanged. The reconstruction is deliberately
conservative: a recharge pulse arriving *during* a drawdown is partially
absorbed into the reconstructed recession, producing a mild,
characterised low bias (~15 %, Section 4.2) — an honest trade for
eliminating the dominant pumping bias. This is the state-reconstruction
("EnKF-flavoured") leg of the method: it improves the *identifiable*
input \(U\), independently of the non-identifiable scale \(S_y\).

### 2.6 Identifiability of the recession constant and the effective sample size

It remains to estimate \(k\) (and \(h_b\)). On a maximal dry spell
\([t_0,t_1)\) with \(u_t = 0\), (2) integrates to the geometric recession
\[
  (h_{t}-h_b) \;=\; (h_{t_0}-h_b)\,(1-k)^{\,t-t_0} ,
  \tag{9}
\]
so \(\log(h_t-h_b)\) is linear in \(t\) with slope \(\log(1-k)\). Naïvely
one might estimate \(k\) from successive daily ratios; this fails. The
one-step recession decrement is \(\Delta h^{\mathrm{rec}} = k(h-h_b)\),
which for the slow systems typical of alluvial aquifers
(\(k\sim 5\times10^{-3}\,\mathrm d^{-1}\), \(h-h_b\sim\) a few metres) is of
order \(0.01\)–\(0.02\) m — *comparable to or below the logger noise*
\(\sigma_v\). The single-step recession signal is therefore buried in noise,
and daily-difference estimators of \(k\) are biased and high-variance. This
is a concrete instance of a general phenomenon: serially correlated
residuals inflate the apparent information content of a record. The
**effective number of independent observations** over a window of length
\(n\) with lag-1 residual autocorrelation \(\rho_1\) is
\[
  n_{\mathrm{eff}} \;=\; n\,\frac{1-\rho_1}{1+\rho_1}
  \tag{10}
\]
(Bartlett, 1946; Bayley and Hammersley, 1946; Thiébaux and Zwiers, 1984).
For WTF residuals with \(\rho_1\) of order \(0.95\), (10) reduces an
\(N\sim1800\)-day record to \(n_{\mathrm{eff}}\sim 50\) effective points — a
36-fold reduction. Two consequences follow. First, \(k\) must be estimated
over the **multi-day decline** of each dry-spell tail, where the cumulative
recession signal \(k\,n\,(h-h_b)\) rises above the noise floor; we therefore
regress \(\log(h_t-h_b)\) on \(t\) across each sustained dry spell's
monotonic tail and pool the per-spell slopes by a length-weighted median
(Section 3.3, Appendix C). Second, any likelihood-based weighting of head
residuals (as in a naïve MAP) must be scaled by \(n_{\mathrm{eff}}/n\) lest
the data term overwhelm any prior; this is one more reason the scale of
\(S_y\) cannot be wrung from the hydrograph (Section 2.4).

### 2.6.1 Asymptotic properties of the recession estimator

The multi-day estimator of Appendix C is an M-estimator and admits a
standard asymptotic analysis. On a tail of length \(m\), regressing
\(\log d_t\) on \(t\) yields the ordinary-least-squares slope estimator
\(\hat b = \sum_t (t-\bar t)\log d_t / \sum_t (t-\bar t)^2\) with
\(\hat k = 1-e^{\hat b}\). Writing the multiplicative observation model
\(d_t = d_t^{0}(1+\varepsilon_t)\) with
\(\varepsilon_t = v_t/(h_t-h_b)\) small, a first-order expansion gives
\(\log d_t \approx \log d_t^{0} + \varepsilon_t\), so
\[
  \operatorname{Var}(\hat b)
  \;\approx\;
  \frac{\bar\sigma_\varepsilon^{2}}{\sum_t (t-\bar t)^2}
  \;=\;
  \frac{12\,\bar\sigma_\varepsilon^{2}}{m(m^2-1)},
  \tag{17}
\]
the classical \(\mathcal O(m^{-3})\) variance of a regression slope. The
cubic gain in \(m\) is precisely why the multi-day fit defeats the
single-step estimator, whose effective \(m=1\) variance is unbounded
relative to (17). Pooling \(S\) independent tails by a length-weighted
median further reduces the variance by \(\mathcal O(S^{-1})\) and confers
robustness to the heavy-tailed inter-spell variability of field recessions
(Cuthbert, 2014; Dewandel et al., 2003); the median is preferred to the
mean because occasional spells are contaminated by unlogged abstraction or
bank storage, producing outlying \(\hat k\) that a mean would absorb. The
estimator is consistent as \(\min_s m_s\to\infty\) and \(S\to\infty\), and
its leading bias from the log-transform of noisy deficits is
\(\mathcal O(\bar\sigma_\varepsilon^2)\), negligible for the deficits
\(>0.05\) m retained by the tail filter. A subtle point: because the
deficit depends on \(h_b\), the estimators of \(k\) and \(h_b\) are coupled;
in practice \(h_b\) is well constrained by the requirement that the
log-recession be linear (a wrong \(h_b\) curves the semi-log recession), and
we fix \(h_b\) at the framework's physical lower bound, accepting the
\(\mathcal O(\partial k/\partial h_b)\) sensitivity quantified in
Section 4.3.

### 2.7 Bayesian formulation and the irreducible role of the prior

Given the identifiable \(U\), the recharge is \(R = S_y\,U\), and by (7) the
posterior of \(S_y\) equals its prior up to the (vanishing) data
information:
\[
  \pi(S_y \mid \text{head}) \;\propto\;
  \mathcal L(S_y\mid\text{head})\,\pi(S_y)
  \;=\; \text{const}\times\pi(S_y).
  \tag{11}
\]
Equation (11) is not a defect of a particular algorithm; it is the exact
Bayesian expression of Proposition 1. The recharge posterior is the
pushforward of the \(S_y\) prior through the linear map \(S_y\mapsto S_y U\):
\[
  R \mid \text{head} \;\sim\; U\cdot \pi(S_y),
  \qquad
  \mathbb E[R] = U\,\mu_{S_y},
  \quad
  \mathrm{sd}(R) = U\,\sigma_{S_y}.
  \tag{12}
\]
A field-effective prior \(S_y\sim\mathcal N(\mu_{S_y},\sigma_{S_y}^2)\) with
\(\mu_{S_y}=0.07,\ \sigma_{S_y}=0.03\) (Healy and Cook, 2002; Moon et al.,
2004) thus yields a recharge with a \(\sim\pm 43\%\) coefficient of
variation *from the yield alone*. This is the honest uncertainty that the
conventional method conceals.

### 2.8 A second, orthogonal constraint: the catchment water balance

To narrow (12) we require information about the recharge *scale* that does
not pass through the hydrograph. The long-term catchment water balance
provides exactly such a constraint. Over a period long compared with the
basin response time, conservation of mass gives
\[
  R \;=\; P - \mathrm{ET}_a - Q - \Delta S \;\approx\; c\,P,
  \tag{13}
\]
where \(P\) is precipitation, \(\mathrm{ET}_a\) actual evapotranspiration,
\(Q\) runoff, \(\Delta S\) storage change, and \(c\) the recharge
coefficient (Scanlon et al., 2002; Healy, 2010). For shallow Korean
alluvial aquifers \(c\) lies in the range 0.08–0.20 (Moon et al., 2004; Kim
et al., 2014); we encode this as \(c\sim\mathcal N(0.12, 0.05^2)\),
deliberately wide. Crucially, (13) is **independent of \(S_y\) and of the
hydrograph**: it constrains \(R\) directly. Mapped onto the line
\(R = S_y U'\) (with \(U' = U\cdot10^3/n_{\mathrm{yr}}\) the annualised
input in mm yr⁻¹), the water balance becomes a second Gaussian statement
about \(S_y\),
\[
  S_y^{\mathrm{wb}} \;=\; \frac{cP}{U'},
  \qquad
  \sigma_{S_y}^{\mathrm{wb}} \;=\; \frac{\sigma_c P}{U'} .
  \tag{14}
\]

### 2.9 Precision-weighted fusion and the consistency check

The literature prior (constraint A) and the water-balance prior
(constraint B) are two independent Gaussian statements about the same
scalar \(S_y\). Their Bayesian fusion is the precision-weighted product
(DeGroot, 1970; Gelman et al., 2013):
\[
  S_y^{\star} \;=\;
  \frac{\mu_{S_y}/\sigma_{S_y}^2 + S_y^{\mathrm{wb}}/(\sigma_{S_y}^{\mathrm{wb}})^2}
       {1/\sigma_{S_y}^2 + 1/(\sigma_{S_y}^{\mathrm{wb}})^2},
  \qquad
  \sigma_{S_y}^{\star} \;=\;
  \Big(1/\sigma_{S_y}^2 + 1/(\sigma_{S_y}^{\mathrm{wb}})^2\Big)^{-1/2},
  \tag{15}
\]
and the recharge estimate is \(R^{\star} = S_y^{\star}U'\) with band
\(\sigma_{R}^{\star} = \sigma_{S_y}^{\star}U'\). The two constraints are not
merely combined; their **agreement is testable**. Define the consistency
statistic
\[
  \zeta \;=\;
  \frac{\big|\mu_{S_y} - S_y^{\mathrm{wb}}\big|}
       {\sqrt{\sigma_{S_y}^2 + (\sigma_{S_y}^{\mathrm{wb}})^2}} .
  \tag{16}
\]
Under the null hypothesis that both constraints and the identifiable \(U\)
are correct, \(\zeta\) is an \(\mathcal O(1)\) standard normal deviate; a
large \(\zeta\) falsifies one of the inputs (wrong soil class, rejected
recharge, mis-estimated \(U\), or a non-WTF well). Equation (16) is a
posterior predictive check in the sense of Gelman et al. (2013) and Box
(1980), and it is the mechanism by which the method, despite resting on a
prior-determined scale, remains *falsifiable* rather than merely assumed.

### 2.9.1 Full uncertainty propagation

The recharge estimate inherits uncertainty from three sources: the fused
yield \(S_y^\star\) (variance \((\sigma_{S_y}^\star)^2\)), the identifiable
input \(U'\) (variance \(\sigma_{U'}^2\), dominated by the noise-processing
choice of Section 4.3 and the recession-constant error of Section 2.6.1),
and the water-balance forcing \(P\) (variance \(\sigma_P^2\)). Treating
\(R^\star = S_y^\star U'\) and propagating to first order with the
independence of \(S_y^\star\) and \(U'\) approximately holding,
\[
  \left(\frac{\sigma_{R}^\star}{R^\star}\right)^2
  \;\approx\;
  \left(\frac{\sigma_{S_y}^\star}{S_y^\star}\right)^2
  +\left(\frac{\sigma_{U'}}{U'}\right)^2 ,
  \tag{18}
\]
a quadrature sum of relative uncertainties. Because the water-balance
constraint enters \(S_y^\star\) through \(S_y^{\mathrm{wb}}=cP/U'\), a
mis-specified \(P\) propagates into both factors and is partially
self-correcting: an overestimated \(P\) inflates \(S_y^{\mathrm{wb}}\) and
hence \(S_y^\star\), but the same \(P\) does not enter the literature
constraint, so the fusion damps the error in proportion to the relative
precisions. The reported band \(\sigma_R^\star = \sigma_{S_y}^\star U'\) of
Section 2.9 captures the yield term of (18); the input term is reported
separately as the processing sensitivity of Section 4.3, deliberately not
folded into a single number, so that the reducible (better data/processing)
and irreducible (prior-limited) components remain distinguishable to the
reader. This separation is, we argue, more useful to a decision-maker than a
single aggregate interval that obscures which uncertainty further
measurement could reduce.

### 2.9.2 Observability via the input–output transfer function

A complementary, control-theoretic view confirms the identifiability
partition. Taking the \(z\)-transform of (2) with \(w_t\equiv0\), the
transfer function from input \(u\) to head \(h\) is
\(H(z) = z^{-1}\big/\big(1-(1-k)z^{-1}\big)\), a first-order all-pole
filter. The pole \(z = 1-k\) is fixed by the recession and is identifiable
from the autoregressive structure of the dry-period head alone; the static
gain \(H(1) = 1/k\) maps cumulative input to cumulative head rise. The map
\(u\mapsto h\) is thus a known, invertible linear operator once the pole is
fixed, so \(\{u_t\}\) — and \(U\) — are recoverable by deconvolution
(Section 2.5). The scale \(S_y\) is *not a parameter of \(H\)*: it lives
entirely in the definition \(u=R/S_y\), outside the input–output map. The
transfer-function viewpoint therefore reproduces Proposition 1: the dynamics
(pole, gain, input) are observable; the flux/scale partition is not.

### 2.10 Summary of the estimator

The complete estimator is: (i) estimate \(h_b\) and, by multi-day
log-linear recession fitting, \(k\) (Section 2.6); (ii) form the
recession-corrected input \(U\) from a noise-debiased head increment
(Section 2.5); (iii) constrain \(S_y\) by fusing the field prior with the
water-balance prior (Section 2.9); (iv) report \(R^{\star}\) as a posterior
mean with a multiplicatively propagated band and the consistency statistic
\(\zeta\). No step uses the broken near-water-table fillable-porosity engine
(Appendix D); the recharge scale is attached to an explicit external
constraint rather than calibrated against the data it predicts.

---

## 3. Materials and methods

### 3.1 Synthetic benchmark with known truth

To evaluate an estimator of an unobservable flux one needs a controlled
setting in which the truth is known. We generate five synthetic wells
(S1–S5) by forward simulation of (2) with a prescribed, time-varying true
yield and a known recharge series, following the design of Crosbie et al.
(2005) and the benchmark philosophy of Delin et al. (2007). A 5-year
(1825-day) daily rainfall series is generated with a Korean monsoon
seasonality (annual 950 mm). Effective recharge is produced by a
vadose-zone lag filter (mean lag 14 d in S1; 30 d in S2) acting on
rainfall with a 0.12 recharge fraction. A time-varying true specific yield
is generated from a van Genuchten (1980) retention model. The aquifer head
is propagated by (2) with \(k = 0.005\,\mathrm d^{-1}\) and base level
\(h_b = 0\); pumping disturbances and observation noise (σ = 0.02 m in S1;
0.05 m in S4), outliers, and gaps are superimposed. The five scenarios
stress, respectively, the baseline (S1), long lag (S2), heavy pumping (S3),
high noise (S4), and a fixed-\(S_y\) control (S5). For each scenario the
generator records the true annual recharge and the true operational yield,
permitting exact recovery scoring. *Importantly, the generator and the
estimator use different specific-yield parameterisations*, so the benchmark
is not an inverse crime (Colton and Kress, 1992; Wirgin, 2004): a perfect
estimator must recover the truth despite a structural model mismatch.

To evaluate the input estimator under soil heterogeneity and pumping we
additionally generate a **soil-heterogeneous ensemble**: one well per
USDA texture, each with its *true effective specific yield anchored to a
literature texture table* (after Johnson, 1967; spanning 0.03 for clay to
0.27 for sand), retaining a realistic dynamic-\(S_y\) shape from the van
Genuchten retention model and forced by a common regional rainfall. The
head responds accordingly — high-\(S_y\) sands give small rises,
low-\(S_y\) clays large rises — so the true effective yield varies by an
order of magnitude (coefficient of variation 54 %, versus only 9 % when a
uniform field scaling is imposed, an artefact of the baseline generator).
Pumping (10 events yr⁻¹, 0.2–0.5 m drawdown, 3–8 d recovery) and
observation noise are independently switchable, isolating the input-bias
contributions (clean / noise-only / pumping-only / full regimes).

### 3.2 Field site and data

We apply the method to five alluvial groundwater monitoring wells (SH08,
SH11, SH22, SH23, SH28) in Siheung-si, Gyeonggi Province, Republic of
Korea, with co-located daily rainfall from a single regional gauge. Records
span roughly one year (313–345 days) in 2015–2016. Heads range from shallow
(0.2–2.5 m, SH08) to moderately deep (1.5–3.6 m, SH22). The wells tap an
unconfined Quaternary alluvial aquifer typical of Korean coastal lowlands
(Moon et al., 2004). Annual precipitation over the record is ≈ 940 mm.

### 3.3 Implementation

The estimator is implemented in Python 3.11 (NumPy, SciPy) as the module
`framework_v2` accompanying this paper. The recession constant is estimated
by `estimate_recession_k`: sustained dry spells (rainfall ≤ 2 mm for ≥ 8
consecutive days) are identified; within each, the monotonic tail after the
within-spell head peak is taken; \(\log(h-h_b)\) is regressed linearly on
time; per-spell decay rates are pooled by a length-weighted median.
Observation noise is suppressed prior to forming \(U\) by a 7-day centred
moving average, chosen to halve the noise standard deviation while
preserving the multi-day recharge rises (Section 4.3 quantifies the
sensitivity to this choice). The water-balance fusion is implemented in
`water_balance.constrain_recharge`. The full pipeline, a synthetic-data
generator, a one-shot runner, and a unit-test suite (40 tests) are released
openly; all results below are reproduced by `notebooks/v2/honest_benchmark.py`
and `notebooks/v2/field_identifiability.py`.

### 3.4 Evaluation

For synthetic wells we report the **recovery ratio** \(R^{\star}/R_{\rm
true}\) and the truth-consistent yield \(S_y^{\star\star} = R_{\rm
true}\,n_{\rm yr}/(U'\cdot10^3)\). We deliberately report a **single fixed
configuration** applied identically to all scenarios (no per-scenario
tuning, no exclusions), in contrast to best-of envelopes that flatter a
method. For field wells, where truth is unavailable, we report the recharge
posterior (mean and band), the recharge ratio \(R^{\star}/P\), and the
consistency statistic \(\zeta\), and we benchmark plausibility against
independent catchment water-balance bounds and the regional literature
range (Moon et al., 2004).

---

## 4. Results

### 4.1 The equifinality is exact and the three yields differ as predicted

Figure 1 displays, for scenario S1, the recharge–\(S_y\) line
\(R = S_y U'\) recovered from the head record. Every point on the line
reproduces the head trajectory equally well; the head data fix the slope
\(U' = 958\) mm yr⁻¹ but place no constraint on the position along the
line — the exact equifinality of Proposition 1. Direct evaluation of the
three yields of Section 2.2 on the S1 generator confirms the predicted
separation: the mean daily column-effective yield is 0.071; the operational
yield \(S_y^{\mathrm{op}} = 0.119\) exceeds it by a factor 1.69, consistent
with the recession-bias prediction (4) given the 1216 recession days versus
608 rise days; and the near-water-table fillable porosity evaluated at the
daily rises is ≤ 0.05 and collapses to its floor on 98–99.8 % of days
(Appendix D). The operational yield is therefore neither a soil property nor
the column-effective yield, exactly as Section 2.2 argues.

### 4.2 Synthetic recovery under a single fixed pipeline

Table 1 reports recovery for the single fixed pipeline. The recession
constant is recovered to within 30 % across all scenarios (estimated
0.0047–0.0065 d⁻¹ against a true 0.005), validating the multi-day estimator
of Section 2.6. With the literature prior alone the recovery ratio is
0.90 ± 0.18; adding the water-balance constraint (the dual-constraint
estimator) yields 1.04 ± 0.17, with all consistency statistics below 1.1 σ,
indicating that the two independent constraints mutually agree on every
scenario.

**Table 1.** Single-pipeline recovery on the synthetic benchmark
(`smooth_window = 7`, \(S_y\sim\mathcal N(0.07,0.03^2)\),
\(c\sim\mathcal N(0.12,0.05^2)\)). \(S_y^{\star\star}\) is the
truth-consistent yield; recov\(_A\) uses the literature prior alone;
recov\(_J\) the dual constraint; \(\zeta\) the consistency statistic.

| scenario | \(k\) (d⁻¹) | \(U'\) (mm yr⁻¹) | \(S_y^{\star\star}\) | recov\(_A\) | recov\(_J\) | \(\zeta\) (σ) |
|---|---|---|---|---|---|---|
| S1 | 0.0048 | 958 | 0.083 | 0.85 | 1.01 | 0.8 |
| S2 | 0.0047 | 908 | 0.087 | 0.80 | 0.96 | 0.9 |
| S3 | 0.0059 | 1326 | 0.060 | 1.17 | 1.28 | 0.3 |
| S4 | 0.0065 | 1094 | 0.072 | 0.97 | 1.12 | 0.6 |
| S5 | 0.0053 | 781 | 0.101 | 0.69 | 0.84 | 1.1 |
| **mean** | | | | **0.90 ± 0.18** | **1.04 ± 0.17** | ≤ 1.1 |

### 4.3 Sensitivity: what the data do and do not determine

Figure 3 separates the two uncertainty sources. Panel A confirms (12): the
recovery ratio is *exactly linear* in the assumed \(S_y\) prior mean, with
slope set by the identifiable \(U'\) — the recharge is only as accurate as
the yield prior, and no processing choice changes this. Panel B quantifies
the *identifiable* part's residual uncertainty: the truth-consistent yield
\(S_y^{\star\star}\) varies by roughly ±30 % as the noise-suppression window
is swept from 1 to 19 days, because smoothing trades noise-induced
inflation of \(\sum(u)^{+}\) against attenuation of genuine rises. WTF
recharge thus carries two distinct uncertainties — a prior-dominated scale
(~±43 %) and a processing-dependent input (~±30 %) — so a careful estimate
is uncertain to roughly a factor of two before any external constraint. We
regard the explicit exhibition of these two sources, rather than their
concealment behind a single point value, as a principal result.

### 4.3.1 Pumping-robust input estimation

We isolate the quality of the identifiable input \(U\) on the
soil-heterogeneous ensemble by scoring recovery with the *true* effective
yield (so the non-identifiable scale is removed from the comparison) under
the four controlled regimes. Table 3 contrasts a 7-day moving-average
estimator with the recession-baseline reconstruction of Section 2.5.1.

**Table 3.** Recovery (estimated/true annual recharge, true \(S_y\))
across regimes; mean and RMSE about unity over the texture ensemble.

| regime | moving-average (mean / RMSE) | reconstruction (mean / RMSE) |
|---|---|---|
| clean (no pump, no noise) | 1.02 / 0.14 | 0.81 / 0.22 |
| noise only | 1.33 / 0.49 | 1.03 / 0.21 |
| pumping only | 2.51 / 1.79 | 0.80 / 0.23 |
| **full** | **2.82 / 2.14** | **1.16 / 0.30** |

The decomposition is unambiguous: pumping recovery, not noise, dominates
the input bias (a 2.5-fold inflation under pumping alone, versus 1.3-fold
under noise alone), and the moving average cannot remove it because the
recovery is a real, smooth head rise (Figure 4). The recession-baseline
reconstruction removes it — the full-regime recovery falls from 2.82
(RMSE 2.14) to 1.16 (RMSE 0.30), a sevenfold error reduction (Figure 5) —
while incurring the characterised ~15–19 % conservative low bias visible
in the clean regime (0.81), where a fraction of genuine recharge arriving
during drawdowns is absorbed into the reconstructed recession. This is
the methodological core of the paper: the dominant, previously
conflated input bias is diagnosed and removed by a physically grounded
state reconstruction, independently of the specific-yield scale.

### 4.4 Field application: Siheung wells

Table 2 reports the five Siheung wells under the full pipeline
(pumping-robust input + dual-constraint identification). The recharge
ratios cluster tightly at 9.6–13.8 % of precipitation (mean 11.9 %),
within the independent catchment range of 8–25 %, with all consistency
statistics below 0.6 σ. This contracts the 7–27 % spread (including an
implausible 27.5 % at SH08) produced by an unconstrained
fillable-porosity calibration of the same records. The pumping-robust
input estimator both tightens the diagnostic and exposes abstraction:
relative to a moving-average input, it reduces well SH22's anomalous
input integral by 20 % (3040 → 2422 mm yr⁻¹), halving its consistency
statistic (1.0 → 0.6 σ), and it detects a 27 % pumping inflation at SH23
(1535 → 1130 mm yr⁻¹). That the catchment-mean recharge is essentially
unchanged (12.3 % → 11.9 %) while the most anomalous well is reconciled
indicates the field wells carry moderate, well-localised abstraction
rather than pervasive pumping.

**Table 2.** Siheung field wells, full pipeline (pumping-robust input +
dual constraint). \(\zeta\) is the consistency statistic; v1 column is
the unconstrained fillable-porosity calibration of the same data.

| well | days | \(P\) (mm) | \(k\) (d⁻¹) | \(U'\) (mm yr⁻¹) | \(S_y^{\star}\) | \(R^{\star}\) (mm) | recharge ratio | \(\zeta\) (σ) | v1 (%) |
|---|---|---|---|---|---|---|---|---|---|
| SH08 | 345 | 936 | 0.0101 | 1620 | 0.070 | 113 | 12.1 % | 0.0 | 27.5 |
| SH11 | 344 | 939 | 0.0187 | 1840 | 0.065 | 119 | 12.7 % | 0.2 | 9.5 |
| SH22 | 334 | 944 | 0.0049 | 2422 | 0.054 | 130 | 13.8 % | 0.6 | 19.3 |
| SH23 | 318 | 944 | 0.0050 | 1130 | 0.080 | 91 | 9.6 % | 0.6 | 13.1 |
| SH28 | 313 | 949 | 0.0041 | 1422 | 0.075 | 106 | 11.2 % | 0.2 | 6.9 |
| **mean** | | | | | | | **11.9 %** | ≤ 0.6 | 13.5 |

These field results are honest about their limitations (Section 5): the
~1-year records preclude a robust *annual* statement, the scale is supplied
by the water-balance prior rather than independently measured, and no ground
truth exists. What the method delivers is a *physically plausible,
mutually consistent, falsifiable* recharge estimate with explicit
uncertainty — not a spuriously precise point value.

---

## 5. Discussion

### 5.1 What is and is not new

The recognition that \(S_y\) dominates WTF uncertainty is old (Healy and
Cook, 2002). What is new here is (i) the *exact* statement that \(S_y\) lies
in the null space of the head likelihood (Proposition 1, Eq. 7), elevating a
known difficulty to a structural theorem; (ii) the explicit separation of
the recharge into an identifiable input \(U\) and a non-identifiable scale
\(S_y\); (iii) the recognition that the recharge scale must be imported from
an orthogonal constraint, with the catchment water balance as the natural
candidate; and (iv) the use of the prior–data consistency as a falsifiable
check. The reformulation also explains, mechanistically, why
fillable-porosity WTF engines can return apparently good recharge for the
wrong reason: an underestimated yield multiplied by an overestimated,
recession-inflated rise integral can coincidentally reproduce the right
total (Section 4.1), a compensation that is fragile to configuration and
that our factorisation removes.

### 5.2 Relation to existing inverse-theory practice

The structure we exploit is familiar from groundwater inverse problems
(Carrera and Neuman, 1986; Yeh, 1986; Hill and Tiedeman, 2007): a
parameter combination is well-determined while its factors are not. Our
contribution is to identify, for the WTF method specifically, *which*
combination is identifiable (\(U = R/S_y\)) and to attach the unidentifiable
factor to an external balance. The precision-weighted fusion (15) is the
linear-Gaussian special case of Bayesian melding (Poole and Raftery, 2000)
and of data assimilation (Evensen, 2009); the consistency statistic (16) is
a one-dimensional posterior predictive check (Box, 1980; Gelman et al.,
2013). The L-curve and discrepancy machinery (Hansen, 1992; Morozov, 1966),
sometimes proposed to "solve" the regularisation-strength problem in WTF
calibration, is — by Eq. (7) — incapable of recovering scale information and
is best reserved for genuinely ill-conditioned (not structurally
non-identifiable) problems.

### 5.3 Limitations

Three limitations bound the claims. First, the identifiable input \(U\) is
itself processing-dependent at the ±30 % level (Section 4.3); a fully
principled treatment would replace the moving-average de-biasing with a
state-space (Rauch–Tung–Striebel) smoother (Rauch et al., 1965) under an
estimated noise model, which we leave to future work. Second, the
water-balance constraint requires reliable annual precipitation,
evapotranspiration, and runoff; for sub-annual or arid records its
informativeness degrades, and the method then reverts to the prior-dominated
estimate (12). Third, the linear-reservoir model (2) assumes an unconfined,
recharge-driven, single-cell aquifer; deep or damped piezometers that
violate these assumptions (as several candidate wells in our exploratory
analysis appeared to) are outside its scope and should be screened, e.g. by
their dynamic range and \(\zeta\). Finally, our field records span ≈ 1 year;
multi-year records spanning several monsoons are required for robust annual
recharge and for testing the inter-annual stability of \(S_y^{\star}\).

### 5.4 Implications for practice

The practical recommendation is a change of reporting culture as much as of
algorithm. WTF recharge should be reported as a distribution
\(R = S_y U\) with the yield uncertainty propagated explicitly (Eq. 12), the
scale anchored to a stated external balance (Eq. 13), and the consistency
statistic (16) reported as a falsifiable check. A single recharge number
without these is, by Proposition 1, an expression of the analyst's \(S_y\)
prior, not a measurement.

---

## 6. Conclusions

1. Under the linear-reservoir model, WTF recharge factorises exactly as
   \(R = S_y\,U\): the head record identifies the recession-corrected input
   integral \(U\) but carries **zero Fisher information** about the specific
   yield \(S_y\). The head fixes the line, not the point.
2. The "operational" specific yield routinely extracted from hydrographs is
   a **recession-biased apparent quantity**, larger than the
   column-effective yield by a factor \(1+\mathcal O(\tau_r/\tau_R)\), and
   distinct from the capillary-fringe-limited near-water-table fillable
   porosity. Conflating them injects bias that head fitting cannot remove.
3. **Pumping recovery is the dominant bias** in the head-input integral
   (a ~2.5-fold inflation, exceeding observation noise), because the
   recovery is a genuine rise that na\"ive estimators read as recharge. A
   recession-baseline reconstruction — exploiting that recharge raises the
   recession envelope while pumping returns to it — removes this bias,
   cutting the input-recovery RMSE sevenfold (2.14 → 0.30) on a
   soil-heterogeneous benchmark, at a characterised ~15 % conservative
   cost.

4. The recession constant must be estimated over **multi-day** dry-spell
   tails, because serial correlation reduces an \(\sim\)1800-day record to
   \(\sim\)50 effective points and buries the single-step recession signal
   in noise.
4. The non-identifiable recharge scale is recovered by fusing a
   field-effective \(S_y\) prior with a **catchment water-balance** prior;
   their precision-weighted combination identifies \(S_y\), and their
   consistency provides a built-in, falsifiable cross-check.
5. On a single fixed synthetic pipeline the method recovers recharge at
   1.04 ± 0.17 of truth; on five Siheung wells it gives a tightly clustered,
   mutually consistent 12.3 % of precipitation, contracting an
   unconstrained 7–27 % spread.
6. The method's value is honesty: it reports what the data determine (the
   line), imports what they cannot (the scale) from an explicit external
   balance, and propagates the irreducible uncertainty into a falsifiable
   recharge posterior.

---

## Appendix A. The linear-reservoir solution and its discretisation

The continuous linear reservoir \( \dot h = q(t) - k(h-h_b)\), with
\(q(t)\) the recharge-driven input rate, has the integrating-factor
solution
\[
  h(t)-h_b = \big(h(0)-h_b\big)e^{-kt} + \int_0^t e^{-k(t-s)}q(s)\,ds .
\]
Discretising at unit step with a zero-order hold on \(q\) yields
\(h_{t+1}-h_b = e^{-k}(h_t-h_b) + \tfrac{1-e^{-k}}{k}q_t\); writing
\(u_t = \tfrac{1-e^{-k}}{k}q_t\) and approximating \(e^{-k}\approx 1-k\) for
\(k\ll1\) recovers (2). The \(\mathcal O(k^2)\) discretisation error is
negligible for \(k\sim5\times10^{-3}\). \(\square\)

## Appendix B. Derivation of the recession-bias factor (4)

Let recharge arrive as isolated pulses separated by recession intervals of
mean length \(\tau_R\). Over one cycle the rise above the antecedent
recession is \(\delta = \bar S_y^{-1}\!\int R\), while the *observed*
positive increment is reduced by the recession that proceeds during the
rise; to first order the raw positive increment is
\(\delta(1 - \tfrac12 k\,\tau_{\mathrm{rise}})\) and the inter-pulse
recession removes a further \(k(h-h_b)\tau_R\) from the denominator base.
Carrying the expansion (Appendix C of the supplement) gives
\(S_y^{\mathrm{op}} = \bar S_y\,[1 + \tfrac{\tau_r}{\tau_R} + \mathcal
O((\tau_r/\tau_R)^2)]\), i.e. (4). For S1, \(\tau_r = 1/k \approx 200\) d is
not small against \(\tau_R\), and the measured ratio 1.69 is consistent with
the leading-order estimate given the event statistics. \(\square\)

## Appendix C. Multi-day recession estimator and pooling

For each maximal dry spell of length \(\ge n_{\min}=8\) days we locate the
within-spell head maximum and retain the monotone tail. On the tail,
deficits \(d_t = h_t-h_b > d_{\min}\) are regressed as
\(\log d_t = a + b\,t\), giving \(k = 1-e^{b}\). Per-spell estimates with
\(-0.05 < k < 0.5\) are pooled by a length-weighted median, robust to the
heavy-tailed spell-to-spell variability. The estimator is consistent as the
number and length of dry spells grow and is insensitive to \(\sigma_v\) at
leading order because the regression averages the noise over the tail.
\(\square\)

## Appendix D. Failure of the near-water-table fillable-porosity engine

The transient van Genuchten fillable-porosity routine of the parent
framework evaluates (3) over the *daily* head increment. Because for
air-entry pressures of the alluvial soils considered (\(\alpha^{-1}\sim
0.07\) m for sand) the profile is essentially saturated below a few
centimetres, the integrand of (3) is \(\approx 0\) for daily rises of
5–40 mm, and a sign convention in the discrete implementation drives the
result negative, so it is clipped to a floor of \(10^{-3}\) on 98–99.8 % of
days (Section 4.1). The engine therefore cannot represent the
column-effective yield at daily resolution; our estimator bypasses it
entirely, treating \(S_y\) as an externally constrained scale rather than a
computed soil property. \(\square\)

## Appendix E. Identifiability under the full stochastic model (Kalman innovations likelihood)

Sections 2.3–2.4 used the noise-free input–output map. We confirm the
conclusion under the full stochastic model (2) with process noise
\(w_t\sim\mathcal N(0,\sigma_w^2)\). The exact likelihood is the Gaussian
innovations form of the Kalman filter (Kalman, 1960; Schweppe, 1965;
Harvey, 1989):
\[
  -\log\mathcal L = \tfrac12\sum_t\Big[\log(2\pi F_t) + \nu_t^2/F_t\Big],
\]
with one-step innovations \(\nu_t = y_t - \hat h_{t\mid t-1}\) and
innovation variances \(F_t = P_{t\mid t-1}+\sigma_v^2\). The predicted head
\(\hat h_{t\mid t-1}\) is produced by propagating (2) with the input
sequence \(u_t = R_t/S_y\); as in the deterministic case, \(\hat
h_{t\mid t-1}\) depends on \((R_t, S_y)\) only through \(u_t\), so the
innovations — and hence the likelihood — are invariant along the family
\(\mathcal F\). The process noise enlarges the innovation variances and thus
*reduces* the information about every parameter, but it cannot create
information about a direction along which the mean prediction is exactly
invariant. Hence \(\partial\nu_t/\partial S_y\equiv 0\) at fixed \(\{u_t\}\),
\(\mathcal I_{S_yS_y}=0\) under the stochastic model as well, and
Proposition 1 stands. The Kalman filter is therefore the correct estimator
of the *identifiable* state and input but is, like every head-based method,
silent on the scale. \(\square\)

## Appendix F. Effective sample size and generalised least squares

Let the head residuals follow a stationary AR(1) process \(r_t = \rho_1
r_{t-1}+e_t\). The variance of the sample mean residual over \(n\) steps is
\(\operatorname{Var}(\bar r) = \tfrac{\sigma_r^2}{n}\big[1 +
2\sum_{j=1}^{n-1}(1-j/n)\rho_1^{\,j}\big]\), which for large \(n\) tends to
\(\tfrac{\sigma_r^2}{n}\tfrac{1+\rho_1}{1-\rho_1}\); equating to
\(\sigma_r^2/n_{\mathrm{eff}}\) gives (10). The same correlation that
inflates the variance also biases ordinary-least-squares standard errors and
motivates the generalised-least-squares (Prais–Winsten / Cochrane–Orcutt)
treatment of serially correlated regression residuals (Cochrane and Orcutt,
1949; Prais and Winsten, 1954). For \(\rho_1\approx0.95\),
\((1+\rho_1)/(1-\rho_1)\approx 39\), recovering the 36-fold reduction quoted
in Section 2.6 (the empirical estimate \(N=1733,\ n_{\mathrm{eff}}\approx
48\)). The operational lesson is that head records, though long in days, are
short in information; estimators and uncertainty statements must use
\(n_{\mathrm{eff}}\), not \(n\). \(\square\)

## Appendix G. Deconvolution recovery of the input and its conditioning

By Appendix 2.9.2, \(\{u_t\}\) is the deconvolution of \(\{h_t\}\) by the
first-order filter \(H(z)\). In the time domain this is exactly (8); its
conditioning is governed by the pole \(1-k\). For \(k\to0\) (no recession)
the filter approaches a pure integrator, the deconvolution differentiates
the head, and high-frequency observation noise is amplified — the origin of
the \(\sum(u)^{+}\) noise-inflation addressed by the smoothing of
Section 2.5 and quantified in Section 4.3. The recession correction
\(+k(h_t-h_b)\) in (8) is the regularising term that distinguishes genuine
recharge-driven rises from noise-driven increments; its magnitude relative
to the noise sets the achievable signal-to-noise of \(U\). This makes
explicit that the residual ±30 % uncertainty in \(U\) is a *deconvolution
conditioning* phenomenon, reducible in principle by a model-based
(Rauch–Tung–Striebel) smoother with a calibrated noise model, which we
identify as the priority methodological extension. \(\square\)

## Appendix H. Reproducibility

All numbers, tables, and figures are regenerated from the open
implementation by the commands in `notebooks/v2/run_all.sh`: synthetic data
via `synthetic.export_for_matlab.export_all_scenarios`; Table 1 and the
recovery statistics via `notebooks/v2/honest_benchmark.py`; Figures 1–3 via
`notebooks/v2/make_identifiability_figures.py`; Table 2 via
`notebooks/v2/field_identifiability.py`; correctness via the 40-test suite
(`pytest`). Field hydrographs are retained locally for data-sharing reasons;
the synthetic benchmark is fully self-contained.

---

## References

> *Author–year associations follow standard usage; volume/page details
> should be completed and verified against a citation database (e.g.
> Crossref) before submission.*

Acharya, S., Jawitz, J.W., Mylavarapu, R.S. (2012). Analytical
expressions for drainable and fillable porosity of phreatic aquifers under
vertical fluxes from evapotranspiration and recharge. *Water Resources
Research*, 48, W11526.

Aster, R.C., Borchers, B., Thurber, C.H. (2018). *Parameter Estimation and
Inverse Problems*, 3rd ed. Elsevier.

Bartlett, M.S. (1946). On the theoretical specification and sampling
properties of autocorrelated time-series. *Supplement to the Journal of the
Royal Statistical Society*, 8(1), 27–41.

Bellman, R., Åström, K.J. (1970). On structural identifiability.
*Mathematical Biosciences*, 7, 329–339.

Beven, K. (2006). A manifesto for the equifinality thesis. *Journal of
Hydrology*, 320, 18–36.

Beven, K., Binley, A. (1992). The future of distributed models: model
calibration and uncertainty prediction. *Hydrological Processes*, 6,
279–298.

Box, G.E.P. (1980). Sampling and Bayes' inference in scientific modelling
and robustness. *Journal of the Royal Statistical Society A*, 143, 383–430.

Cao, G., Scanlon, B.R., Han, D., Zheng, C. (2016). Impacts of thickening
unsaturated zone on groundwater recharge in the North China Plain.
*Journal of Hydrology*, 537, 260–270.

Carrera, J., Neuman, S.P. (1986). Estimation of aquifer parameters under
transient and steady state conditions: 1. Maximum likelihood method
incorporating prior information. *Water Resources Research*, 22, 199–210.

Childs, E.C. (1960). The nonsteady state of the water table in drained
land. *Journal of Geophysical Research*, 65, 780–782.

Colton, D., Kress, R. (1992). *Inverse Acoustic and Electromagnetic
Scattering Theory*. Springer.

Cochrane, D., Orcutt, G.H. (1949). Application of least squares regression
to relationships containing autocorrelated error terms. *Journal of the
American Statistical Association*, 44, 32–61.

Cramér, H. (1946). *Mathematical Methods of Statistics*. Princeton
University Press.

Crosbie, R.S., Binning, P., Kalma, J.D. (2005). A time series approach to
inferring groundwater recharge using the water table fluctuation method.
*Water Resources Research*, 41, W01008.

Cuthbert, M.O. (2010). An improved time series approach for estimating
groundwater recharge from groundwater level fluctuations. *Water Resources
Research*, 46, W09515.

Cuthbert, M.O. (2014). Straight thinking about groundwater recession.
*Water Resources Research*, 50, 2407–2424.

DeGroot, M.H. (1970). *Optimal Statistical Decisions*. McGraw-Hill.

Delin, G.N., Healy, R.W., Lorenz, D.L., Nimmo, J.R. (2007). Comparison of
local- to regional-scale estimates of ground-water recharge in Minnesota,
USA. *Journal of Hydrology*, 334, 231–249.

Dewandel, B., Lachassagne, P., Bakalowicz, M., Weng, P., Al-Malki, A.
(2003). Evaluation of aquifer thickness by analysing recession hydrographs.
*Journal of Hydrology*, 274, 248–269.

Evensen, G. (2009). *Data Assimilation: The Ensemble Kalman Filter*, 2nd
ed. Springer.

Gelman, A., Carlin, J.B., Stern, H.S., Dunson, D.B., Vehtari, A., Rubin,
D.B. (2013). *Bayesian Data Analysis*, 3rd ed. CRC Press.

Gillham, R.W. (1984). The capillary fringe and its effect on water-table
response. *Journal of Hydrology*, 67, 307–324.

Hansen, P.C. (1992). Analysis of discrete ill-posed problems by means of the
L-curve. *SIAM Review*, 34, 561–580.

Hansen, P.C. (1998). *Rank-Deficient and Discrete Ill-Posed Problems*. SIAM.

Hansen, P.C., O'Leary, D.P. (1993). The use of the L-curve in the
regularization of discrete ill-posed problems. *SIAM Journal on Scientific
Computing*, 14, 1487–1503.

Healy, R.W. (2010). *Estimating Groundwater Recharge*. Cambridge University
Press.

Healy, R.W., Cook, P.G. (2002). Using groundwater levels to estimate
recharge. *Hydrogeology Journal*, 10, 91–109.

Heppner, C.S., Nimmo, J.R. (2005). A computer program for predicting
recharge with a master recession curve. *USGS Scientific Investigations
Report* 2005-5172.

Harvey, A.C. (1989). *Forecasting, Structural Time Series Models and the
Kalman Filter*. Cambridge University Press.

Hill, M.C., Tiedeman, C.R. (2007). *Effective Groundwater Model
Calibration*. Wiley.

Jazwinski, A.H. (1970). *Stochastic Processes and Filtering Theory*.
Academic Press.

Johnson, A.I. (1967). Specific yield — compilation of specific yields for
various materials. *USGS Water-Supply Paper* 1662-D.

Kalman, R.E. (1960). A new approach to linear filtering and prediction
problems. *Journal of Basic Engineering*, 82, 35–45.

Kim, G.-B., et al. (2014). Estimation of groundwater recharge in Korea:
review and analysis. *Journal of the Geological Society of Korea* (and
references therein).

Maillet, E. (1905). *Essais d'hydraulique souterraine et fluviale*.
Hermann, Paris.

Meinzer, O.E. (1923). *Outline of Ground-Water Hydrology, with Definitions*.
USGS Water-Supply Paper 494.

Moon, S.-K., Woo, N.C., Lee, K.S. (2004). Statistical analysis of
hydrographs and water-table fluctuation to estimate groundwater recharge.
*Journal of Hydrology*, 292, 198–209.

Morozov, V.A. (1966). On the solution of functional equations by the method
of regularization. *Soviet Mathematics Doklady*, 7, 414–417.

Mualem, Y. (1976). A new model for predicting the hydraulic conductivity of
unsaturated porous media. *Water Resources Research*, 12, 513–522.

Nachabe, M.H. (2002). Analytical expressions for transient specific yield
and shallow water table drainage. *Water Resources Research*, 38, 1193.

Nimmo, J.R., Horowitz, C., Mitchell, L. (2015). Discrete-storm
water-table fluctuation method to estimate episodic recharge. *Groundwater*,
53, 282–292.

Obuobie, E., Diekkrueger, B., Agyekum, W., Agodzo, S. (2012). Groundwater
level monitoring and recharge estimation in the White Volta River basin of
Ghana. *Journal of African Earth Sciences*, 71, 80–86.

Poole, D., Raftery, A.E. (2000). Inference for deterministic simulation
models: the Bayesian melding approach. *Journal of the American Statistical
Association*, 95, 1244–1255.

Posavec, K., Bačani, A., Nakić, Z. (2006). A visual basic spreadsheet macro
for recession curve analysis. *Ground Water*, 44, 764–767.

Prais, S.J., Winsten, C.B. (1954). Trend estimators and serial correlation.
*Cowles Commission Discussion Paper* 383.

Raue, A., Kreutz, C., Maiwald, T., Bachmann, J., Schilling, M.,
Klingmüller, U., Timmer, J. (2009). Structural and practical identifiability
analysis of partially observed dynamical models by exploiting the profile
likelihood. *Bioinformatics*, 25, 1923–1929.

Rothenberg, T.J. (1971). Identification in parametric models.
*Econometrica*, 39, 577–591.

Rao, C.R. (1945). Information and accuracy attainable in the estimation of
statistical parameters. *Bulletin of the Calcutta Mathematical Society*, 37,
81–91.

Rauch, H.E., Tung, F., Striebel, C.T. (1965). Maximum likelihood estimates
of linear dynamic systems. *AIAA Journal*, 3, 1445–1450.

Schweppe, F.C. (1965). Evaluation of likelihood functions for Gaussian
signals. *IEEE Transactions on Information Theory*, 11, 61–70.

Risser, D.W., Conger, R.W., Ulrich, J.E., Asmussen, M.P. (2005). Estimates
of ground-water recharge based on streamflow-hydrograph methods,
Pennsylvania. *USGS Open-File Report* 2005-1333.

Scanlon, B.R., Healy, R.W., Cook, P.G. (2002). Choosing appropriate
techniques for quantifying groundwater recharge. *Hydrogeology Journal*, 10,
18–39.

Sophocleous, M. (1985). The role of specific yield in ground-water recharge
estimations: a numerical study. *Ground Water*, 23, 52–58.

Sophocleous, M. (1991). Combining the soilwater balance and water-level
fluctuation methods to estimate natural groundwater recharge. *Journal of
Hydrology*, 124, 229–241.

Sophocleous, M. (2002). Interactions between groundwater and surface water:
the state of the science. *Hydrogeology Journal*, 10, 52–67.

Thiébaux, H.J., Zwiers, F.W. (1984). The interpretation and estimation of
effective sample size. *Journal of Climate and Applied Meteorology*, 23,
800–811.

Tikhonov, A.N., Arsenin, V.Y. (1977). *Solutions of Ill-Posed Problems*.
Winston & Sons.

van Genuchten, M.T. (1980). A closed-form equation for predicting the
hydraulic conductivity of unsaturated soils. *Soil Science Society of
America Journal*, 44, 892–898.

White, W.N. (1932). *A Method of Estimating Ground-Water Supplies Based on
Discharge by Plants and Evaporation from Soil*. USGS Water-Supply Paper 659-A.

Walter, E., Pronzato, L. (1997). *Identification of Parametric Models from
Experimental Data*. Springer.

Wirgin, A. (2004). The inverse crime. *arXiv:math-ph/0401050*.

Yeh, W.W.-G. (1986). Review of parameter identification procedures in
groundwater hydrology: the inverse problem. *Water Resources Research*, 22,
95–108.
