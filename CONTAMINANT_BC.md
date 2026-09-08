# Contaminant solver — boundary conditions, foliage, and indoor filtration

Covers the transport boundary conditions (as implemented in the D3Q7 live-flow
forward scalar, the production path), the porous-foliage handling, and the indoor
filtration metric in the exposure objective. The scalar rides the same geometry as
the airflow solve, so its wall/outlet/top conditions mirror the flow's.

## 1. Transport boundary conditions

Cell types (shared with the flow): FLUID (0), GROUND (1), SHELL (2, solid building
or porous park), INDOOR (3). The advection–diffusion of concentration C is subject
to four conditions, classified per face:

| Boundary | Condition | Discrete treatment (D3Q7) |
|---|---|---|
| **Solid wall** (building, ground) | no advective flux + surface deposition | face carries zero advective flux (`advect_dC`: solid neighbour skipped, bounce-back handles the wall); a per-cell dry-deposition sink removes C at the wall |
| **Domain outlet** (open face) | advective outflow, zero inflow | face velocity extrapolated from the interior cell; interior C carried out, nothing enters |
| **Domain top / lateral** (free-slip / symmetry) | zero-flux | no advective or diffusive loss through the lid or symmetry planes |
| **Source** | emission (inflow of the scalar) | accidental burst: mass M injected over the ~2 s pulse at the source cell(s), then off |

**Deposition closure (particulate).** At a solid face the dry-deposition velocity
v_d (Zhang 2001) is converted to a per-link sticking probability α = 8·v_dep_lb in
the D3Q7 half-way bounce-back closure: the absorbed flux ≈ (α/8)·C removes scalar
from the adjacent fluid cell into a deposited-mass map (`I.dep`). Gravitational
settling w_s (Stokes + Cunningham) adds to the flux on the upward-facing ground
face. A passive gas is the special case v_d = 0, w_s = 0 (pure no-flux wall).

**Why the outlet + deposition pair matters.** Together, advective outflow and
surface deposition drain the released mass, M = M_out + M_dep, so the cumulative
dose Θ = ∫C dt converges and the accidental-burst run self-terminates at 99%
clearance. A closed domain with no deposition would never drain — Θ would diverge.

## 2. Foliage handling (porous parks)

Parks are **SHELL cells with a calibrated permeability** — the one place the scalar
(and the wind) is *partly transmitted* rather than fully blocked. The partial
("grey") bounce-back reflects a fraction σ = 1 − perm of each population and
transmits the rest, giving a distributed momentum/scalar sink that represents a tree
canopy (Dardis & McCloskey 1998; Walsh et al. 2009).

**Calibration.** The permeability is set to reproduce canopy drag F = ρ·C_d·A·U²
(Wilson & Shaw 1977, C_d = 0.2; leaf-area density A ≈ 1.3–1.6 m²/m³ for an in-leaf
urban tree crown, Karlsruhe Norway-maple measurements). The exact momentum law of
the closure, ρu_out = (2·perm − 1)·ρu_in (verified to machine precision), yields the
resolution- and velocity-explicit calibration

    perm = 1 − ½·C_d·A·dx·u_lb

(e.g. ≈ 0.98–0.998 for dx = 0.5–4 m), replacing the previous hand-set 0.8, which was
~20× too resistive. Honest caveats carried in PARK_POROSITY.md: the closure is a
*linear* (Darcy) drag matched to the *quadratic* canopy drag at one reference speed
(velocity-specific), it is isotropic and single-layer, and the value is *verified*
(matches the closure and the derived formula) but not yet *validated* against a
measured canopy attenuation profile.

## 3. Indoor filtration metric (exposure solver)

Indoor exposure is **not** a transport boundary condition — buildings are
impermeable to the plume (wind and scalar go around them). It is a **linear
receptor-side reduction** of the outdoor concentration, following the single-zone
infiltration model of Riley, McKone, Lai & Nazaroff (2002):

    F_inf(d) = P(d)·a / (a + k(d)) ,     C_in(d) = F_inf(d)·C_out(d)

for particle size d, with P the penetration factor (Liu & Nazaroff 2001), a the
air-exchange rate, and k the indoor deposition loss rate (Lai & Nazaroff 2000). The
indoor concentration is a constant multiple of the outdoor concentration the
building's envelope sees — so no indoor transport is resolved, and the operator stays
linear (the adjoint/superposition properties survive).

**In the objective.** The exposure J = ⟨w, Θ⟩ folds F_inf into the receptor weight
w, split by the NHAPS time-activity budget (Klepeis 2001, f_in ≈ 0.87, f_out ≈ 0.075):

    w(x) = envelope cells:  f_in · F_inf   (indoor occupants breathe F_inf·C_out)
         + pedestrian cells: f_out          (outdoor occupants breathe C_out)

Deposited mass on building surfaces is a *sink*, NOT an indoor-exposure term — the
two pathways sample the same airborne C_out, separated by the time budget
(f_in + f_out ≤ 1), so there is no double-count. Default F_inf = 0.62 (Allen 2012,
MESA Air); per-building P, a, k are the city_builder7-bridge refinement. Assumptions
(well-mixed zone, no indoor sources, steady limit, constant wind-independent a) are
enumerated in INFILTRATION_MODEL.md.

## Sources
- Zhang, L. et al. (2001). Size-segregated dry deposition scheme. *Atmos. Environ.* 35:549.
- Wilson, N.R. & Shaw, R.H. (1977). Higher-order closure for canopy flow. *J. Appl. Meteorol.* 16:1197.
- Dardis, O. & McCloskey, J. (1998). Partial-bounce-back LBM for porous media. *Phys. Rev. E* 57:4834.
- Riley, W.J., McKone, T.E., Lai, A.C.K. & Nazaroff, W.W. (2002). Indoor PM of outdoor origin. *Environ. Sci. Technol.* 36:200.
- Liu, D.L. & Nazaroff, W.W. (2001). Pollutant penetration across building envelopes. *Atmos. Environ.* 35:4451.
- Lai, A.C.K. & Nazaroff, W.W. (2000). Indoor particle deposition. *J. Aerosol Sci.* 31:463.
- Allen, R.W. et al. (2012). Residential infiltration of outdoor PM2.5 (MESA Air). *Environ. Health Perspect.*
- Klepeis, N.E. et al. (2001). NHAPS time-activity. *J. Expo. Anal. Environ. Epidemiol.* 11:231.
