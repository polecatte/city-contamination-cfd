# Urban LBM Dispersion-Optimization — Architecture & Mathematical Foundations

This document describes the full pipeline — from the parametric city builder to
the Bayesian optimizer — and the mathematics under each stage, with sources. Per
the project convention, every physics/algorithmic decision names its reference.

A note on provenance: equations tied to code I have read (unit conversion,
settling, deposition, infiltration, occupancy, inlet) are stated as implemented;
standard method equations (LBM equilibrium, MRT, WALE, GP/EI) are stated in
their canonical form. Where a citation is the project's own and I could not
verify it independently, it is marked as such.

---

## 0. Pipeline overview

```
 design parameters x ∈ [0,1]^9
        │  to_physical (param_space.py)
        ▼
 ┌─────────────┐   ┌────────────┐   ┌──────────────┐   ┌────────────────┐
 │ city builder │→ │ voxelizer  │→ │ LBM flow      │→ │ scalar transport│
 │ (geometry,   │   │ (D3Q19 grid│   │ (D3Q19 MRT +  │   │ (D3Q7 ADE,      │
 │  zoning, pop)│   │  + types)  │   │  WALE LES)    │   │  settling, dep) │
 └─────────────┘   └────────────┘   └──────────────┘   └────────────────┘
                                                              │
                          ┌───────────────────────────────────┘
                          ▼
                 ┌──────────────────┐   ┌──────────────┐   ┌────────────┐
                 │ exposure model   │→ │ objective J   │→ │ Bayesian   │
                 │ (3 pathways,      │   │ (scenario-    │   │ optimizer  │→ x*
                 │  occupancy, F_inf)│   │  averaged)    │   │ (multi-res)│
                 └──────────────────┘   └──────────────┘   └────────────┘
```

The chain is a deterministic map `x → J(x)`; the optimizer minimizes `J`. Because
an optimizer exploits any unphysical behavior in the map, the design philosophy
fixes scenario variables (population, wind, source, particle-size distribution) and
keeps only **layout** in the search space.

---

## 1. City builder (`city_builder7.h`)

A strictly circular **ring/central-core** model. Blocks tile the city; each is
assigned a land use, height, footprint, population, and effective occupancy.

### 1.1 Radial zoning with business spread (`patchiness`)
Land use is driven by a **business-assignment metric** — by default the strictly
circular radial distance `r` from the business centre. Business blocks are filled
from the lowest-metric block outward (a solid central circle); the count is set by
the employment balance (§1.4), not a knob. Remaining blocks default to `RES_HIGH`
(and become `RES_LOW` where short, §1.3); parks are carved out by §1.4/Step-4.

`patchiness` ∈ [0,1] blends that metric toward a spatially-coherent value-noise
field, spreading the business district off-centre:

  r_eff = (1 − patchiness)·r + patchiness · n(x,y)·R      n ∈ [0,1]

`n` is value noise on a hashed integer lattice with quintic smoothstep
interpolation (`PATCH_SCALE_M ≈ 160 m` ≈ 3–4 blocks), deterministic (same city ⇒
same pattern) and resolution-independent (evaluated in metres — warm-restart /
optimizer safe). At `patchiness=0` business is a concentric core; at `1` it
follows the noise into coherent off-centre patches, relocating the work-population.
*Source:* value noise with smoothstep — Perlin (1985; 2002, "Improving Noise,"
ACM SIGGRAPH); Ebert et al. (2003), *Texturing & Modeling*.

(Removed knobs, for readers of older docs: the earlier radial-perturbation
patchiness, the `biz_inner_frac` central void, and the `mixed_frac` mixed-use band
are all retired — the model is now PARK / BUSINESS / RES_HIGH / RES_LOW only.)

### 1.2 Building heights
Heights are **pure morphology** — decoupled from population (v8). A CBD field
decays with radius on top of a fixed baseline:

  h(r) = BASE_HEIGHT_M + cbd_peak · exp(−cbd_decay · r²)

`BASE_HEIGHT_M = 9 m` (3 floors, fixed — the former `base_height` knob carried
~4% of variance and was redundant with the CBD tail; 3 floors sits just below the
RES_LOW threshold so the periphery is low-density). Per-block heterogeneity is a
**log-normal** multiplier `h ← h · exp(√12 · roughness · ξ)`, `ξ = block_noise ∈
[−0.5,0.5]` (std 1/√12), so `√12` makes σ(ln h) ≈ `roughness` ≈ the height
coefficient of variation. Result is clamped by a slenderness cap `h_max =
SLENDERNESS(=7) · min(block_w, block_d)`. `cbd_decay` (1e−6…3e−5 m⁻²) sets the
core extent (e-folding radius 1/√decay ≈ 1000 m … 183 m). *Note:* this `roughness`
is building-height heterogeneity, **not** aerodynamic z₀ (which enters via the
inlet, §3.6). *Sources:* height-CoV effect on canopy flow — Xie, Coceal & Castro
(2008), *Boundary-Layer Meteorol.* 129:1; Nakayama, Takemi & Nagai (2011),
*J. Appl. Meteorol. Climatol.* 50:1692.

### 1.3 Footprint, setback, and low-density coverage
The global `coverage` fill-ratio is **retired** (v8): every building fills its lot
to a fixed **4 m setback** (`SETBACK_M`; `cov ≡ 1`). Plan-area density is now
controlled physically by `street_width` + block size, not an abstract ratio.
Blocks that resolve to ≤ `RES_LOW_MAX_FLOORS` (4) floors become **RES_LOW** and
are shrunk to `RES_LOW_COVERAGE = 25 %` of the lot (a house with a yard; U.S.
Census 2024 median ≈ 2,146 ft² home / 8,506 ft² lot). Floor area = footprint ×
floors, floors = ⌊h / floor_height⌋ (residential 3 m, business 4 m).

### 1.4 Population and employment
Total population is a **FIXED** scenario input `population_total` (v8 — replaces
the former `target_density`), so designs are compared at equal headcount (total
exposure Σ w·C, not per-capita). It is allocated across residential floor area by
capacity. Employment is balanced: workers = `LABOR_RATE`(0.47) · population (BLS
CPS: 62% LFPR × 78% aged 16+), and business floor area is grown from the core
outward (two-pass α solve) until it can hold that labour force at ~14 m²/worker
(U.S. GSA P100 2024). This keeps population **emergent from built form** — no
free population field the optimizer could teleport into a low-C corner.

### 1.5 Effective occupancy (time budget)
Each block carries an **effective inhabitance** `eff_inh` = people present,
time-averaged. The microenvironment time budget is the 24-h population average
from the **National Human Activity Pattern Survey** — *Klepeis et al. (2001),*
*J. Expo. Anal. Environ. Epidemiol.* 11(3):231–252 (n=9386, EPA): residence
0.69, other-indoor 0.18, in-vehicle 0.055, outdoors 0.076 — mapped to pathways
(residence→residential buildings, other-indoor→business, outdoors→parks +
street, in-vehicle→street). The split is deliberately **24-h averaged with no
time-of-day profile**: the release time is unknown/adversarial, so occupancy is
the expectation over a uniformly-distributed release hour (same robustness
principle as the fixed density and wind aggregation).

---

## 2. Voxelization (`voxelize.h`)

The city is rasterized onto the D3Q19 grid. Cell types: `FLUID` (outdoor air),
`GROUND` (solid, full bounce-back), `SHELL` (envelope), `INDOOR` (interior air).

- **Solid buildings** (recommended): non-park buildings → full bounce-back
  (perm = 0), aerodynamically correct. Parks → high-permeability porous canopy.
  *Sources:* solid bluff-body urban CFD (COST 732 / Franke et al. 2007;
  Tominaga et al. 2008); trees-as-porous-media (Merlier, Jacob & Sagaut 2018,
  *Atmos. Environ.* 195, 89–103).
- **Vertical extent:** `nz` ≥ 5H headroom above the tallest building, rounded to
  a power of two. *Source:* COST 732 / Franke et al. (2007) domain guidance.
- **Per-cell deposition velocity:** size-dependent (Zhang et al. 2001, §5) when a
  particle diameter is supplied, else constant per surface type.

---

## 3. LBM flow solver (`lbm_solver.cpp`, `lbm_kernels_*`)

### 3.1 Lattice Boltzmann fundamentals
The discrete-velocity Boltzmann equation on a D3Q19 lattice evolves particle
distributions `f_i(x,t)` by collide-and-stream:

  f_i(x + c_i Δt, t + Δt) = f_i(x,t) + Ω_i(f)

with macroscopic moments ρ = Σ f_i, ρu = Σ c_i f_i. The discrete equilibrium is

  f_i^eq = w_i ρ [ 1 + (c_i·u)/c_s² + (c_i·u)²/(2c_s⁴) − u²/(2c_s²) ]

`c_s² = 1/3` (lattice units), `w_i` the D3Q19 weights. *Sources:* Chen & Doolen
(1998), *Annu. Rev. Fluid Mech.* 30, 329; Succi (2001), *The Lattice Boltzmann
Equation*; Krüger et al. (2017), *The Lattice Boltzmann Method* (Springer).

### 3.2 MRT collision
Collision is performed in moment space, `m = M f`, with a diagonal relaxation
matrix `S`: `Ω = −M⁻¹ S (m − m^eq)`. MRT relaxes different moments at different
rates, improving stability at high Reynolds number over single-relaxation BGK.
*Sources:* d'Humières (2002), *Phil. Trans. R. Soc. A* 360, 437; Lallemand & Luo
(2000), *Phys. Rev. E* 61, 6546.

### 3.3 Sub-grid turbulence (WALE LES)
The unresolved scales are modeled with the Wall-Adapting Local Eddy-viscosity
(WALE) closure:

  ν_t = (C_w Δ)² · (S^d_{ij} S^d_{ij})^{3/2} / [ (S̄_{ij} S̄_{ij})^{5/2} + (S^d_{ij} S^d_{ij})^{5/4} ]

with `C_w ≈ 0.325`, `Δ` the filter width (cell size), `S^d` the traceless
symmetric part of the squared velocity-gradient tensor. WALE gives the correct
near-wall `ν_t ~ y³` scaling without dynamic procedures. The effective
relaxation time becomes `τ = 3(ν + ν_t) + ½`. *Source:* Nicoud & Ducros (1999),
*Flow Turbul. Combust.* 62, 183.

### 3.4 Physical ↔ lattice units, and the viscosity floor
Low-Mach scaling fixes the lattice inlet speed `u_lb = Ma·c_s` (Ma = 0.1), then

  dt = u_lb · Δx / U_inlet,   ν_lb = ν_phys · dt / Δx²,   D_lb = D_mol · dt / Δx²

A stability floor `τ ≥ 0.503` (i.e. ν_lb, D_lb ≥ ~1e−3) is enforced. **Consequence
(documented limitation):** at practical resolutions the floor — not the physical
value — sets the effective Reynolds/Péclet number, so the simulated building
Re ≈ 577 sits below the Re-independence threshold (Re_H ≳ 1.1×10⁴; Snyder 1972),
and wind *speed* barely affects the pattern. Wind *direction* is the meaningful
flow variable.

### 3.5 Boundary conditions
- **Outlet:** zero-gradient convective.
- **Top (+z) and lateral (±y):** free-slip / symmetry (specular reflection of the
  wall-normal populations) at ≥5H clearance. This is the COST-732 / AIJ symmetry
  recommendation for urban CFD (Franke et al. 2007; Tominaga et al. 2008, *JWEIA*
  96:1749). *Bug-fix history:* the lateral faces previously used a clamped
  self-pull that recirculated tangential momentum and contaminated the near-boundary
  field (max|u| at the boundary reached ~2.3× freestream); replaced with true
  specular reflection mirroring the top lid.
- **Solid walls:** half-way bounce-back (no-slip).
- **Porous (parks):** partial bounce-back with reflected fraction β — a fraction
  β of the population is bounced, (1−β) transmitted, giving a tunable canopy
  resistance. *Source (method family):* partial-bounce-back porous LBM, e.g.
  Walsh, Burwinkle & Saar (2009), *Comput. Geosci.* 35, 1186 (cite as the
  partial-bounce-back family; the project's β values are calibration targets).
- *Note:* the ±y lateral BC is a symmetry plane, **not** a fresh ABL inflow, so
  oblique wind (`wind_direction` ≠ 0) has no upstream fetch in y and remains staged
  out of the live search until an oblique-inflow validation passes.

### 3.6 Sheared turbulent ABL inlet (`abl_inlet.h`)
Mean profile — neutral-ABL log law:

  U(z) = (u*/κ) · ln((z − d + z₀)/z₀),   u* = U_ref κ / ln((z_ref − d + z₀)/z₀)

κ ≈ 0.41. *Source:* Richards & Hoxey (1993), *J. Wind Eng. Ind. Aerodyn.* 46–47,
145; homogeneity caveats Hargreaves & Wright (2007); Blocken, Stathopoulos &
Carmeliet (2007). Turbulence — Random Flow Generation, a divergence-free random-
Fourier field that is a closed form of (x,y,z,t):

  v(x,t) = √(2/N) Σ_{n=1}^N [ p_n cos(k_n·x̃ + ω_n t̃) + q_n sin(k_n·x̃ + ω_n t̃) ],   k_n·p_n = k_n·q_n = 0

*Sources:* Kraichnan (1970), *Phys. Fluids* 13, 22; Smirnov, Shi & Celik (2001),
*J. Fluids Eng.* 123, 359. Fluctuation intensities follow surface-layer
similarity σ_u:σ_v:σ_w ≈ 2.5:1.9:1.25 × u* (Panofsky & Dutton 1984; Stull 1988).
*(Wired on the CPU backend; the GPU mirror is specified but not yet wired.)*

---

### 3.7 Mean-flow time-averaging (`lbm_solver.cpp run()`)
The scalar transport (§4) runs on a **frozen mean flow**, so the flow phase must
produce a *statistically stationary time-average*, not an instantaneous snapshot
(the turbulent RFG inlet never reaches a steady state). Phase A is therefore two
stages: **A1 spin-up** flushes the start-up transient (default 2 domain
flow-throughs `n_ft = L/u_lb`); **A2** then accumulates a running mean
(`ux_sum/uy_sum/uz_sum/rho_sum`, sample count `n_avg`) and checks
**statistical stationarity** every `window` steps via a probe (max |ū|): the mean
is declared stationary when the probe's relative change is below `stat_tol` for two
consecutive windows *and* at least `min_avg = 3·n_ft` samples have accumulated.
The downloaded velocity and density fields are `sum/n_avg`. *Bug-fix history:*
averaging was previously never wired into `run()` (accumulate was never called, so
`n_avg = 0` and the "mean flow" was actually an instantaneous snapshot); the old
warm-up used an instantaneous RMS(Δu)/U criterion that never converges for
turbulent inflow. The `smoke` validation subcommand now asserts `n_avg > 0` before
any run trusts the mean. Config knobs: `spinup_steps`, `avg_steps` (= window),
`avg_threshold` (= stat_tol), `max_warmup` (cap); all auto-default from `n_ft`.

---

### 4.1 Advection–diffusion LBM
A separate D3Q7 distribution `g_i` carries the contaminant; its moments give the
concentration `C = Σ g_i` advected by the flow `u` with diffusivity `D`. The
equilibrium is the first-order advection-diffusion equilibrium
`g_i^eq = w_i C (1 + c_i·u / c_s²)`. *Sources:* advection-diffusion LBM —
Krüger et al. (2017), ch. 8; van der Sman & Ernst (2000), *J. Comput. Phys.* 160,
766. *(Implementation note/known issue: the D3Q7 sound speed `c_s² = 1/4` differs
from the hard-coded 1/3, leaving a normalization offset in absolute concentration
— flagged for correction.)*

### 4.2 Turbulent diffusion
Turbulent scalar mixing uses a turbulent Schmidt number: `D_t = ν_t / Sc_t`,
`Sc_t ≈ 0.7`. *Source:* Tominaga & Stathopoulos (2007), *Atmos. Environ.* 41,
8091 (Sc_t for CFD dispersion).

### 4.3 Numerical flux limiting
A van-Leer TVD flux limiter controls the high-Péclet oscillations of the bare
scheme (positivity-preserving). *Sources:* van Leer (1974), *J. Comput. Phys.*
14, 361; Sweby (1984), *SIAM J. Numer. Anal.* 21, 995.

### 4.4 Gravitational settling
Per particle size, terminal velocity by Stokes' law with Cunningham slip:

  w_s = (ρ_p − ρ_air) g d_p² C_c / (18 μ),   C_c = 1 + Kn(1.257 + 0.4 e^{−1.1/Kn}),   Kn = 2λ/d_p

added as a downward drift in the D3Q7 scalar. *Sources:* Hinds (1999), *Aerosol
Technology*; Seinfeld & Pandis (2016), *Atmospheric Chemistry and Physics*.

### 4.5 Polydisperse source (`psd.h`)
The contaminant source is a **lognormal activity distribution** parameterized by
the Activity Median Aerodynamic Diameter (MMAD) and GSD, discretized into N
sectional bins of mass fraction `f_i` and representative diameter `d_ae,i`;
the physical diameter for settling is `d_phys = d_ae √(ρ₀/ρ_p)` (ρ₀ = 1000).
Each bin is transported with its own w_s and size-dependent deposition; fields
are summed Σ f_i (·)_i. *Source:* lognormal aerosol size distributions and aerodynamic diameter —
Hinds (1999), *Aerosol Technology*, 2nd ed.; typical defaults MMAD 1 µm, GSD 2.
*Finalized scheme:* MMAD = 1 µm, GSD = 2, 6 bins (mass fractions
2.2/13.6/34.2/34.2/13.6/2.2 %, aerodynamic diameters 0.18–5.7 µm).

### 4.6 Source and time integration
A point source emits `Q` per step at the source cell for the release duration;
the solver records the instantaneous concentration and the **time-integrated air
concentration** TIAC = ∫C dt (the airborne inhalation driver).

---

## 5. Dry deposition (`deposition.h`)

Total deposition velocity adds gravitational settling and surface capture in
parallel, `v_d = v_g + v_ds`, with the resistance/collection model

  v_ds = ε₀ u* (E_B + E_IM + E_IN) R₁,
  E_B = Sc^{−γ},   E_IM = (St/(α+St))²,   E_IN = ½ (d_p/A)²,   R₁ = e^{−√St}

(Brownian diffusion, inertial impaction, interception, rebound), giving the
characteristic deposition-velocity minimum in the accumulation mode. Per-surface
land-use parameters {A, α, γ} map PARK→vegetation (strong capture) vs
buildings→urban/smooth. *Sources:* Zhang, Gong, Padro & Barrie (2001), *Atmos.
Environ.* 35, 549, on the Slinn (1982) framework; refinements Petroff & Zhang
(2010); Emerson et al. (2020), *PNAS* 117, 26076. **Result:** physical PARK
deposition (~0.1–0.4 cm/s) is 10–30× lower than the old hand-set 3 cm/s,
removing an optimizer-exploitable "over-greening" knob.

---

## 6. Exposure

### 6.1 Two pathways
Effective exposure = airborne + deposited-surface exposure. Airborne exposure
scales with airborne TIAC; deposited-surface exposure with deposited contaminant.
The two-pathway (airborne/deposited) decomposition is standard aerosol-exposure practice.

### 6.2 Occupancy weighting and indoor infiltration (`occupancy.h`, `infiltration.h`)
People are distributed across pathways by the NHAPS budget (§1.5). Indoor
occupants breathe infiltration-reduced air; the steady indoor/outdoor ratio with
penetration, deposition, and **indoor filtration** is

  C_in/C_out = (P·a_inf + (1−η_mv)·a_mech) / (a_inf + a_mech + k + λ_filt)

P = size-dependent penetration (Liu & Nazaroff 2003; Stephens & Siegel 2012),
a = air-exchange rate, k = indoor deposition (Lai & Nazaroff 2000), λ_filt =
filtration loss (= CADR/V for a cleaner). *Sources:* Nazaroff (2004), *Indoor
Air* 14(s7):175; Liu & Nazaroff (2001), *Atmos. Environ.* 35, 4451; Chen & Zhao
(2011), *Atmos. Environ.* 45, 275. **Street** occupants (in-vehicle + sidewalk
share) are placed on the road network and breathe full outdoor air (I/O = 1, no
infiltration) — the most exposed pathway per capita.

### 6.3 Objective: concentration × effective inhabitance
The optimization objective is the effective-inhabitance-weighted concentration,

  J = Σ_locations C(location) · inh_eff(location),

i.e. the inner product of the dispersed concentration field (mass-weighted
TIAC) and the **effective inhabitance** map. `inh_eff` (`eff_inh`) already folds
in the NHAPS occupancy time budget (§1.5), so it represents where and how much
people are present; C is the time-integrated air concentration each population
parcel is exposed to — the facade-adjacent outdoor air for building occupants and
the local cell for street/park occupants. This is a raw exposure functional
(concentration·person·time): no exposure weights, no infiltration factor F_inf,
and no deposited-surface exposure term. It is a closed functional of the dispersed concentration
field and the inhabitance map. (Contaminant-specific exposure weighting and the indoor
F_inf reduction remain available in `infiltration.h` for a later
exposure-calibrated variant; the present objective is the concentration·inhabitance
product itself.)

---

### 6.4 Reverse (adjoint) solver — source-side evaluation
The same linear objective can be evaluated from the source side via the discrete
adjoint of the scalar transport on the frozen mean flow. Forcing the adjoint with
the receptor field w(x) yields the exposure footprint F(x) (effective exposure per
unit release at x); then J = Σ_x F(x)·s(x) for any source distribution s. This
matches the forward Σ w·C exactly (source–receptor reciprocity) and lets one solve
score any/all source distributions. To keep the forward and reverse solvers in
exact agreement the objective uses a consistent first-order linear-upwind pair
(M and its exact transpose Mᵀ), so there is no limiter nonlinearity to reconcile;
the production van Leer D3Q7 scalar (§4) is untouched and used for physical
dispersion. Source mask s(x) is uniform over open city spaces (roads + parks) at
ground level. See `REVERSE_SOLVER_NOTE.md` for the full derivation, the gather-form
kernels (CPU `adjoint_transport.h`, GPU `adjoint_transport_gpu.cu`), and the
machine-precision reciprocity verification.

---

## 7. Objective and optimizer (`param_space.py`, `run_optimization.py`)

### 7.1 Search space and scenario constants
The optimizer searches a **9-D** unit cube mapped to physical layout parameters by
`to_physical` (linear or log scaling). The dimensions are `block_w`, `block_d`,
`cbd_peak`, `cbd_decay` (log), `patchiness`, `park_centrality`, `park_fraction`,
`roughness`, and `street_width`; see `param_space.py` for the exact bounds and a
one-line justification on each, and `PARAM_SPACE_NOTE.md` for the full rationale.
**Scenario constants** — `population_total` (fixed headcount, replacing the former
`target_density`), wind direction, source location, and the particle-size
distribution — are *fixed* outside the search space, because each scales or steers
the objective in ways an optimizer would exploit (e.g. depopulating the city to
cut collective exposure). Population is kept **emergent from built form** for the same
reason. `wind_direction` is staged but inactive (the lateral BCs are not yet a
fresh oblique-inflow — see §3.5 / `param_space.py`). The robust objective is the
aggregate over a wind rose / source ensemble; the present configuration runs one
direction and should be aggregated before the optimized layout is trusted.

### 7.2 Bayesian optimization
A Gaussian-process surrogate models `J(x)`: prior `f ~ GP(μ, k)`, posterior mean/
variance closed-form given evaluations, typically a Matérn kernel. The next
point maximizes Expected Improvement,

  EI(x) = E[max(0, f_best − f(x))] = (f_best − μ(x))Φ(z) + σ(x)φ(z),   z = (f_best − μ(x))/σ(x)

*Sources:* Jones, Schonlau & Welch (1998), *J. Glob. Optim.* 13, 455 (EGO);
Rasmussen & Williams (2006), *Gaussian Processes for Machine Learning*; Shahriari
et al. (2016), *Proc. IEEE* 104, 148 (review).

### 7.3 Multi-resolution (multi-fidelity) schedule
A coarse-cell stage explores cheaply; the best `seed_topk` points seed a fine-
cell stage. Because the unit-cube design is resolution-independent, seeds
transfer directly. Each evaluation warm-restarts the flow from the nearest prior
field (matching grid), cutting warm-up cost. Early termination stops a stage on
budget exhaustion, stalled relative improvement, or EI collapse. *Sources
(multi-fidelity BO):* Forrester, Sóbester & Keane (2007), *Proc. R. Soc. A* 463,
3251; Kandasamy et al. (2017), *ICML*.

---

## 8. Known approximations and limitations (for the reader's calibration)

These are documented so results are read with the right caveats (see
`VERIFICATION_AUDIT.md` for the full treatment):

1. **Sub-similarity Reynolds number** — the viscosity floor fixes Re_b ≈ 577 ≪ the
   Re-independence threshold; the flow regime is not dynamically similar to the
   real ABL (Snyder 1972).
2. **Under-resolved LES** — at 4 m, buildings are 3–8 cells tall; below the AIJ/COST
   guidance of ~10 cells/building. No objective grid-convergence study yet.
3. **Scalar normalization** — D3Q7 `c_s²` mismatch leaves an offset in absolute
   concentration; the scalar path is not yet validated against a dispersion
   benchmark (e.g. COST 732 / CODASC).
4. **GPU ABL inlet** — now wired on the GPU (`lbm_kernels.cu`: device inlet plane
   uploaded per step, per-cell read in `kern_flow`) but compiled/tested on the CPU
   backend only; verify the CUDA build compiles and shows inlet shear before
   trusting GPU results.
5. **Single scenario** — one wind direction / source in the current config;
   aggregation over a wind rose and source ensemble is required for a trustworthy
   optimized layout.
6. **Exposure weights** — placeholders in the lab driver; insert contaminant-specific
   exposure-weight values for absolute exposure.

---

## 9. Consolidated references

**LBM & turbulence** — Chen & Doolen (1998) *Annu. Rev. Fluid Mech.* 30, 329;
Succi (2001) *The Lattice Boltzmann Equation*; Krüger et al. (2017) *The Lattice
Boltzmann Method*; d'Humières (2002) *Phil. Trans. R. Soc. A* 360, 437;
Lallemand & Luo (2000) *Phys. Rev. E* 61, 6546; Nicoud & Ducros (1999) *Flow
Turbul. Combust.* 62, 183; Walsh, Burwinkle & Saar (2009) *Comput. Geosci.* 35,
1186.
**Inlet / ABL** — Richards & Hoxey (1993) *JWEIA* 46–47, 145; Hargreaves & Wright
(2007) *JWEIA* 95, 355; Blocken, Stathopoulos & Carmeliet (2007) *Atmos. Environ.*
41, 238; Kraichnan (1970) *Phys. Fluids* 13, 22; Smirnov, Shi & Celik (2001) *J.
Fluids Eng.* 123, 359; Panofsky & Dutton (1984) *Atmospheric Turbulence*; Stull
(1988) *Boundary Layer Meteorology*; Snyder (1972) similarity criteria (EPA).
**Domain / urban CFD** — Franke et al. (2007) COST 732 best-practice guideline;
Tominaga et al. (2008) *JWEIA* 96, 1749; Tominaga & Stathopoulos (2007) *Atmos.
Environ.* 41, 8091; Merlier, Jacob & Sagaut (2018) *Atmos. Environ.* 195, 89.
**Transport / numerics** — van der Sman & Ernst (2000) *JCP* 160, 766; van Leer
(1974) *JCP* 14, 361; Sweby (1984) *SIAM JNA* 21, 995.
**Aerosol / deposition** — Hinds (1999) *Aerosol Technology*; Seinfeld & Pandis
(2016); Zhang et al. (2001) *Atmos. Environ.* 35, 549; Slinn (1982) *Atmos.
Environ.* 16, 1785; Petroff & Zhang (2010) *GMD* 3, 753; Emerson et al. (2020)
*PNAS* 117, 26076.
**Aerosol / size distribution** — Hinds (1999) *Aerosol Technology*, 2nd ed.
**Exposure / occupancy** — Klepeis et al. (2001) *J. Expo. Anal. Environ.
Epidemiol.* 11(3), 231; Tsang & Klepeis (1996) EPA/600/R-96/148; Nazaroff (2004)
*Indoor Air* 14(s7), 175; Liu & Nazaroff (2001) *Atmos. Environ.* 35, 4451;
(2003) *Aerosol Sci. Technol.* 37, 565; Chen & Zhao (2011) *Atmos. Environ.* 45,
275; Lai & Nazaroff (2000) *J. Aerosol Sci.* 31, 463; Stephens & Siegel (2012)
*Indoor Air* 22, 501.
**Noise / geometry** — Perlin (2002) "Improving Noise," *ACM SIGGRAPH*; Ebert et
al. (2003) *Texturing & Modeling*.
**Optimization** — Jones, Schonlau & Welch (1998) *J. Glob. Optim.* 13, 455;
Rasmussen & Williams (2006) *GPML*; Shahriari et al. (2016) *Proc. IEEE* 104,
148; Forrester, Sóbester & Keane (2007) *Proc. R. Soc. A* 463, 3251; Kandasamy et
al. (2017) *ICML*.
