# Verification Audit — Urban LBM Layout-Optimization Pipeline

**Scope.** Adversarial review of the chain *params → city layout → airflow → contaminant transport → exposure → minimized-by-optimizer*, against the premise that a Bayesian optimizer will exploit any unphysical behaviour or numerical artifact in any link. Three questions:

1. Does the city builder behave reasonably at all extremes, with continuous changes?
2. Have any system-wide variables been overlooked or under-considered?
3. Can every critical decision-point be grounded in published precedent?

**Method.** (a) Instrumented the real `city_builder7.h` + `voxelize.h` and swept all 12 optimizer parameters across their full `param_space.py` ranges (80 samples each), plus 402 corner/random samples over the whole 12-D space, measuring every objective-relevant output. (b) Read the solver/voxelizer/objective code for the physics chain. (c) Cross-checked each modeling choice against the urban-CFD / LBM-dispersion literature. **Not done here:** a full-solver exposure sweep and a grid-convergence study of the objective (flagged as the top recommended next step; the CPU solver is too slow for many runs in this environment).

---

## Verdict at a glance

| Question | Verdict |
|---|---|
| **1. Builder reasonable & continuous at extremes** | **Robust but not smooth.** No crashes, empty domains, or degenerate geometry in 402 extreme samples. But there are genuine staircase discontinuities (block sizing → block count; vertical domain `nz` in powers of two), large dead zones (height saturates, most layout knobs barely move population), and a one-sided density controller that under-shoots by up to **−72%** at high density. |
| **2. Overlooked system-wide variables** | **Yes — several, two of them decisive.** (i) *[RESOLVED v8]* `target_density` **was** an optimizer variable that scaled population ≈5× (collective exposure ∝ population, so exposure was minimizable by depopulating the city); it has since been **removed** — population is now a fixed scenario constant. (ii) The live optimization runs **one fixed wind direction and one fixed, known source** — the layout will overfit to that single scenario. Plus: no ABL inlet profile, no inflow turbulence, sub-similarity Reynolds number, under-resolved LES, an unvalidated scalar/deposition path, and no atmospheric stability. |
| **3. Precedent at all critical decision-points** | **Mostly yes; three weak points.** LBM-LES + MRT + WALE + COST domain sizing + Stokes–Cunningham settling + porous-media trees are all well-precedented. **Not** well-grounded: the uniform laminar inlet, the viscosity-floor "wind-speed-doesn't-matter" justification (effective Re is below the Re-independence threshold), and permeable *building* envelopes. |

The headline: **the builder is not where the optimizer will break the physics — the objective definition and the flow regime are.** The most exploitable failures are upstream (depopulation, single-scenario overfit) and downstream (unvalidated scalar/deposition), not in the geometry generator.

---

## Part 1 — Continuity and behaviour at extremes

### 1.1 Robustness (good news)
Across 402 corner + random samples spanning the full 12-D cube: `num_blocks` ∈ [54, 180] (never zero), voxel height `nz` ∈ [32, 256] (always a real fluid column, never collapses to a slab), `fluid_frac` ∈ [0.96, 0.99], `park_frac` ∈ [0.11, 0.60] (respects the 5–60% clamps). No exceptions, no NaNs, no empty cities. The generator degrades gracefully; an optimizer cannot drive it into a geometrically undefined state.

### 1.2 Discontinuities (the GP surrogate's blind spot)
The Bayesian optimizer's Gaussian-process surrogate assumes a smooth response. Two sources violate that:

- **Block-count steps.** Sweeping `block_w` 24→72 m produces **6 discrete jumps** in block count (72→144 blocks); `block_d` 16→40 m produces **5**. Each step instantaneously re-runs the two-pass business assignment and re-tiles zoning, so geometry is piecewise-constant in these axes — a staircase, not a ramp.
- **Vertical-domain quantization.** `nz` is the tallest building height → ×5 headroom → next power of two. Across the space it jumps 32→64→128→256. This is simultaneously an **up-to-8× compute swing** and a **discontinuous change in the physical domain** (the free-slip lid height moves in steps), driven by layout parameters (block dimensions, coverage) that set the tallest building.

Neither is catastrophic — aggregate metrics like population stay bounded because of the density controller — but the EI acquisition can stall on a step edge or waste budget exploring a flat tread. A monotone reparameterization or a discreteness-aware kernel would help.

### 1.3 Dead zones (no gradient → optimizer wanders)
- **Height saturates.** `cbd_peak` spans 20→120 m but realized `max_height` only moves 28→32 m and is **flat above cbd_peak ≈ 30 m** — the `SLENDERNESS` clamp (max height = 7 × min block dimension) caps it. So **~75% of the `cbd_peak` range produces an identical city.** `base_height` is similarly near-inert. Two of twelve search dimensions are mostly wasted.
- **Layout barely moves population.** Total inhabitance varies <2% across the full range of *every* layout parameter (e.g. `coverage` moves it by ~10 people out of ~7,130; `park_centrality` by ~12). A closed-loop density controller ("nudges") pins population to target regardless of layout. This is good (it closes the per-layout depopulation loophole) but means most of the objective's population term is decided before airflow enters.

### 1.4 The density controller is one-sided and saturates
`pop_err` ranges from **−71.8%** to **+3.5%** across the sample set, with a strong negative bias. *[Historical — the `target_density` controller was removed in v8; total population is now fixed, so this surface is no longer in play.]* The old controller could nudge population *down* but not exceed the geometric capacity ceiling (height clamp + block count), so at high target density realized population fell far below target — a saturating, nonlinear surface the optimizer could exploit for reasons unrelated to dispersion.

*(See `builder_continuity_sweeps.png` for the full 12-parameter response panel.)*

---

## Part 2 — Overlooked / under-considered system-wide variables

Ranked by how readily an optimizer exploits them.

### 🔴 Critical

**1. `target_density` was an optimizer variable → collective-exposure reward hack. [RESOLVED v8]**
Population scaled ≈5× (2,907 → 14,444) as `target_density` went 8,000 → 40,000, and collective exposure ∝ population, so the optimizer could have cut the objective ~5× by pure depopulation. **This has been closed:** `target_density` was removed from the search space and total population is now a fixed scenario constant (designs are compared at equal headcount, total exposure Σ w·C).

**2. Single fixed wind direction + single fixed, known source.**
`param_space.FIXED` pins `wind_angle = 0` and one source location, yet the project's own physics notes say wind *direction* is "the meaningful optimization variable," and `OBJECTIVE_SPEC` calls for minimax / wind-rose aggregation over directions — "NOT yet defined." Optimizing a layout against one direction and one known source invites textbook overfitting: the optimizer will sculpt directional asymmetries (channels/baffles) that divert *this* plume from population and fail for any other wind or source. **Fix: aggregate over a wind rose (worst-case minimax is the conservative choice for a hazardous release) and over a source ensemble, before trusting any optimized layout.**

### 🟠 Major (physics fidelity the optimizer inherits)

**3. No atmospheric boundary-layer inlet; no inflow turbulence.** The inlet writes a single uniform velocity at every height (plug flow). Real urban dispersion requires a vertically-sheared mean profile (log/power-law) with consistent turbulence, and LES specifically needs time-dependent synthetic turbulence at the inlet (vortex/SEM methods). With a steady uniform inlet the WALE LES has no resolved upstream turbulence to act on — turbulence exists only in building wakes.

**4. Sub-similarity Reynolds number; "wind speed doesn't matter" is not justified.** The viscosity floor fixes the effective building Reynolds number at **≈ 577** (project's own figure), while the real value is **≈ 1.1×10⁷** (32 m building, 5 m/s). The Re-independence regime that *would* license ignoring wind speed begins around **Re_H > 11,000** (Snyder), or ~4,000 even for a small cube — 7–20× above 577. So the flow is in a low-Re/transitional regime that is neither physical nor Re-independent; wakes, separation, and canyon vortices won't match reality, and the optimizer tunes geometry to a non-physical flow. (`U_inlet` is consequently a dead fixed parameter too.)

**5. Under-resolved LES.** At 4 m cells a 12–32 m building is only **3–8 cells tall**; AIJ/COST guidance is on the order of ten cells across each building dimension. The vertical resolution is inadequate for LES of wake/canyon turbulence, and **no grid-convergence study of the exposure objective exists** (the only convergence evidence is the Poiseuille flow test). The objective could shift qualitatively at 2 m.

**6. The scalar/exposure path is the least-validated link and the exposure rides entirely on it.** Acknowledged but worth restating as system risk: the D3Q7 scalar hard-codes c_s²=1/3 against true weights of 1/4, leaving a normalization offset in absolute concentration; a small negative undershoot remains post-TVD. Only the *flow* is validated (Poiseuille); the scalar field has never been checked against a dispersion benchmark (e.g. COST 732 / CODASC metrics: hit rate, FAC2).

**7. Deposition velocities are uncalibrated and exploitable.** Per-surface `dep_vel` values are size-independent, hand-set, "meant to be tuned," with `DEP_PARK` (0.030) ≈ 30× `DEP_BIZ` (0.001). Deposited-surface exposure ∝ deposition, so the optimizer will ring the source/city in parks to "scrub" the plume — a result driven by an arbitrary constant, not calibrated physics.

### 🟡 Worth fixing / documenting

- **Permeable building envelopes.** Buildings use partial-bounce-back β = 0.005–0.02, so wind and scalar infiltrate them and "indoor" cells carry occupants → indoor exposure. Vegetation-as-porous-media is standard; *buildings* are normally solid. The infiltration values are unvalidated, and indoor sheltering is another uncalibrated knob the optimizer can lean on.
- **No atmospheric stability.** Isothermal/neutral only — no buoyancy or Monin–Obukhov stability. Ground-level concentrations are strongly stability-dependent; neutral-only is an unstated single-scenario assumption.
- **Contaminant chain gaps.** No explicit chemical transformation / decay, plume rise, or wet scavenging in the transport (decay may be folded into exposure weights — unclear and should be documented).
- **Buffers below best practice in the BO config.** `param_space.FIXED` uses 120 m lateral (≈3.75H) and 200 m downstream (≈6H) for a 32 m city, below COST's ≥5H lateral and ≪ its ~15H downstream wake-recovery recommendation. The outlet then sits in a still-recovering wake. (`default_buffers()` at 200 m uniform is compliant; the speed-trimmed BO buffers are not.)
- **"roughness" is a misnomer.** The parameter is building-height jitter, not aerodynamic surface roughness z₀. The quantity that actually governs the ABL — ground roughness — is absent from the model entirely.
- **Time/averaging.** `release_time` = 300 s vs ~200 s domain transit (1,000 m / 5 m/s) ≈ 1.5 flow-throughs; the TIAC may be near-source-biased and not quasi-steady, and the warm-up steadiness test (RMS<2e-3) may end before large-scale wake unsteadiness develops.
- **Optimizer plumbing.** Staircase discontinuities (§1.2) violate the GP smoothness prior; `PENALTY = 1e9` for failed evaluations can distort GP normalization (already flagged).

---

## Part 3 — Precedent at each critical decision-point

| Decision-point | Choice in code | Published precedent | Verdict |
|---|---|---|---|
| Solver family | D3Q19 LBM-LES for urban dispersion | Merlier, Jacob & Sagaut 2018 (Atmos. Environ.); ProLB; OpenLB urban DTs; Athens canyon LBM | ✅ established |
| Collision | MRT in moment space | d'Humières 2002; Lallemand & Luo 2000 | ✅ established |
| SGS turbulence | WALE | Nicoud & Ducros 1999; standard in canyon LES | ✅ model OK, but under-resolved (§2.5) |
| Domain sizing | 5×40 m buffers, 5H top headroom, blockage low | COST 732 / Franke et al. 2007; Tominaga et al. 2008 (blockage <3%) | ✅ for `default_buffers`; ⚠️ BO-trimmed buffers below 5H lateral / 15H downstream |
| Trees/parks | High-permeability porous canopy (β=0.8) | Porous-media vegetation in LBM canyon LES (Merlier 2018) | ✅ established |
| **Building envelope** | Partial-bounce-back permeable shell (β=0.005–0.02) | Buildings normally solid bounce-back; porous precedent is vegetation only | ❌ non-standard / unvalidated |
| **Inlet BC** | Uniform steady plug flow | Best practice = sheared profile + time-dependent synthetic turbulence (vortex/SEM); COST/AIJ | ❌ against best practice |
| **Wind-speed independence** | Viscosity floor → ignore U; "Re-independent" | Re-independence needs Re_H ≳ 11,000 (Snyder); sim Re≈577 | ❌ claim unsupported below threshold |
| Settling | Stokes + Cunningham slip | Textbook aerosol physics (Hinds; Seinfeld & Pandis) | ✅ established |
| Surface deposition | Per-surface deposition velocity | Resistance/deposition-velocity models (Slinn; Sehmel; Zhang et al. 2001) | ⚠️ concept OK, values uncalibrated |
| Exposure metric | Airborne + deposited-surface exposure; minimax/wind-rose | 2-pathway (airborne/deposited) decomposition is standard aerosol-exposure practice | ✅ framework standard; ⚠️ aggregation not yet implemented |
| Optimizer | Multi-resolution Bayesian opt + warm restart | BO / multi-fidelity BO; ML surrogates for urban flow (e.g. UrbanFlow-3K) | ✅ method sound; objective design is the weak link |

---

## Prioritized recommendations

1. **[DONE v8] Removed `target_density` from the search space** — total population is now fixed. Closed the ≈5× depopulation hack.
2. **Implement the wind-rose / minimax aggregation and a source ensemble before optimizing.** Single-scenario layouts are not trustworthy; this is the difference between a valid study and an overfit one.
3. **Validate the scalar field** against a dispersion benchmark (COST 732 / CODASC) with COST metrics, and fix the D3Q7 c_s² normalization. Until then, treat absolute exposure as indicative only.
4. **Grid-convergence study of the objective** at 8/4/2 m; report whether the optimal layout is resolution-stable.
5. **Upgrade the inlet** to a sheared ABL profile + synthetic turbulence; reassess whether the viscosity floor is acceptable or whether a higher resolution / different closure is needed to reach a representative Re regime.
6. **Calibrate deposition velocities** (size-dependent) and **document the permeable-building infiltration** assumption with a reference or a sensitivity study; both are currently free knobs.
7. **Restore COST-compliant buffers** for production runs (≥5H lateral, ≥15H downstream) or quantify the error from the trimmed ones.
8. **Mitigate the staircase discontinuities** (discreteness-aware kernel or reparameterization) and retire the near-inert dimensions (`cbd_peak`, `base_height`) or widen blocks so height is no longer clamped.

## Caveats of this audit
Part 1 is empirical against the real builder/voxelizer. Parts 2–3 are code-reading + literature cross-checks; I did **not** run the full LBM solver across the parameter space here, so claims about how the *flow/scalar* respond at extremes are inferred from code and the project's own validation notes, not re-measured. Recommendation 4 (objective grid-convergence) is the most important thing this audit could not itself perform.
