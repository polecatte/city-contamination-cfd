# Polydisperse Particle Size Distribution — Design + Implementation Note

Replaces the monodisperse single-diameter assumption with a realistic
mass-weighted size distribution, size-resolved settling and deposition, and
multi-bin transport. Every physics decision cites its source (prose and code).

## Why this matters for the exposure objective

The scenario is a contaminant release; exposure = airborne inhalation +
deposited-surface exposure. Airborne exposure scales with the
**airborne** time-integrated concentration; deposited-surface exposure with **deposited**
activity. Particle size sets how activity splits between these: coarse particles
settle and deposit near the source (deposited-surface exposure), fine particles stay airborne
and travel downwind (airborne inhalation). A single diameter collapses that
split and lets the optimizer tune a layout against one footprint. Even within a
modest distribution (MMAD = 1 µm, GSD = 2) the per-bin **settling velocity spans
≈500×** and the deposition velocity varies ≈5× — so monodispersity is a real
fidelity loss, not a rounding error.

## What was implemented

**1. Activity size distribution (`psd.h`).** Lognormal activity distribution
parameterized by MMAD + GSD, discretized into N sections (sectional method;
section mean diameter per bin), returning per-bin mass fraction and a
representative diameter. Aerodynamic→physical conversion `d_phys = d_ae·√(ρ0/ρ_p)`
preserves settling velocity (Hinds 1999).
**Sources:** lognormal aerosol size distributions and aerodynamic diameter — Hinds (1999), *Aerosol Technology*, 2nd ed. Typical defaults: MMAD = 1 µm, GSD ≈ 2; measured accidental-release distributions span ≈ 0.6 µm (fine plumes) to ≈ 6 µm (coarse resuspension).

**2. Size-dependent dry deposition (`deposition.h`).** Replaces the constant
per-surface `DEP_*` with the resistance model `v_d = v_g + v_ds`, where the
surface capture `v_ds = ε₀·u\*·(E_B + E_IM + E_IN)·R1` sums Brownian diffusion,
impaction, interception, and rebound. Yields the canonical V-shaped v_d(d_p)
with a minimum in the accumulation mode.
**Sources:** Zhang et al. (2001), *Atmos. Environ.* 35, 549–560, on the Slinn (1982) framework; minimum-location and refinements: Petroff & Zhang (2010), *Geosci. Model Dev.* 3, 753; Emerson et al. (2020), *PNAS* 117, 26076. Per-surface collectors map PARK → vegetation (small A → strong interception/impaction) vs buildings → urban/smooth, which **derives** `DEP_PARK ≫ DEP_BIZ` from physics.
**Finding:** the physical park deposition is ~0.1–0.4 cm/s, **10–30× lower** than the old hand-set `DEP_PARK = 3 cm/s`. That constant was an uncalibrated overestimate of "parks scrub more" — precisely the knob the audit warned the optimizer would exploit (over-greening to scrub the plume). The Zhang model removes it.

**3. Multi-bin transport (`polydisperse_demo.cpp`, Option A — implemented).**
Warm the flow once, save a checkpoint, then transport each activity bin on the
**same deterministic flow** (warm-restart reuse) with that bin's settling and
size-dependent deposition; accumulate mass-weighted fields:
- deposited-surface exposure ∝ Σ_i f_i · deposition_i
- airborne inhalation ∝ Σ_i f_i · TIAC_i

No kernel change. Cost ≈ 1 warm-up + N releases. The deterministic RFG inlet
replays identically per bin, so all bins see the same turbulent realization.
**Demonstration (real run, 2 bins):** the 9.8 µm bin deposits ~2.3× more than
the 0.92 µm bin (0.47% vs 0.20% of emitted) — correct ordering. Contrast is
small only because the demo domain/release are deliberately tiny; the machinery
and the size ordering are what's validated.

**4. `param_space.py`.** Added `source_amad`, `source_gsd`, `n_size_bins` to
`FIXED` as scenario constants (like the fixed population, they are not optimizer
design variables); `particle_diam` retained as the single-size fallback.

## Production transport path (Option B — scoped, recommended for the BO)

Option A re-evolves the flow per bin (cheap warm-up reuse, but N release passes).
For the optimizer's many evaluations, transport all N bins **simultaneously** on
one flow evolution:
- carry N scalar fields `g[N][2]` (and N `cmac`), one settling velocity per bin, and an N-indexed per-cell deposition (the surface type is fixed; only v_ds(d_p) changes, so store N values per surface cell or recompute per bin id);
- one source split by mass fraction across bins;
- accumulate per-bin deposition and TIAC; weight at the exposure stage.
Cost ≈ 1 warm-up + 1 release, with N× the scalar work (flow solved once). This is
a kernel change (CPU + the CUDA mirror), the analogue of the existing single-scalar
loop. Memory grows by ~N× the scalar arrays only.

## Remaining refinement at the exposure stage
Inhaled exposure per unit airborne concentration is itself **size-dependent** via
the respiratory deposition fraction: fine particles penetrate deeper. For full fidelity, weight each bin's TIAC by a size-dependent
inhalation exposure weight, not mass fraction alone. This belongs in
`exposure_objective.py` (not in this workspace) and is a per-bin scalar multiplier.

## References (this change)
- Hinds (1999). *Aerosol Technology*, 2nd ed. — lognormal size distributions, aerodynamic diameter, MMAD.
- Zhang, Gong, Padro & Barrie (2001). A size-segregated particle dry deposition scheme. *Atmos. Environ.* 35, 549–560.
- Slinn (1982). Predictions for particle deposition to vegetative canopies. *Atmos. Environ.* 16, 1785–1794.
- Petroff & Zhang (2010). Analytical aerosol dry deposition model. *Geosci. Model Dev.* 3, 753–769.
- Emerson et al. (2020). Revisiting particle dry deposition. *PNAS* 117, 26076–26082.
