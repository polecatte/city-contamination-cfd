# Inlet & Building-Regime Upgrade — Design + Implementation Note

This pass implements three audit follow-ups. Every physics/modeling decision below names its source; the same citations appear in the code comments.

---

## Change 1 — `target_density` removed from the optimizer search space (done)

Agreed scenario constraint, not a design variable. The audit measured population scaling ~5× (2,907→14,444) across the old 8,000–40,000 range, and collective exposure ∝ population, so the optimizer could minimize exposure by depopulating the city. `param_space.py` now fixes total population (`population_total`); the search space is **9-D**. A different population is a different goal → a new run.

---

## Change 2 — Sheared turbulent ABL inlet (implemented + validated, CPU)

Replaces the uniform plug-flow inlet (constant velocity at every height, zero turbulence) with an atmospheric-boundary-layer inlet.

### Mean profile — neutral-ABL log law
$$U(z) = \frac{u_*}{\kappa}\ln\!\frac{z-d+z_0}{z_0}, \qquad u_* = \frac{U_{ref}\,\kappa}{\ln\!\frac{z_{ref}-d+z_0}{z_0}}$$
**Source:** Richards & Hoxey (1993), *J. Wind Eng. Ind. Aerodyn.* 46–47, 145–153. von Kármán κ = 0.41; z₀ = aerodynamic roughness length (urban ≈ 0.5–1 m); d = displacement height. Maintaining this profile to the city face also needs consistent wall treatment — **Hargreaves & Wright (2007)**, *JWEIA* 95, 355–369; **Blocken, Stathopoulos & Carmeliet (2007)**, *Atmos. Environ.* 41, 238–252. The power-law form (**Davenport 1960; Wieringa 1992**) is an accepted alternative.

### Turbulence — Random Flow Generation (RFG)
A synthetic random-Fourier field, divergence-free by construction (each mode's amplitude vectors are built ⟂ to its wavevector), and a **closed-form function of (x, y, z, t)** — hence stateless, deterministic, and identical on CPU and GPU.
**Sources:** Kraichnan (1970), *Phys. Fluids* 13, 22–31 (random Fourier modes); **Smirnov, Shi & Celik (2001)**, *J. Fluids Eng.* 123, 359–371 (divergence-free anisotropic construction).
**Why RFG over alternatives:** the synthetic-eddy method (**Jarrin et al. 2006**, *Int. J. Heat Fluid Flow* 27, 585–593) and digital-filter methods (**Klein et al. 2003**, *J. Comput. Phys.* 186, 652–665; **Xie & Castro 2008**, *Flow Turbul. Combust.* 81, 449–470) require per-step eddy/field state — awkward for an LBM equilibrium inlet that is re-evaluated cell-by-cell every step. RFG needs none.

### Fluctuation amplitude — surface-layer similarity
σ_u : σ_v : σ_w ≈ 2.5 : 1.9 : 1.25 × u\* (**Panofsky & Dutton 1984**, *Atmospheric Turbulence*; **Stull 1988**, *An Introduction to Boundary Layer Meteorology*). Settable.

### Validation (`test_abl_inlet`, see `abl_inlet_validation.png`)
- Mean profile reproduces the analytic log law; U(z_ref) = U_inlet.
- Realized intensities: σ_u ≈ 1.26, σ_v ≈ 0.93, σ_w ≈ 0.63 m/s vs targets 1.26/0.96/0.63 — match.
- Residual **divergence 0.7%** of (σ/Δx) — essentially solenoidal (per-component gains are mild because each mode is individually divergence-free).
- Time series shows a turbulence-like spectrum rolling off near the −5/3 inertial slope.

### Solver integration (CPU done & tested; CUDA specified)
- `abl_inlet.h` — the generator (header-only, deterministic).
- `lbm_solver.h` — `Config` gains `inlet_profile` (0 = legacy uniform, default; 1 = ABL) + `abl_*` params (backward-compatible: zero-init keeps old behavior).
- `lbm_gpu.h` — `Impl` holds the `ABLInlet` + a host-filled `inlet_plane`.
- `lbm_solver.cpp` — builds the inlet at construction; `fill_inlet_plane()` refills the per-cell plane (in lattice units) each step, in both warm-up and release phases.
- `lbm_kernels_cpu.cpp` — inlet BC reads the per-cell plane when `inlet_profile==1`.
- `main_cpu.cpp` — enables it (z₀ = 0.7 m, z_ref = 40 m, L = 40 m, 100 modes).
- **Stability test:** empty channel, 2,400 steps with the inlet ON → max|u| finite (0.078 LU), no blow-up, flow stays unsteady (as intended for LES dispersion).
- **CUDA:** the device mirror is specified precisely in `lbm_kernels.cu` (device buffer + per-step `cudaMemcpy` of `inlet_plane` + identical read). **Not compiled/tested here**, consistent with the project's GPU policy — a one-function change.

### Honest caveat
The viscosity floor still fixes the effective building Re ≈ 577, below the Re-independence threshold (Re_H > ~1.1×10⁴; **Snyder 1972**). Injected turbulence therefore dissipates faster than physical. This inlet is a necessary correctness fix but does **not** resolve the Re-regime issue — that remains a separate work item.

---

## Change 3 — Alternatives to the permeable-building regime (model implemented + validated; solver rewire specified)

The permeable shell (partial bounce-back β = 0.005–0.02) lets the LBM momentum field flow *through* buildings — physically wrong for sealed structures, and an uncalibrated knob the optimizer can exploit. Porous-media treatment is justified for **vegetation only** (**Merlier, Jacob & Sagaut 2018**, *Atmos. Environ.* 195, 89–103).

### Options considered

| Option | Flow treatment | Indoor exposure | Grounding | Verdict |
|---|---|---|---|---|
| **A. Permeable shell (status quo)** | partial bounce-back through whole building | infiltrated scalar in INDOOR cells | porous precedent is vegetation, not buildings | uncalibrated, exploitable |
| **B. Solid building + infiltration model** *(recommended)* | full bounce-back (correct wakes/canyons) | algebraic F_inf × façade C_out | Liu & Nazaroff 2001; Chen & Zhao 2011; COST 732 | decoupled, calibrated, cheap |
| **C. Porous/Brinkman building** | Darcy–Forchheimer volume resistance | bulk through-flow | porous-media (vegetation) | still lets bulk flow through; wrong for sealed buildings |
| **D. Solid + facade flux coupling** | full bounce-back + thin indoor scalar layer fed by a façade transfer coefficient | CFD-resolved indoor box | mass-transfer + infiltration | most faithful, most work |

### Recommended: Option B — solid buildings + indoor infiltration model
With no indoor sources, the equilibrium indoor/outdoor ratio is the **infiltration factor**:
$$F_{inf} = \frac{P\,a}{a + k}$$
P = size-dependent envelope penetration; a = air-exchange rate (1/h); k = indoor deposition loss rate (1/h).
**Sources:** Liu & Nazaroff (2001), *Atmos. Environ.* 35, 4451–4462; Chen & Zhao (2011), *Atmos. Environ.* 45, 275–288. Penetration physics: Liu & Nazaroff (2003), *Aerosol Sci. Technol.* 37, 565–573; Stephens & Siegel (2012), *Indoor Air* 22, 501–513. Indoor deposition: Lai & Nazaroff (2000), *J. Aerosol Sci.* 31, 463–476. Keep parks porous (Merlier 2018).

### Implemented + validated (`infiltration.h`, `demo_infil`, `infiltration_model.png`)
- `penetration_factor(d_p)` peaks ≈ 0.95 at ~0.3 µm (accumulation mode), falling for ultrafine and coarse — matching Liu & Nazaroff (2003) / Stephens & Siegel (2012).
- Mass-distribution-integrated class values: **PM2.5 F_inf ≈ 0.46** (in the measured 0.3–0.82 band, Chen & Zhao 2011) and **PM10 ≈ 0.10** (coarse penetrates poorly — correct).
- **Surfaced caveat:** evaluating F_inf at a PM class's *cutoff* diameter (the project's `particle_diam = 10 µm` for PM10) badly under-predicts class infiltration, because the class mass sits well below the cutoff. Either integrate over the size distribution or use a mass-mean diameter (~0.3–0.5 µm for PM2.5).

### Remaining integration (specified, not yet wired — needs the exposure stage)
1. `voxelize.h`: add a `solid_buildings` mode tagging buildings `CELL_GROUND` (full bounce-back) while keeping parks porous; attach each building's occupant count + a façade-cell list instead of distributing inhabitance to INDOOR fluid cells.
2. Exposure stage (`exposure_objective.py`, not in this workspace): for each building, sample outdoor TIAC at the façade/roof and form indoor exposure = F_inf × (façade TIAC); deposited-surface exposure unchanged. This removes the permeable-shell knob and replaces it with the literature-grounded F_inf (calibratable, with explicit sensitivity to a and k).

---

## References (used in code and prose)
- Blocken, Stathopoulos & Carmeliet (2007). CFD simulation of the atmospheric boundary layer: wall-function problems. *Atmos. Environ.* 41, 238–252.
- Chen & Zhao (2011). Review of indoor–outdoor particles: I/O ratio, infiltration factor, penetration factor. *Atmos. Environ.* 45, 275–288.
- Davenport (1960); Wieringa (1992) — ABL power-law profile / roughness classification.
- Hargreaves & Wright (2007). On the use of the k–ε model in commercial CFD software to model the neutral ABL. *JWEIA* 95, 355–369.
- Jarrin, Benhamadouche, Laurence & Prosser (2006). A synthetic-eddy-method for inflow conditions for LES. *Int. J. Heat Fluid Flow* 27, 585–593.
- Klein, Sadiki & Janicka (2003). A digital filter based generation of inflow data. *J. Comput. Phys.* 186, 652–665.
- Kraichnan (1970). Diffusion by a random velocity field. *Phys. Fluids* 13, 22–31.
- Lai & Nazaroff (2000). Modeling indoor particle deposition. *J. Aerosol Sci.* 31, 463–476.
- Liu & Nazaroff (2001). Modeling pollutant penetration across building envelopes. *Atmos. Environ.* 35, 4451–4462.
- Liu & Nazaroff (2003). Particle penetration through building cracks. *Aerosol Sci. Technol.* 37, 565–573.
- Merlier, Jacob & Sagaut (2018). LBM-LES of pollutant dispersion in street canyons incl. tree planting. *Atmos. Environ.* 195, 89–103.
- Panofsky & Dutton (1984). *Atmospheric Turbulence*. Wiley.
- Richards & Hoxey (1993). Appropriate boundary conditions for CWE models using the k–ε model. *JWEIA* 46–47, 145–153.
- Smirnov, Shi & Celik (2001). Random flow generation technique for LES and particle-dynamics modeling. *J. Fluids Eng.* 123, 359–371.
- Snyder (1972/1981). Similarity criteria for the application of fluid models to dispersion studies (Re-independence threshold). EPA.
- Stephens & Siegel (2012). Penetration of ambient submicron particles into buildings. *Indoor Air* 22, 501–513.
- Stull (1988). *An Introduction to Boundary Layer Meteorology*. Kluwer.
- Xie & Castro (2008). Efficient generation of inflow conditions for LES of street-scale flows. *Flow Turbul. Combust.* 81, 449–470.
