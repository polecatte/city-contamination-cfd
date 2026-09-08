# Validation test suite — airflow + forward contaminant

Two batteries: the **airflow** solver (against analytical solutions and measured
data, incl. the HRR cube) and the **forward contaminant** solver (no blow-up,
conservation laws, analytical Gaussian dispersion, experimental data). Each test
lists its reference, pass criterion, where it runs, and status.

Legend: [✓ verified here] runs & passes in-sandbox now · [lab] needs GPU/long run.

## A. Airflow battery  (driver: airflow_validation.cpp, orch: run_labverify.sh)

### Analytical (closed-form)
- **A1 Poiseuille channel** — laminar channel vs the analytical parabola; recovers ν
  from the profile. Pass: profile R²>0.999, ν error <2%. [lab]
- **A2 Blasius boundary layer** — flat-plate BL vs Blasius similarity solution.
  Pass: δ99 and profile within a few %. [lab]

### Measured-data comparison
- **T4 Cube reattachment Xr/H** — surface-mounted cube vs Silsoe (band 1.0–2.3).
  Uses the FIXED band-averaged reattachment detector (verified: finds a Xr/H=1.5
  synthetic bubble, ignores a laminar Xr/H=3.0 long bubble, rejects 1-cell noise).
  Pass: Xr/H in band. [lab — the low-ν HRR run]
- **T7 Cube Cp** — windward/leeward/roof/side pressure vs Silsoe bands. Pass: all
  four in band (last run: +0.69/−0.24/−0.40/−0.34, all in band). [lab]
- **T2 ABL homogeneity** — inlet→outlet log-law profile drift (Richards & Hoxey).
  Pass: <10% drift with WALL_MODEL=1. [lab — wall-model fix untested]
- **Inflow RFG turbulence** — synthetic inlet σu/σv/σw vs Panofsky–Dutton
  (2.5/1.9/1.25). **[✓ verified: within ~2% across all heights, correct anisotropy
  and height decay]**.

### Solver integrity
- **T1 mass / incompressibility** (per wind angle), **T3 WALE** (wake/channel ν_t
  ratio), **T5 lateral confinement** (Xr invariant vs clearance), **T6 Reynolds
  independence**, **T8 SHELL permeability**, **grid convergence (GCI)**,
  **determinism** (bitwise repeat), **low-ν stability probe** (HRR holds where reg
  diverges). [lab]

### HRR cube (the collision-operator physics test)
Run the cube/cubebench/wale modes at NU_FLOOR=5e-3 then 3e-3 (where reg diverged),
COLL=hrr, SPINUP_FT=3, MAX_WARMUP=60000. Pass: stable (finite), Xr/H→band, WALE
engaged (wake/channel≫1). [lab — the definitive HRR run, still pending]

## B. Forward-contaminant battery

### Analytical dispersion  (driver: plume_validation.cpp)   **[✓ ALL VERIFIED HERE]**
Production QUICK operator on a prescribed uniform laminar flow (scheme decoupled
from turbulence — the standard way to check a dispersion scheme):
- **PUFF advection** — point release centroid advects at U·t. **PASS (0.0%)**.
- **PUFF diffusion** — crosswind variance grows as 2·D·t (Gaussian spreading);
  confirms QUICK's numerical diffusion is negligible. **PASS (σ²=4.800 vs 4.800)**.
- **CONS mass** — interior mass conserved. **PASS (drift 2e-7)**.
- **STABLE (no blow-up)** — finite, bounded. **PASS** (overshoot 10% = QUICK
  Godunov non-monotonicity on the sharp delta, bounded & clipped; smoother pulse
  releases overshoot far less).
- **PLUME** — steady continuous source, crosswind σy²=2·D·x/U (Gaussian plume).
  **PASS (σy²=8.012 vs 8.000, 0.2%)**.

- **QUICK unphysical-value characterisation** (overshoot_test.cpp) — QUICK is
  non-monotone (Godunov), so it can undershoot below zero near sharp gradients.
  **[✓ verified: negatives are BOUNDED (worst-case sharp delta damps 7.9%→4.7%),
  mass-conserving, and contribute 1.3% of the dose worst-case, 0.007% for a
  realistic smooth pulse release]**. Not clipped in the linear dose: clipping
  max(0,Θ) is nonlinear and would break the source-superposition property, for a
  <0.01% gain — so the linear scheme is kept and negatives are negligible. Clip is
  reserved for FIELD VISUALISATION only (or a nonlinear threshold metric).

### Numerical-diffusion certification  (airflow_validation numdiff)
QUICK vs first-order upwind numerical diffusivity. Prior result: QUICK D_num≈0 vs
upwind ≈½UΔx — ~11× less diffusive. [✓ prior]

### Live-flow integrity  (driver: forward_live.cpp)
On the actual live turbulent path with the burst release:
- **Live stability** — burst does not blow up on turbulent flow; field finite. [lab]
- **Mass-budget closure** — burst: emitted = deposited + airborne + outflow, closes
  to <1% at 99% clearance. [lab — mechanism built, first run pending]
- **Ensemble sanity** — dose spread across seeds bounded and physical. [lab]

### Experimental  (driver: must_benchmark.cpp + must_score.cpp)
- **MUST dispersion** — Mock Urban Setting Test container array vs field data,
  Chang & Hanna (2004) scorecard: FAC2≥0.5, |FB|≤0.3, NMSE≤1.5, hit-rate≥0.66.
  [lab — needs the trial's receptor/observed data]

## How to run

- Forward analytical (fast, self-contained):  `bash run_forward_validation.sh`
  (builds + runs plume_validation; smoke-checks forward_live builds).
- Airflow battery (GPU):  `COLL=hrr NU_FLOOR=5e-3 bash launch.sh -- bash run_labverify.sh cube cubebench wale`
- MUST:  `./must_benchmark <trial> <dx> <steps> receptors.csv && ./must_score observed.csv must_predicted.csv`

## Initial-condition integrity (the "ran the wrong operator" guard)

Every solver-consumed setting resolves as: **explicit env overrides → else the
driver's value → else a safe struct default**; no setting silently overrides a
driver's intent (this fixed a live bug where driver-set HRR was reset to reg with no
COLLISION env). The collision banner prints on every run, and the production drivers
(forward_live, must_benchmark, exposure_solve) **hard-assert** the effective
collision/scalar matches intent after construction and abort loudly on a mismatch
(unless an env explains it). Verified: precedence unit-test passes all cases; all
drivers compile with the assertion.

## Status summary

Verified in-sandbox now: the inflow RFG turbulence and the entire forward analytical
dispersion battery (puff advection+diffusion, conservation, stability, Gaussian
plume). Pending on the lab GPU: the airflow battery (incl. the definitive low-ν HRR
cube), the live-flow burst integrity, and the MUST experimental comparison — all
have their drivers/analysis built and the analysis logic checked; they need the
hardware and, for MUST, the field data.
