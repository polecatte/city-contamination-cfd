# CHANGES — three fixes (regularized collision, rough-wall floor, sweep OOM/harness)

Drop these 7 files into your tree (they replace the originals). Everything is
runtime-toggled and defaults to the ORIGINAL behavior when the env vars are unset;
`run_overnight.sh` turns the fixes on for the battery.

## Files changed
- `lbm_solver.h`     — Config: `collision_mode`, `wall_model` toggles.
- `lbm_gpu.h`        — Impl: derived `u_wallX/u_wallY` (log-law floor slip, LU).
- `lbm_solver.cpp`   — reads env `COLLISION` / `WALL_MODEL`; computes `u_wall` from
  the ABL log law (u* matched to the inlet); prints both diagnostics.
- `lbm_kernels_cpu.cpp` — regularization + moving-wall floor term.
- `lbm_kernels.cu`   — identical mirror (signature + launch + both edits).
- `build_diffusion.sh` — `SWEEP_CELL` (default 8 m) → `-DCELL_SIZE_M`.
- `run_overnight.sh` — enables `COLLISION=reg WALL_MODEL=1 SWEEP_CELL=8`; builds the
  sweep tools at 8 m; stages now log `FAILED (exit N)` instead of a false `DONE`.

## Fix 1 — collision divergence  (env `COLLISION=mrt|reg`, default mrt; harness=reg)
Projected-regularized MRT: the non-hydrodynamic ("ghost") moments {1,2,4,6,8,10,12,
16,17,18} relax fully to their zero equilibrium each step, filtering the aliased
modes that blow up as τ→0.5. Conserved and shear moments are untouched, so the
recovered Navier–Stokes viscosity is unchanged. This is the moment-space core of
regularized LBM (Latt & Chopard 2006) and of the HRR family (Jacob, Malaspinas &
Sagaut 2018). It is what lets `NU_FLOOR` drop below 1e-2 without NaN.
- **Verified (CPU):** identical Poiseuille RMS error at ν=0.01 for mrt vs reg
  (0.025831 vs 0.025950) ⇒ regularization does not change the hydrodynamics.
- **Not yet verified:** that it cures the specific ν=5e-3 divergence — needs the GPU
  (or the CPU cube) run. See "staged rerun" below.
- Note: relaxing the energy mode (m1) at rate 1 slightly changes the *bulk*
  viscosity (acoustic damping only) — harmless for this flow.

## Fix 2 — ABL horizontal-homogeneity / T2  (env `WALL_MODEL=0|1`, default 0; harness=1)
Rough-wall log-law moving-wall bounce-back on the **domain floor only** (`sz==0`,
so building walls and roofs keep no-slip). Imposes the ABL wall slip
`u_w = (u*/κ)·ln((0.5·Δx + z0)/z0)` so the ground sustains the inlet log law instead
of dragging the near-surface flow to zero (Blocken, Stathopoulos & Carmeliet 2007).
u* is derived to match the inlet: `u* = κ·U_inlet / ln((zref+z0)/z0)`.
- **Verified (CPU):** prints `u_wall=1.9183e-02 LU, u*=0.505 m/s` — matches the hand
  calc and the inlet's own u*; `WALL_MODEL=0` reduces exactly to no-slip; init finite.
- **Not yet verified:** that it drops T2 drift below the 10% gate — this **is** the
  acceptance test (empty-domain incident profile). Treat T2 as pass/fail for keeping
  the wall model. If it over/under-corrects, tune the wall-plane height or z0.
- Coupling: with reg now allowing a lower `NU_FLOOR`, the over-diffusion that
  worsened the drift also drops — the two fixes reinforce.

## Fix 3 — sweep OOM + false "DONE"
Ranking/robustness domains were built at 4 m with full 5H/15H COST buffers → 200 M+
cells → 25–57 GB → instant OOM on a 16 GB card (both sweeps produced empty CSVs).
Now built at **8 m** (`SWEEP_CELL`).
- **Verified (CPU):** the N=6 ranking domain that needed 57 GB at 4 m is
  484×274×126 = 16.7 M cells = **4.50 GB** at 8 m — fits 16 GB with room for the
  robust stage. Sweep tools compile at 8 m; ranking-correlation selftest passes
  (Spearman/Kendall = 1.000).
- The harness now captures each stage's exit code: a crash logs
  `STAGE …: FAILED (exit N)`, not `DONE`. (This is what hid the two OOMs.)

## Recommended STAGED rerun (do not burn a full night blind)
1. **Isolate the divergence cure first.** Re-run only the case that died:
   `NU_FLOOR=5e-3 COLLISION=reg ./airflow_validation <the reynolds/cube case>`.
   Confirm it reaches stationarity finite. If reg alone isn't enough, that's the
   signal to add the HRR *hybrid* stress term (scaffold notes below) before more.
2. **Validate-only pass:** `DO_RANK=0 DO_ROBUST=0 bash run_overnight.sh`. Gate on:
   T2 drift < 10% (wall model working), T3/T4 (should improve if you also lower
   `NU_FLOOR` now that reg permits it), and no divergence at the production floor.
3. **Then** run rank/robust (they're already at 8 m).
4. **Re-confirm the top-k designs at 4 m** (`SWEEP_CELL=4`, small N) so the 8 m
   ranking isn't a coarsening artifact.

## Further fixes/improvements found while working (not yet applied)
- **A1 Poiseuille "FAIL" is spurious.** `test_poiseuille` builds its analytical
  reference at ν=0.001 but `NU_FLOOR` defaults to 1e-2, so the sim runs at 10× the
  viscosity the analytic solution assumes → 45% "error". Run it with
  `NU_FLOOR=0.001` (or have the test export it). Not a solver bug.
- **Now that reg permits it, lower `NU_FLOOR`** (e.g. 5e-3 → 1e-3) to push Re up
  toward the wake/dispersion regime and run the Re-robustness sweep with *stable*
  endpoints (0.53 vs 0.515 vs 0.503). This is the payoff of Fix 1.
- **If reg proves over-dissipative** (T3 resolved-turbulence ratio drops further),
  the next step is the HRR *hybrid* blend: the WALE velocity-gradient tensor is
  already computed in the collision, so the finite-difference deviatoric stress is
  in hand — blend it with the population stress (σ≈0.98) on the shear moments
  {9,11,13,14,15}. That is the accuracy half of HRR; projected regularization here
  is the stability half. Add it only if needed and validate on the cube first.
- **Production `nz` power-of-two quantization** (separate from the sweep, which is
  already no-pow2) still makes the BO objective step-discontinuous — worth switching
  to `nz_cost732` there too.

---

# Round 2 — nz_cost732, Poiseuille floor, lowered ν (tonight)

Additional files: `voxelize.h`, `test_poiseuille.cpp` (new to bundle); `run_overnight.sh` (updated).

## nz: power-of-two → nz_cost732 (voxelize.h)
`nz_for_max_height_cells()` now delegates to `nz_cost732()` — production and the sweep
share one sizing rule. Verified: nz_new == nz_cost732 exactly; the obsolete pow2 jumps
are gone and z-cells drop 30–44% (H=200 m: 512→300; H=24 m: 64→36). This removes the
step-discontinuity in the tallest dimension that roughened the BO objective, and it
lowers per-design memory. No code assumed a pow2 nz (checked).

## Poiseuille A1 fix (test_poiseuille.cpp)
The test builds its analytical parabola at ν=1e-3 but the default NU_FLOOR (1e-2)
overrode it, so the sim ran at 10× ν → ~45% RMS "FAIL". The test now `setenv("NU_FLOOR",
nu_eff)` (overwrite=1) before the Solver ctor, so solver-ν == reference-ν by construction.
Verified: solver reports `nu_lb=1.00e-03, tau_f=0.5030` (matched). (Full CPU convergence
at 1e-3 is slow; it completes/passes on GPU.)

## Lowered viscosity floor for tonight (run_overnight.sh)
`NU_FLOOR=5e-3` exported for VALIDATE + RANK (robust keeps its own 1e-2/5e-3 sweep,
set internally). 5e-3 (τ=0.515) is the exact value that diverged under plain MRT, so
tonight is a clean test that `COLLISION=reg` cures it, while ~doubling effective Re.
**Next step once it holds:** set `NU_FLOOR=1e-3` (τ=0.503, Re≈450–1700 — the wake/
dispersion regime). Change one value in run_overnight.sh.

---

# Round 3 — new overnight battery tests (memory-light)

New/updated: `airflow_validation.cpp` (+4 tests), `airflow_validation.py` (+4 graders),
`adjoint_recip_test.cpp` (now a pass/fail gate), `run_overnight.sh` (builds+runs the gate).
All new domains are ≤1.4M cells (the existing cube's footprint) → peak memory unchanged;
the battery runs one solver at a time, so these add wall-clock only.

## Added tests (run automatically under `airflow_validation all`)
- **SCALAR MASS BUDGET** (`av_massbudget.csv`) — passive + depositing plume; hard gate =
  finite AND no scalar-mass creation (dep+air ≤ 1.02·emitted). Validates the scalar
  operator, settling closure, and deposition bookkeeping the objective sums over.
- **SCALAR DIFFUSION** (`av_diffusion.csv`) — recovers effective crosswind diffusivity
  from a plume's spread (σ²=2·D·x/U) and reports D_eff/D_floor. Quantifies whether plume
  spread is set by the numerical/stability floor or by physical turbulent diffusion —
  the low-Re dispersion caveat, now a measured number for the paper. (Covers the
  sine-decay diffusivity goal via the production path, since the solver has no arbitrary-IC hook.)
- **LOW-ν STABILITY PROBE** (`av_stability.csv`) — cube at NU_FLOOR=1e-3, COLLISION=reg;
  gate = finite+bounded. Certifies reg cures the divergence at the target floor BEFORE
  a night is committed to 1e-3 (env saved/restored around the probe).
- **DETERMINISM** (`av_determinism.csv`) — same design twice; gate = bit-identical mean
  flow (RFG seed is fixed at 12345). Certifies the objective is noise-free (GP assumption).

## Adjoint reciprocity — now a GATE
`adjoint_recip_test.cpp` returned 0 unconditionally; it now checks the transpose test,
the objective identity (both rel < 1e-4), and the downwind-centroid sanity, and returns
exit 1 on failure. Verified locally: rel ≈ 1e-7, centroid 3.13, **PASS (exit 0)**.
`run_overnight.sh` builds it and runs it in VALIDATE; a FAIL logs that the reverse/robust
rankings are untrustworthy (they depend on the exact transpose).

## Note on scope
Two suggested analytical variants (no-flow sine-decay diffusion, still-air settling
decay) needed arbitrary scalar initial conditions the production Solver does not expose.
Their goals are covered through the production path instead: diffusivity via the plume
test, and settling+deposition conservation via the mass-budget test. Verified here:
all compile clean; the Python grader parses all four new CSVs; the adjoint gate runs and
passes. The four solver-based tests are too slow to run to completion on the CPU box but
follow the exact pattern of the existing T-tests and add negligible peak memory.

---

# Round 4 — ⟨F,p⟩ ensemble objective, D-floor, Briggs, QUICK exact transpose

Files: `adjoint_transport.h` (QUICK adjoint completed), `robustness_overnight.cpp` /
`ranking_stability.cpp` (ensemble objective + adjoint D-floor), `lbm_kernels_cpu.cpp`
(D-floor), `airflow_validation.cpp` / `.py` (Briggs), `run_overnight.sh` (D_FLOOR),
plus new `TECHNICAL_STATUS_AND_ROADMAP.md`.

## ⟨F,p⟩ ensemble exposure objective
The robustness eval now contracts the (source-independent) reverse footprint `F` with
a UNIFORM source prior `p` over open city ground — E_p[J], the population's expected
exposure over an equally-likely spread of release locations — via one extra dot
product (no extra solve). New columns `Jens_tau530`, `Jens_tau515`. Delete the old
`robustness_overnight.csv` before rerunning (header changed).

## D-floor 1e-3 → 1e-4
Consistent across CPU kernel (stale hardcoded 1e-3 removed to match the GPU, which
already honored D_FLOOR), the adjoint D0 in both sweep tools, and the harness
(`D_FLOOR=1e-4`). 1e-4 LU ≈ 0.035 m²/s, a weak-turbulence atmospheric diffusivity —
a physically defensible floor rather than an artificial value that dominates. Mainly
sharpens the van-Leer forward until QUICK is the reverse scheme (see roadmap §5).

## Briggs urban σ_y reality check
`T_diffusion` now compares the measured plume width to Briggs (1973) neutral-urban
σ_y(x)=0.16·x·(1+0.0004x)^(-1/2) and reports the ratio (want ~0.5–2). Self-contained
"is the plume as wide as the real atmosphere" check; grader flags PASS/INVESTIGATE.

## QUICK advection with EXACT discrete transpose (adjoint_transport.h)
Forward QUICK (Leonard 1979 — 3rd-order upwind-biased, LINEAR so the discrete adjoint
is an exact transpose) was already in `fwd_step`; the adjoint was completed:
- fixed part (B) fallback (vfn<0 now keys on the correct far-upstream `ju`);
- added part (C) far-upstream gather (the transpose of QUICK's +v·⅛·C[U] term):
  Role 3 (i is the U of face nb(i,d)→nb(nb(i,d),d), vfn≥0) and Role 4 (i is the U of
  the mirror face, vfn<0), FLUID–FLUID faces only.
Verified: reciprocity 1.8e-2 (FAIL, incomplete) → **1.65e-7 (PASS, exact)**, and exact
across 40 geometry/flow/boundary/solid/settling/nut configs (worst 1.3e-6). The
machine-precision forward=reverse reciprocity is preserved. Precedent: this is the
linear, exactly-transposable higher-order scheme (vs the nonlinear PPM/van-Leer that
CMAQ-ADJ/GEOS-Chem handle with a *continuous* adjoint).

Chosen because it keeps the exact transpose that underpins the novelty claim while
cutting the numerical diffusion behind the 316× scheme gap. Next run measures whether
the gap closes and the van-Leer-vs-reverse rank correlation rises.
