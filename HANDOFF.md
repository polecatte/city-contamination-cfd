# HANDOFF — Urban LBM Dispersion-Optimization

> ⚠ **OUT OF DATE — this file predates the OpenLB migration.** It describes the project at
> v8.3, with the custom solver as *the* path and a 9-D search space, and it does not mention
> OpenLB anywhere. It also cites four files that are not in this bundle (`bayesopt.py`,
> `evaluate.cpp`, `evaluate.py`, `OVERNIGHT_DIAGNOSTIC_NOTE.md`) and several that moved into
> `tests/`. **Start at `README.md`**, then `OPENLB_PORT_STATUS_AND_VERIFICATION.md` for the
> port's real state and `CLEANUP.md` for where else the docs have drifted from the code.
> What remains reliable here: the builder description (§4), the search-space reasoning (§5),
> and the design-philosophy note on keeping population emergent.


This was the single entry point for picking the project up cold (see the notice above —
that role now belongs to `README.md`). It summarizes what
the project is, how it's structured, how to build/run it, what has been verified vs.
what is still untested, the known bugs/open questions, and where to go next. Deeper
detail lives in `ARCHITECTURE.md` (math + structure) and the topic `*_NOTE.md` files
(catalogued at the end).

---

## 1. What this is
A pipeline that **optimizes city layout to minimize population exposure to an
airborne contaminant**. A parametric city builder turns ~9 design knobs into a 3-D
urban geometry; a Lattice-Boltzmann solver computes the mean wind field; a scalar
transport model disperses a ground-level release through it; an exposure model
weights concentration by where people actually are (time-budget occupancy, indoor
infiltration); and a Bayesian optimizer searches the design space.

The objective is **linear in concentration**: `J = Σ_x w(x)·C(x)`, where `w` is the
effective inhabitance (people present, 24-h averaged). Linearity is deliberate — it
lets the reverse/adjoint solver evaluate all source locations at once.

**Design philosophy (important):** an optimizer will exploit any unphysical freedom
in the map. So scenario variables (total population, wind, source location, particle
sizes) are FIXED outside the search space, and population is kept **emergent from
built form** — there is no free "population" field the optimizer could teleport into
a low-concentration corner. Only *layout* is searched.

---

## 2. Current status (v8.3)
- **Builder:** stable and heavily exercised. Current usage classes are PARK /
  BUSINESS / RES_HIGH / RES_LOW only.
- **CPU flow/transport path:** compiles and runs; used for all validation here.
- **GPU (CUDA) path:** `nvcc` is NOT available in this dev container, so **all CUDA
  is UNTESTED on-device**. The CPU kernel (`lbm_kernels_cpu.cpp`) is a line-for-line
  mirror of the CUDA kernel (`lbm_kernels.cu`); compiling the CPU path with
  `-DWITH_LBM` catches source errors, and the two were verified to mirror on the
  recently-changed regions (free-slip lateral BC, `rho_sum` density accessor + its
  `n_avg` normalization). **The next overnight GPU run is the real test.**
- **Invariants + acceptance checks:** pass (population, employment, block accounting).

The user runs heavy sims on their own lab box (RTX A4000, 16 GB, CUDA 12.2). `nvcc`
needs `g++-10` via `CCBIN` there (the box's default g++-11 is too new).

---

## 3. Program structure & data flow
```
 x ∈ [0,1]^9 ──to_physical──▶ city builder ──▶ voxelizer ──▶ LBM flow (D3Q19 MRT+WALE)
 (param_space.py)             (city_builder7.h) (voxelize.h)  (lbm_solver.cpp,
                                                               lbm_kernels_{cpu.cpp,.cu})
        ┌──────────────────────────────────────────────────────────┘
        ▼
 scalar transport (D3Q7 ADE + settling + deposition) ──▶ exposure model
 (lbm_kernels_*, psd.h, deposition.h)                     (occupancy.h, infiltration.h,
        │                                                  indoor_exposure.h)
        ▼
 objective J = Σ w·C ──▶ Bayesian optimizer (bayesopt.py*, run_optimization.py)
 (reverse/adjoint: adjoint_transport.h / adjoint_transport_gpu.cu)
```
`*bayesopt.py` is part of the optimizer core and lives outside this bundle (see §9).

**Key files**
- `city_builder7.h` — the parametric builder (the most-edited file; §4).
- `param_space.py` — THE single definition of the search space + scenario constants.
- `voxelize.h`, `occupancy.h`, `deposition.h`, `infiltration.h`, `psd.h` — geometry
  → grid, time-budget occupancy, dry deposition, indoor infiltration, particle sizes.
- `lbm_solver.{h,cpp}` — solver driver incl. the Phase-A time-averaging to
  stationarity (§6).
- `lbm_gpu.h` — the `Solver::Impl` struct, `gpu::` decls, and the memory guard.
- `lbm_kernels.cu` / `lbm_kernels_cpu.cpp` — the LBM + transport kernels (keep these
  two LINE-IDENTICAL in the physics; edit both together).
- `airflow_validation.cpp` (+`.py`) — the physics validation suite (§7).
- `run_overnight.sh` — the master overnight runner (build → smoke → validate → sweeps
  → analysis), detached with nohup.
- `ranking_stability.cpp`, `robustness_overnight.cpp` — the design-ranking sweeps.
- `sensitivity.{cpp,py}` — first-order Sobol screening of the builder knobs.

---

## 4. The city builder (v8.3) — what actually happens
A strictly **circular ring/central-core** model. See `ARCHITECTURE.md §1` for the
math; the essentials:

- **Zoning:** business fills from the lowest business-metric block outward (a solid
  central circle). `patchiness` ∈ [0,1] blends that metric toward a coherent
  value-noise field, spreading business into off-centre patches (0 = concentric
  core, 1 = coherent districts). Non-business blocks default to RES_HIGH.
- **Heights (pure morphology):** `h = BASE_HEIGHT_M(9 m, fixed) + cbd_peak·exp(−cbd_decay·r²)`,
  then a **log-normal** per-block multiplier `exp(√12·roughness·ξ)` so σ(ln h) ≈
  roughness ≈ height CoV; clamped by slenderness `7·min(block_w,block_d)`.
- **RES_LOW (low-density housing):** a residential block resolving to ≤4 floors
  becomes RES_LOW and is shrunk to **25 % lot coverage** (house + yard). It's
  height-driven, so it appears at the periphery of modest/sharp-CBD cities and under
  strong roughness, and is (correctly) absent when a tall broad CBD makes the whole
  city high-density. The 9 m base was chosen specifically to sit just below the
  4-floor threshold so a low-density ring emerges.
- **Footprint:** buildings fill their lot to a fixed 4 m setback (`cov ≡ 1`; the old
  global coverage knob is retired). In renders the gray is the lot; a large gray
  margin = a RES_LOW yard.
- **Parks:** placed by maximin dispersal + a smooth **radial** `park_centrality`
  (0 = edge ring, 0.5 = distributed, 1 = central cluster) with a position-hashed,
  axis-decorrelated tie-break. (The previous two-phase maximin+adjacency-clustering
  formed one-sided park WALLS biased by block aspect — fixed.)
- **Population/employment:** FIXED `population_total` allocated by residential
  floor-area capacity; workers = 0.47·population, business floor area balanced to
  hold them at ~14 m²/worker. Verified: pop and workers conserve across every
  parameter setting.

---

## 5. Search space (9-D) — see `PARAM_SPACE_NOTE.md` for full bound justifications
`block_w [24,72] m`, `block_d [16,40] m`, `cbd_peak [20,120] m`,
`cbd_decay [1e-6,3e-5] m⁻² (log)`, `patchiness [0,1]`, `park_centrality [0,1]`,
`park_fraction [0,0.40]`, `roughness [0,0.80]`, `street_width [8,40] m`.
`wind_direction [0,π/4]` is staged but **inactive** (lateral BC is a symmetry plane,
not an oblique inflow). Fixed/retired: `base_height` (fixed 9 m), `population_total`
(replaces `target_density`), `coverage` (→ street_width), `biz_inner_frac`
(→ patchiness), `mixed_frac`, `cbd_aspect/cbd_angle/biz_aspect` (circular model).

---

## 6. Build & run
Environment on the lab box: `nvcc` present, needs `CCBIN=g++-10`. In this dev
container `nvcc` is absent — compile the CPU path with `-DWITH_LBM` to check sources.

- GPU flow build: `bash build_gpu.sh` (env `ARCH=-arch=sm_86`, `CCBIN=g++-10`,
  `CUDA_LIB=/usr/local/cuda/lib64`) → `kernels.o + solver.o + lab_test`.
- Sweeps/transport build: `bash build_diffusion.sh gpu` → `ranking_stability`,
  `robustness_overnight`, `diffusion_compare`.
- Sweep args: `ranking_stability run N city_m rel warm` (def 6 512 60 4000);
  `robustness_overnight run N city_m rel warm` (def 48 768 120 10000).
- Free a stuck GPU: `pkill -9 -x lab_test`.
- **Overnight:** `bash run_overnight.sh` builds foreground (fail-fast) then runs
  detached: smoke → validation → ranking → robustness → analysis, `nohup`+`disown`,
  timestamped `overnight.log`. Recommended: run inside `tmux`; close the SSH session
  only AFTER the "detached PID" line prints; check `logind` `KillUserProcesses=no`.

Memory: ~289 B/cell (72 device floats/cell; single source of truth
`device_floats_per_cell()` in `lbm_gpu.h`). ~50 M-cell ceiling on 16 GB. The memory
guard runs at every `Solver` construction (`cudaMemGetInfo`/`sysinfo` pre-flight,
`exit(3)` if it won't fit). `INFLOW_H`/`OUTFLOW_H` env knobs trim the domain to fit.

---

## 7. Testing & validation status
Validation suite: `airflow_validation.cpp` subcommands
`mass/abl/wale/cube/lateral/reynolds/cp/shell/blasius/resolution/smoke/all`,
visualized by `viz_airflow.py` (5 paper figures). `run_overnight.sh` runs `smoke`
FIRST and aborts the battery if it fails.

**Passed (CPU):** Blasius flat-plate BL (δ99 ∝ x^0.5, ν-independent); WALE ν_t ≈ 0
in pure shear; divergence ≈ 0.04 %. Builder invariants + example acceptance checks.

**The `smoke` gate is the linchpin of the next GPU run** — it asserts `n_avg > 0`
(time-averaging actually engaged), density accessor works, flow finite. Watch
`overnight.log` for the `[smoke] PASS` line BEFORE trusting any cube/Cp/ABL numbers.

**Known caveats to re-check on properly-averaged means (last run's were instantaneous
snapshots and are mostly invalid):** cube wake reattachment length (was Xr/H ≈ 4.9,
too long); weak wake ν_t; ABL homogeneity (44–70 % drift was a snapshot artifact —
re-measure); Cp signs (were all-positive on snapshots + an untested density path).

---

## 8. KNOWN BUGS / OPEN ITEMS (the honest list)
1. **All CUDA is device-untested.** Highest risk. Verify via the overnight smoke gate.
2. **Mean-flow physics unverified on real time-averages** — see §7 caveats.
3. **Oblique wind inactive** — lateral BC is a symmetry plane, no y-fetch. Needs an
   oblique-inflow validation (mass closure, plume clear of +y boundary, sane mean at
   15/30/45°) and likely a larger `buf_yp` before `wind_direction` joins the search.
4. **Robustness sweep at tall Hmax** still needs `OUTFLOW_H` trimming to fit 16 GB
   (default 8 in `run_overnight.sh`).
5. **Weight-side under-parameterized vs C-side.** patchiness & RES_LOW reshape the
   city as intended but their leverage on J itself is unmeasured — needs a sensitivity
   on J after the flow model is validated.
6. **No objective grid-convergence study yet** (resolution test added to the suite;
   general guidance ~10 cells/building, CELL fixed at 4 m — user refuses coarser).
7. **Analytical LBM tests (Taylor-Green, Couette)** need periodic/moving-wall BCs
   not currently present — future work.

---

## 9. Not in this bundle (external / unchanged)
`bayesopt.py` and the optimizer core imported by `run_optimization.py` live outside
this bundle. `evaluate.py`/`evaluate.cpp` (the driver the optimizer calls) may also
be external. Anything importing `bayesopt` won't run here (e.g. `test_opt_logic.py`
imports it transitively); its layout-mapping asserts were fixed to the current params
and verified in isolation.

---

## 10. Recent changes (this session's changelog)
- Averaging bug fixed: `run()` now spins up then time-averages to statistical
  stationarity (was returning instantaneous snapshots; `n_avg` was 0).
- Density accessor (`rho_sum`) added for mean pressure/Cp (CUDA mirrored, untested).
- Free-slip lateral (±y) BC via specular reflection (was clamped self-pull that
  contaminated the boundary).
- Memory guard, wind-aware `minimal_domain`, smoke gate, resume for sweeps.
- Builder: `base_height` fixed at 9 m; `roughness` exaggerated to a log-normal
  height-CoV knob [0,0.8]; `patchiness` reintroduced as a business-spread knob;
  RES_LOW low-density housing reintroduced; **park placement axis-bias bug fixed**
  (dispersed maximin + radial centrality + symmetric tie-break).
- Code hygiene: stale comments removed (param_space base_height, builder step
  history, solver RMS-warmup print); defunct analysis scripts (`sweep_audit.py`,
  `exploit_scan.py`) and `test_opt_logic.py` repointed to the 9-D param set.

---

## 11. Consolidated sources
Numerics/turbulence: Krüger et al. (2017) *The Lattice Boltzmann Method*; Nicoud &
Ducros (1999) WALE, *Flow Turbul. Combust.* 62:183. ABL inlet: Richards & Hoxey
(1993); Kraichnan (1970) / Smirnov et al. (2001) RFG; Panofsky & Dutton (1984);
Stull (1988). Urban-CFD BCs & validation: Franke et al. (2007) COST-732; Tominaga et
al. (2008) *JWEIA* 96:1749; Richards, Hoxey & Short (2001) Silsoe cube. Morphology:
Xie, Coceal & Castro (2008) *BLM* 129:1; Nakayama, Takemi & Nagai (2011) *JAMC*
50:1692; Oke (1988) canyon H/W. Occupancy/exposure: Klepeis et al. (2001) NHAPS,
*JEAEE* 11:231. Deposition/porous media: standard resistance model; Walsh, Burwinkle
& Saar (2009) *Comput. Geosci.* 35:1186. Grid convergence: Roache (1997) / ASME V&V
20. Noise: Perlin (1985; 2002); Ebert et al. (2003). Density/area standards: U.S.
GSA P100 (2024), AHS (2023), U.S. Census (2024), BLS CPS. Full context in
`ARCHITECTURE.md §9` and the relevant `*_NOTE.md`.

---

## 12. Documentation map (which files are current)
- `ARCHITECTURE.md` — math + structure. **Updated this session** (v8.3).
- `PARAM_SPACE_NOTE.md` — the 9-D search space + per-bound justifications. **Current.**
- `README.md` — quick start + file manifest. **Updated this session.**
- `AIRFLOW_VALIDATION_NOTE.md` — the validation suite + what each test grades.
- `VERIFICATION_AUDIT.md` — continuity/extremes audit of the builder (some param
  names predate v8.3 but the findings on discontinuities/dead-zones still hold).
- Topic notes (narrow scope, still accurate): `OBJECTIVE_NOTE.md`,
  `OCCUPANCY_NOTE.md`, `INDOOR_FILTRATION_NOTE.md`, `INLET_AND_BUILDING_NOTE.md`,
  `POLYDISPERSE_NOTE.md`, `REVERSE_SOLVER_NOTE.md`, `RANKING_STABILITY_NOTE.md`,
  `OVERNIGHT_DIAGNOSTIC_NOTE.md`, `GPU_KERNEL_AUDIT.md`.
- Contact-sheet renders: `examples_parameters.png` (one city per knob, verified),
  `examples_park_centrality.png`, `examples_larger.png`, `examples_roughness.png`.
