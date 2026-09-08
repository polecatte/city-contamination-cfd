# Project Map — `urban_openlbm`

A parametric urban-CFD pipeline for **exposure-driven city design**: a parametric city
builder generates geometries, a Lattice-Boltzmann wind solver drives a scalar/deposition
pollutant-exposure model, and a parameter-space + sensitivity + optimization layer searches
designs against the exposure objective.

Objective: **`J = (1/|Ω|) Σ_x w(x)·Θ(x)`** — receptor weighting `w` contracted with the
cumulative dose `Θ = ∫C dt`, from a burst released over Ω (every outdoor ground cell).

> **Status: mid-migration to OpenLB 1.8** (see `OPENLB_MIGRATION_PLAN.md`). The custom
> solver still runs and is kept as reference; the OpenLB engine is being built in parallel.
> Tests live in `tests/`; dead artifacts + superseded docs are quarantined in `_to_delete/`.

---

## 0 · OpenLB migration (new — Stage A / B bridge)

| File | Role | Status |
|---|---|---|
| `OPENLB_MIGRATION_PLAN.md` | the plan: three-stage swap, keep/retire lists, sequencing | current |
| `OPENLB_STEP12_STATUS.md`, `OPENLB_STEP3_STATUS.md` | progress + gate results | current |
| `openlb_geometry.h` | Stage A→B **geometry bridge**: voxel grid → OpenLB material map | done, gated |
| `gen_openlb_geom.cpp` | Stage A driver: builds city, writes `material_map.dat` / `source_mask.u8` / `receptor_w.f32` | done, gated |
| `geometry_loader.h` | Stage B: read material map + Ω mask; stamp a `SuperGeometry` | reader verified |
| `abl_inlet_olb.h` | ABL/RFG inlet as an OpenLB `AnalyticalF3D` | scaffold (`CONFIRM 1.8`) |
| `urban_flow.cpp` | Stage B **airflow app**: D3Q19 + WALE + ABL inlet + outlet + spin-up | scaffold (`CONFIRM 1.8`) |

Material scheme: `1 fluid · 2 wall(ground+buildings) · 3 inlet · 4 outlet · 5 slip · 6 porous(parks)`.
Ω stays a separate `source_mask.u8` (an AD-lattice property, not an NSE material).

## 1 · LBM airflow solver (custom engine — KEPT AS REFERENCE during migration)

| File | Role |
|---|---|
| `lbm_solver.h` / `lbm_solver.cpp` | Solver driver, `Config`, run phases, settling velocity |
| `lbm_kernels_cpu.cpp` | CPU kernels: D3Q19 stream+collide (MRT / regularized / HRR), WALE, D3Q7 scalar |
| `lbm_kernels.cu`, `lbm_gpu.h`, `adjoint_transport_gpu.cu` | CUDA backend + GPU adjoint |
| `abl_inlet.h` | ABL inlet turbulence (log-law + random-flow generator) — **ported to OpenLB, verified** |
| `main_cpu.cpp` | Legacy single-size entry point |
| `forward_city.cpp` | **Current production entry** — density-city forward flow+scalar solve (→ Stage A once OpenLB lands) |

Docs: `LBM_SOLVER.md`, `WALE_MODEL.md`, `HRR_ARCHITECTURE.md`, `GPU_KERNEL_AUDIT.md`,
`INLET_AND_BUILDING_NOTE.md`, `PRODUCTION_SOLVE.md`, `FORWARD_LIVE.md`

## 2 · Scalar transport, deposition & exposure

| File | Role |
|---|---|
| `adjoint_transport.h` | Reverse/adjoint transport operator (matches forward QUICK) |
| `reverse_objective.h` | Receptor field `w`, source mask, `J = ⟨w,Θ⟩`, reciprocity check |
| `exposure_receptor.h` | Building-anchored inhabitance → receptor weighting |
| `deposition.h`, `psd.h` | Surface deposition; particle-size distribution / bins |
| `infiltration.h`, `indoor_exposure.h` | Envelope infiltration `F_inf`; indoor dose |
| `occupancy.h` | NHAPS time-budget occupancy (home/work/park) |
| `exposure_solve.cpp`, `forward_live.cpp` | Full exposure pipeline; live-burst ensemble |
| `anon_exposure.py` | Cheap source-anonymous advection-diffusion-removal proxy |

Docs: `EXPOSURE_METRIC.md`, `ADJOINT_SOLVER.md`, `REVERSE_SOLVER_NOTE.md`, `CONTAMINANT_BC.md`,
`INFILTRATION_MODEL.md`, `INDOOR_FILTRATION_NOTE.md`, `RELEASE_MECHANISM.md`, `OBJECTIVE_NOTE.md`,
`POLYDISPERSE_NOTE.md`, `OCCUPANCY_NOTE.md`, `PARK_POROSITY.md`

## 3 · City generation & morphology

| File | Role |
|---|---|
| `city_builder7.h` | Core block-grid builder (heights, parks, employment balance, NHAPS) — untouched base |
| `city_zoning.h` | Zoning overlays: mixed potential field, Gray-Scott reaction-diffusion, **density-field population model** (`dens_enable`) |
| `voxelize.h`, `must_geom.h` | Geometry → voxel grid for the LBM; MUST field-experiment geometry |
| `gen_density_city.cpp` | **Density-field population zoning driver (current)** |
| `gen_rd_city.cpp`, `gen_zoning_demo.cpp` | Reaction-diffusion / potential-field zoning drivers |
| `gen_examples.cpp`, `gen_larger.cpp`, `gen_parks.cpp`, `gen_roughness.cpp`, `gen_cost732.cpp`, `gen_paramsheet.cpp`, `gen_paramspace.cpp` | Other geometry generators |
| `render_city.py`, `render_domain.py`, `city3d.py`, `visualize_forward.py` | Production renderers / domain + 3D + forward-field viz |
| `gen_city_v2.py`, `gen_city_gallery.py`, `render_zoning_demo.py`, `pattern_strategies.py`, `rd_coarse.py`, `gen_paramspace.py`, `citydemo.py`, `render_examples.py`, `render_larger.py`, `render_parks.py`, `render_roughness.py`, `render_paramsheet.py` | City-gen suites / galleries / renderers |

Doc: `PARAM_SPACE_NOTE.md`.

## 4 · Parameter space, sensitivity & optimization

| File | Role |
|---|---|
| `param_space.py`, `param_space_v2.py` | Earlier 9-D / 16-D morphology spaces |
| `param_space_density.py` | **12-D density-field space (current)** |
| `run_optimization.py` | Multi-resolution Bayesian optimization |
| `sensitivity.cpp`, `sensitivity.py`, `sensitivity_density.py` | Morphology-metric + density-param sensitivity / Morris screening |
| `metrics.h`, `rank_stats.h` | Morphology metrics; rank statistics |

Docs: `PARAM_SPACE_NOTE.md`, `RANKING_STABILITY_NOTE.md`.

## 5 · Tests & validation → `tests/`

All test drivers, validation binaries, diagnostics, demos, and their plot/analysis scripts
now live in **`tests/`**. Compile from the project root with `-I.`
(`g++ -O3 -std=c++17 -I. tests/<file>.cpp`). Contents include:

- **Airflow:** `airflow_validation.cpp/.py`, `airflow_validation_plots.py`, `plume_validation.cpp`,
  `oblique_validation.cpp`, `oblique_divergence_test.cpp`, `overshoot_test.cpp`,
  `test_poiseuille.cpp`, `test_abl_inlet.cpp`, `test_abl_run.cpp`, `abl_inlet_verify.cpp` (OpenLB inlet)
- **Dispersion / deposition:** `diffusion_compare.cpp`, `test_deposition.cpp`, `test_psd_dep.cpp`, `adjoint_recip_test.cpp`
- **Field benchmark:** `must_benchmark.cpp`, `must_score.cpp`, `canopy_calib.cpp`
- **Robustness / demos / probes:** `robustness_overnight.cpp`, `ranking_stability.cpp`, `check_examples.cpp`,
  `probe.cpp`, `show_bins.cpp`, `size_domains.cpp`, `demo_indoor/infil/occupancy.cpp`, `polydisperse_demo.cpp`, `lab_test.cpp`
- **Optimizer audits / logic:** `test_opt_logic.py`, `sweep_audit.py`, `exploit_scan.py`, `verify_redundancy.py`
- **Plot / analysis (Py):** `plot_*.py`, `analyze_phase0.py`, `viz_airflow.py`, `viz_borders.py`,
  `adjoint_test.py`, `lab_visualize.py`, `re_compare.py`, `render_material_map.py`
- **Runner scripts:** `run_airflow_validation.sh`, `run_forward_validation.sh`, `run_oblique_*.sh`,
  `run_labverify.sh`, `package_labverify.sh`, `run_stability_sweep.sh`, `re_test.sh`, `run_overnight.sh`
- `tests/hrr_unit.cpp` (already there)

Docs (in root): `VALIDATION_ROSTER.md`, `TEST_SUITE.md`, `VERIFICATION_AUDIT.md`, `AIRFLOW_VALIDATION_NOTE.md`.

## 6 · Build & run scripts (root)

- **Build (custom engine):** `build.sh`, `build_diffusion.sh`, `build_gpu.sh`, `fix_gpu_build.sh`, `launch.sh`
- **Run (production):** `run_forward_city.sh`, `run_exposure.sh`
- **Phase-0 cluster campaign:** `run_phase0.sh`, `submit_phase0.sh`, `phase0_aces.slurm`

## 7 · Data & config (`.csv`)

- **Inflow / ABL:** `abl_profile.csv`, `abl_tseries.csv`, `av_inflow{,_profiles,_spectrum}.csv`
- **Physics / demo:** `psd_bins.csv`, `vd_curve.csv`, `infil.csv`, `indoor_demo.csv`, `occupancy_demo.csv`, `sensitivity.csv`

## Top-level docs

`README.md`, `ARCHITECTURE.md`, `CHANGES.md`, `HANDOFF.md`, `OPENLB_MIGRATION_PLAN.md`.

---

## `_to_delete/` (quarantine — review then `rm -rf`)

Compiled binaries checked into the repo (`gen_density_city`, `gen_paramspace`, `gen_rd_city`,
`gen_zoning_demo`, `plume_validation`) and superseded docs (`OBLIQUE_DIVERGENCE_DIAGNOSIS.md`,
`OVERNIGHT_DIAGNOSTIC_NOTE.md`, `TECHNICAL_STATUS_AND_ROADMAP.md`, `IMPROVEMENT_PLAN.md` —
the old-solver bug diagnoses and pre-OpenLB roadmaps, superseded by `OPENLB_MIGRATION_PLAN.md`).

## Data flow

```
 param_space_*.py ──> gen_density_city (city_builder7.h + city_zoning.h) ──> city .txt
                                                                              │
                                         voxelize.h ──> geometry ────────────┤
                                                                              │
                       ┌─ Stage A: gen_openlb_geom.cpp ──> material_map.dat, source_mask.u8, receptor_w.f32
                       │
   abl_inlet.h ──> Stage B flow (custom lbm_solver NOW / OpenLB urban_flow NEXT) ──> u, ν_t ──> D3Q7 scalar C
                                                                              │
   occupancy.h ──> receptor w ───────────────┐                              v
                                              └──> J = (1/|Ω|)Σ w·Θ  <──── Θ = ∫C dt   (Stage C)
                                                        │
                                          run_optimization.py / sensitivity_density.py
```
