# OpenLB Migration Plan — `urban_openlbm`

Replacing the custom D3Q19 airflow + D3Q7 scalar + deposition stack with an
**OpenLB-based** flow/transport engine, while preserving the project's real IP: the
parametric city builder, the receptor/occupancy/infiltration model, the exposure metric,
the parameter space, and the visualization suite.

Target library: **OpenLB 1.8** (User Guide released Aug 2025) — C++ template LBM with
GPU (CUDA), OpenMP/MPI, LES turbulence (Smagorinsky / consistent-Smagorinsky / WALE),
advection–diffusion lattices with Navier–Stokes coupling, porous-media dynamics, and a
particle subsystem.

> **Assumptions this plan runs on** (say the word to change any): (1) OpenLB takes over
> airflow **and** scalar transport + deposition as a coupled NSE+AD run; (2) deposition
> is Eulerian (settling velocity + surface sink), preserving source-superposition
> linearity; (3) the adjoint/reverse path is *not* ported in v1 (forward burst-over-Ω
> only); (4) GPU production on the A4000, CPU prototype in the cloud; (5) nothing is
> deleted until you approve the retire list in §5.

---

## 1. Why this split is the right seam

The current design already has a clean seam the migration can exploit: the city builder
produces geometry + a receptor field `w` + a source set Ω, the solver produces a
concentration/dose field, and the exposure metric contracts `⟨w, Θ⟩/|Ω|`. Only the
middle box — *solve the flow, transport the burst, deposit* — is being replaced. So the
migration is fundamentally a **swap of one well-bounded engine**, not a rewrite of the
science.

The key property we must not break is **linearity of the dose in the source**. Your metric
relies on releasing a unit burst over every Ω cell at once and treating the result as the
superposition of per-cell releases (`J(s₁+…+sₙ)=ΣJ(sᵢ)`). OpenLB's advection–diffusion
lattice is linear, and an Eulerian settling+sink deposition is linear, so this survives.
A Lagrangian particle deposition would *not* preserve it — that is the one modeling fork
that would ripple into the metric, and it's why the Eulerian default is chosen.

---

## 2. Target architecture — three stages

Because an OpenLB app is a standalone `main()` in the OpenLB build tree, the cleanest
architecture splits today's monolithic `forward_city.cpp` into three cooperating stages
that pass files:

```
 ┌─ STAGE A · scene build (existing C++, no OpenLB) ──────────────────────────┐
 │  city_builder7.h + city_zoning.h  →  blocks, heights, land use             │
 │  voxelize.h                        →  cell-type / material grid            │
 │  exposure_receptor.h + occupancy.h →  receptor field w                     │
 │  Ω assembly (ground cells)         →  source mask                          │
 │  writes:  material_map.dat, source_mask.u8, receptor_w.f32, meta.txt       │
 └───────────────────────────────────────────────────────────────────────────┘
                                   │  (self-describing binaries — unchanged format)
                                   ▼
 ┌─ STAGE B · OpenLB app  `urban_flow`  (the new engine) ─────────────────────┐
 │  reads material_map + source_mask                                          │
 │  SuperGeometry ← material numbers (buildings solid, parks porous, Ω ground)│
 │  NSE lattice  : D3Q19, WALE LES, ABL log-law + synthetic-turbulence inlet  │
 │  AD lattice   : D3Q7, coupled to NSE velocity, settling offset + sink      │
 │  burst over Ω → integrate Θ = ∫C dt, deposition field, per-frame slices    │
 │  writes:  umean.f32, nut.f32, theta.f32, deposition.f32, frames/, exposure_timeseries.csv │
 └───────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
 ┌─ STAGE C · exposure reduction + viz (existing Python/C++) ─────────────────┐
 │  J = (1/|Ω|) Σ_x w(x)·Θ(x)   (reverse_objective.h contraction, unchanged)  │
 │  visualize_forward.py, render_domain.py, city3d.py  (consume same fields)  │
 └───────────────────────────────────────────────────────────────────────────┘
```

Stages A and C are **kept almost verbatim** — they are the IP. The binary interchange
format (5-int header `[nx,ny,nz,dx*1000,ncomp]`) stays identical, so `visualize_forward.py`
and the metric need no changes beyond pointing at the new run directory. That is the design
goal: the OpenLB engine is a drop-in for the middle box.

---

## 3. Module-by-module mapping (current → OpenLB)

| Current mechanism | File | OpenLB 1.8 replacement |
|---|---|---|
| D3Q19 stream+collide, MRT/HRR | `lbm_kernels*.{cu,cpp}` | `SuperLattice` with `D3Q19` descriptor + BGK/MRT/regularized dynamics |
| WALE eddy viscosity | in kernels | `WALEBGKdynamics` (or consistent-Smagorinsky); built-in |
| ABL log-law inlet + RFG turbulence | `abl_inlet.h` | **custom** velocity BC: log-law mean via `AnalyticalF` + ported synthetic-turbulence generator (§6.1) |
| Porous park canopy (partial bounce-back, perm 0.8) | `voxelize.h` + kernels | `PorousBGKdynamics` / partial-bounce-back material, tuned to the same C_d, LAD |
| D3Q7 scalar, QUICK/van-Leer, TRT diffusion | `lbm_kernels*`, `adjoint_transport.h` | second **AD lattice** (`D3Q7` AD descriptor) coupled to NSE via a velocity coupling; OpenLB's AD dynamics |
| Settling velocity (Stokes+Cunningham) | `deposition.h`, `psd.h` | advection-velocity offset `−w_s ẑ` on the AD lattice (per size bin) |
| Surface deposition sink | `deposition.h` | **custom post-processor** on ground/building boundary cells (must be GPU-side; §6.3) |
| Burst source over Ω | `forward_city.cpp` srcmask | AD source term gated by the imported source-mask material |
| Θ = ∫C dt, mass budget | `lbm_solver.cpp` | time-accumulator field on the AD lattice + reductions |
| Frozen-mean / warm restart | `lbm_solver.cpp` | OpenLB checkpoint (`save`/`load`) of the converged NSE lattice |
| GPU backend | `lbm_kernels.cu`, `lbm_gpu.h` | OpenLB `Platform::GPU_CUDA` — same app code, no separate kernel file |
| Adjoint/reciprocity | `adjoint_transport*`, `reverse_objective.h` | **not ported in v1** (forward-only); revisit if the optimizer needs gradients |

---

## 4. Files to KEEP (the IP — untouched or lightly adapted)

- **City generation:** `city_builder7.h`, `city_zoning.h`, `voxelize.h` (adds a material-map
  writer for OpenLB), `must_geom.h`, the `gen_*_city.cpp` drivers.
- **Receptor & exposure:** `exposure_receptor.h`, `occupancy.h`, `infiltration.h`,
  `indoor_exposure.h`, `reverse_objective.h` (the `w`-contraction half), `psd.h`,
  `metrics.h`, `rank_stats.h`; data `vd_curve.csv`, `psd_bins.csv`, `infil.csv`.
- **Search & analysis:** `param_space*.py`, `sensitivity_density.py`, `run_optimization.py`,
  `anon_exposure.py`, `test_opt_logic.py`, `verify_redundancy.py`.
- **Visualization:** `visualize_forward.py`, `render_*.py`, `city3d.py`, `render_domain.py`.
- **Docs:** all `*.md` (they describe models we're keeping; a few get an OpenLB addendum).
- **`forward_city.cpp`:** refactored into Stage A (scene build) — the solver call is removed
  and replaced by writing the OpenLB inputs.

## 5. Files to RETIRE (old engine — set aside on approval)

Moved into `_to_delete/` (the bridge cannot hard-delete; you empty that folder yourself).
Grouped so you can veto any group:

- **Custom solver core:** `lbm_solver.cpp`, `lbm_solver.h`, `lbm_kernels.cu`,
  `lbm_kernels_cpu.cpp`, `lbm_gpu.h`, `main_cpu.cpp`.
- **Inlet / transport / deposition to be reimplemented in OpenLB:** `abl_inlet.h`,
  `deposition.h`, `adjoint_transport.h`, `adjoint_transport_gpu.cu`. *(Keep as reference
  until the OpenLB equivalents pass their gates — see §7 — then retire.)*
- **Old-solver validation & tests:** `airflow_validation.cpp/.py`, `airflow_validation_plots.py`,
  `plume_validation.cpp`, `oblique_validation.cpp`, `oblique_divergence_test.cpp`,
  `overshoot_test.cpp`, `test_poiseuille.cpp`, `test_abl_inlet.cpp`, `test_abl_run.cpp`,
  `diffusion_compare.cpp`, `test_deposition.cpp`, `test_psd_dep.cpp`, `adjoint_recip_test.cpp`,
  `adjoint_test.py`, `must_benchmark.cpp`, `must_score.cpp`, `canopy_calib.cpp`,
  `robustness_overnight.cpp`, `ranking_stability.cpp`, `re_compare.py`, `size_domains.cpp`,
  `probe.cpp`, `show_bins.cpp`.
- **Old build/run scripts:** `build.sh`, `build_gpu.sh`, `build_diffusion.sh`,
  `fix_gpu_build.sh`, `launch.sh`, `run_airflow_validation.sh`, `run_oblique_*.sh`,
  `run_stability_sweep.sh`, `run_forward_validation.sh`, `run_labverify.sh`,
  `package_labverify.sh`, `re_test.sh`, `run_overnight.sh`, `run_phase0.sh`,
  `submit_phase0.sh`, `phase0_aces.slurm`.
- **Ambiguous — flag before moving:** `exposure_solve.cpp`, `forward_live.cpp`,
  `lab_test.cpp` (old solver-coupled drivers; their receptor/exposure logic is worth
  salvaging into Stage C before retiring the solver calls).

Everything in the retire list is either the engine we're replacing or a test/validation of
that specific engine. The MUST field-benchmark (`must_*`) validation is retired only because
it drives the old solver — the benchmark itself should be **re-created against the OpenLB
engine** in §7, since field-experiment validation is exactly what earns the new engine trust.

## 6. The three genuine porting challenges (where the work actually is)

### 6.1 ABL inlet with synthetic turbulence — the hardest piece
OpenLB ships LES (the `nozzle3d` example is the canonical turbulent case) but **not** a
turnkey atmospheric-boundary-layer inlet with divergence-free synthetic turbulence. Your
`abl_inlet.h` already implements the Kraichnan/Smirnov random-flow-generator (RFG) log-law
inlet; the work is porting that as a custom OpenLB velocity boundary that writes the
per-step inflow plane. This is well-trodden in the literature (divergence-free spectral
inflow generators for ABL LES) and your RFG parameters carry over directly — but budget it
as the single biggest task and validate it first (mean-shear + turbulence-intensity
profiles) before anything downstream.

### 6.2 Preserving exposure linearity & the Ω burst
The AD lattice and Eulerian deposition are linear, so `J(Σsᵢ)=ΣJ(sᵢ)` holds — but the
burst-over-Ω injection, the `Θ=∫C dt` accumulator, and the mass budget must be
reimplemented as OpenLB operators. Add a regression that releases from a single cell and
checks the field equals the corresponding slice of the full-Ω release (linearity guard),
mirroring the old reciprocity test's role.

### 6.3 GPU operator parity
On `Platform::GPU_CUDA`, every custom piece — the turbulence inlet, the source injection,
the settling offset, and especially the **surface deposition sink** — must be written as
OpenLB operators/post-processors so they run on-device. Anything left as host-side
per-cell C++ (as several current kernels are) will not run on GPU. This is the constraint
that most shapes how the custom code is written; plan every custom operator as GPU-first.

Two smaller items: **porous parks** map to `PorousBGKdynamics` / partial-bounce-back tuned
to the same drag (C_d≈0.2, LAD≈1) — a calibration task, not a research one; and the
**geometry bridge** should stamp OpenLB material numbers directly from the existing voxel
grid (rather than via STL) so the new solid/porous/ground classification matches the old
voxelization cell-for-cell.

### 6.4 One face per cell — domain-boundary adjacency

**This was not anticipated and it is worth stating at length, because it is the first place
the migration exposed a latent defect in geometry that had always passed its own gate.**

The custom solver applied boundary conditions per material number and never needed to know
which way a boundary faced. Bounce-back is direction-agnostic, and the old inlet wrote a
prescribed profile onto whatever cells carried `MAT_INLET`. So the material map only ever
had to be a correct *census*: every cell classified exactly once, sums reconciling against
the voxelizer. It was, and Stage A's gate proves it.

OpenLB's interpolated and slip boundaries need more than a census. Each boundary cell must
have a well-defined discrete inward normal, which OpenLB derives from the cell's material
neighbourhood: it looks for the direction in which the bulk fluid lies. A boundary cell with
no bulk-fluid neighbour has no derivable normal, and `boundary::set<>` refuses with
`std::runtime_error: Could not set Boundary`.

The original overlay produced exactly that. It ran three passes in sequence — inlet/outlet
on the wind-aligned faces, then the lateral faces, then the top — each claiming any cell
still marked `MAT_FLUID`. Because the inlet pass ran first and swept its entire plane, it
claimed the two columns where the `x=0` plane meets the lateral slip planes at `y=0` and
`y=ny-1`. Those cells are on the inlet, but every neighbour of theirs is another boundary
cell: `+x` is SLIP, not fluid. The same happened mirrored at the outlet, and along the seam
where the top face meets the lateral faces. At the default operating point:

| material | orphaned cells | location |
|---|---|---|
| 3 INLET | 341 | `x=0`, columns `y=0` and `y=ny-1` |
| 4 OUTLET | 341 | `x=nx-1`, same columns |
| 5 SLIP | 350 | `z=nz-1` top, seams along `y=0` and `y=ny-1` |

1 032 cells out of 2.63 M — 0.04 % — and the run aborted on the first of them.

**The rule now: a cell belongs to exactly one domain face.** `build_material_map` counts how
many of the six faces a cell lies on. Zero means interior. One means the cell takes that
face's material — INLET, OUTLET, or SLIP. **Two or more means the cell is on a box edge or
corner, and it becomes `MAT_DONOTHING`.**

Assigning the edges to no-dynamics is not a workaround, it is the physically correct answer.
A cell on two domain faces has no unique inward normal *by construction* — that is a property
of the box, not of the discretization. And such a cell touches no fluid cell at all, so it
cannot exchange populations with the flow and cannot influence the solution regardless of
what dynamics it carries. Giving it no dynamics says exactly that. The alternative readings
are worse: an arbitrary normal is a fabricated boundary condition, and leaving it as bulk
fluid puts an unconstrained cell outside the domain's boundary.

The bottom edges need no special handling: `z=0` is `MAT_GROUND` from the type pass, and the
overlay only reclassifies cells still marked `MAT_FLUID`.

**Consequences for Stage A's gate.** Three counts change, and the fourth reconciliation
identity changes shape:

| | before | after |
|---|---|---|
| MAT 0 DONOTHING | 0 | **1 032** |
| MAT 3 INLET | 14 696 | **14 355** |
| MAT 4 OUTLET | 14 696 | **14 355** |
| MAT 5 SLIP | 59 675 | **59 325** |
| identity 4 | `fluid+inlet+outlet+slip = FLUID` | `fluid+in+out+slip+donothing = FLUID` |

The edge cells were carved out of fluid, so they belong on the fluid side of that identity.
FLUID, WALL, POROUS and GROUND are untouched, all five identities still PASS, and `Ω` stays
22 812 — the source mask is ground-level and never touched a domain face.

**Why change Stage A rather than the solver.** The alternative was to leave the map alone and
have `prepareLattice` set boundaries on a trimmed indicator. That works, but it puts the
solver's boundary layout permanently out of step with the material map, which is the one
artefact a reviewer can inspect directly — and it leaves a map that is wrong in a way that
only shows up in a solver nobody has run yet. Re-baselining the Gate 3 reference numbers is
a one-time cost paid while the numbers are cheap; explaining a standing discrepancy is not.

**What this says about the migration.** The defect was latent in the geometry for the entire
life of the custom solver and its own gate could not see it, because the gate checks counts
and the defect is in adjacency. A validated community solver is stricter than a bespoke one
in ways that are not predictable in advance — that strictness is part of what the migration
buys, and this is the first instance of it paying out. It also argues for keeping Gate 5
(OpenLB's own per-material voxel counts) permanently rather than treating it as a one-off:
it is the only check that sees the geometry the way the solver does.

## 7. Sequencing (each step gated by a test before the next)

1. **Environment.** Confirm/instal OpenLB 1.8 on the A4000 (GPU build) and a CPU build in
   the cloud for fast compile checks. *Gate: `nozzle3d` example builds and runs on both.*
2. **Geometry bridge.** `voxelize.h` → OpenLB `SuperGeometry` material map; render it and
   diff against the current voxel summary. *Gate: material counts match the old voxelizer.*
3. **Airflow only.** NSE + WALE + ABL/RFG inlet on a city; compare mean-shear and
   turbulence-intensity profiles and a cube-wake reattachment `Xr/H` against the benchmark
   band (~1.6) — the very check the old solver failed. *Gate: `Xr/H` in band, ABL drift <10%.*
4. **Scalar + deposition.** Add the coupled AD lattice, settling offset, surface sink;
   run a single-cell release. *Gate: linearity guard (§6.2) + mass budget closes ~99%.*
5. **Full forward run.** Burst over Ω, `Θ` accumulation, per-frame slices, timeseries;
   wire Stage A and Stage C around it. *Gate: end-to-end `J` finite; viz suite renders.*
6. **Retire.** Once steps 3–5 pass, move the §5 reference files into `_to_delete/`.
7. **Re-validate.** Rebuild the MUST field benchmark and a plume validation against the
   OpenLB engine (replacing the retired ones) — the credibility payoff of the whole move.

## 8. Environment reality check

This session reaches your Mac (the connected folder) and a cloud Linux container, but **not
the A4000 box directly**. I can build and smoke-test OpenLB CPU-side in the cloud to
de-risk compilation and the custom operators, then hand you a GPU build recipe for the
A4000; you run production there (as before). If you'd rather I drive the A4000 build, we'd
need it reachable from this session — otherwise the loop is: I write + CPU-verify here →
you `scp`/pull to the A4000 → GPU run. Confirming whether OpenLB is already installed on
that box is step-1 blocking, so tell me if it is and which version.

## 9. What I need from you to start executing

Nothing blocking for the plan itself. To move into code I need: (a) any changes to the five
assumptions at the top; (b) whether OpenLB is installed on the A4000 and at what version;
(c) go-ahead to set aside the §5 files. On your word I'll start with steps 1–2 (environment
+ geometry bridge), which are non-destructive and independent of the retire decision.
