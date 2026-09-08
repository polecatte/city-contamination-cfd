# Directory cleanup — audit & bundles (2026-09-08)

A pass over all 130 tracked files asking: what is genuinely dead, what is duplicated, and
what documentation no longer describes the code. Method was mechanical, not impressionistic
— a reference graph (who includes/imports/invokes each file), an orphan scan, and a check
of every ``backticked`` filename in every `.md` against the filesystem.

**Nothing was hard-deleted.** Two files moved to `_to_delete/`, which is this project's
existing quarantine convention (`OPENLB_MIGRATION_PLAN.md` §5: "the bridge cannot
hard-delete; you empty that folder yourself"). Everything else is *classified*, not touched,
because the calls involved are yours: several look dead but are load-bearing, and the
migration plan explicitly gates the big retirement on OpenLB passing gates it has not passed.

---

## Bundle 1 — TO DELETE (moved to `_to_delete/`, review then `rm -rf`)

| File | Evidence it is dead |
|---|---|
| `adjoint_transport_gpu.cu` | **The old GPU adjoint.** Included by zero files, compiled by zero build scripts (`build_gpu.sh` and `fix_gpu_build.sh` never mention it), and the GPU path has never run on-device at all — `HANDOFF.md` §2 records that `nvcc` was never available. `OPENLB_MIGRATION_PLAN.md` assumption (3) defers the adjoint past v1 entirely. Dead on every axis. |
| `cleanup.sh` | **Already executed, now a no-op.** It is the script that created `tests/` and quarantined the four superseded docs. Every source path it moves is already gone, so re-running it does nothing. Superseded by this document. |

**Correction — `build.sh` was quarantined here and then restored.** It compiles
`lab_test.cpp` from a path that no longer exists, so it looked dead. It is not:
`build_diffusion.sh:26` requires the `kernels.o` + `solver.o` that only `build.sh`
(CPU) or `build_gpu.sh` (GPU) produce, and tells the user to run it. Quarantining it
would have silently removed the CPU route into the diffusion build. It has been put
back with its two stale paths fixed (`tests/lab_test.cpp` with `-I.`,
`tests/lab_visualize.py`) and **verified by running it** — it builds `lab_test` and
`solver` cleanly. That is one item off the Bundle 2.2 list, fixed rather than filed.

---

## Bundle 2 — TO UPDATE (kept in place; each needs a specific edit)

Ordered by how much damage the staleness does.

### 2.1 `HANDOFF.md` — the worst offender, and it is the designated entry point
Its first line is "read this first". It contains **zero** mentions of OpenLB. It describes
the project at "v8.3" with a 9-D search space and the custom solver as *the* path, which is
one whole migration out of date — so the one file a person opens cold is the one that will
mislead them. It also references four files that do not exist anywhere in the bundle
(`bayesopt.py`, `evaluate.cpp`, `evaluate.py`, `OVERNIGHT_DIAGNOSTIC_NOTE.md`) and six that
moved into `tests/`.
**Fix:** either rewrite §2–3 around the migration and point at `README.md` /
`OPENLB_PORT_STATUS_AND_VERIFICATION.md`, or demote it to `HANDOFF_PRE_OPENLB.md` and let
`README.md` be the entry point. The second is cheaper and loses nothing.

### 2.2 Ten runner scripts broken by the earlier `tests/` move
`cleanup.sh` moved the test sources but not the paths inside the scripts that build them,
and its own closing note under-called this as "a few". The scripts reference sources by bare
name, and none of them `cd` anywhere, so they fail from the project root (the file is in
`tests/`) *and* from `tests/` (where `-I.` no longer reaches the headers).

| Script | Broken reference(s) |
|---|---|
| `build_gpu.sh` | `lab_test.cpp` (same fix as `build.sh`, which is already done) |
| `run_phase0.sh` | `analyze_phase0.py` |
| `phase0_aces.slurm` | `airflow_validation.cpp`, `analyze_phase0.py` |
| `tests/run_airflow_validation.sh` | `airflow_validation.cpp/.py`, `airflow_validation_plots.py`, `test_poiseuille.cpp`, `viz_airflow.py` |
| `tests/run_forward_validation.sh` | `overshoot_test.cpp`, `plume_validation.cpp` |
| `tests/run_oblique_validation.sh` | `oblique_validation.cpp` |
| `tests/run_oblique_diagnosis.sh` | `oblique_divergence_test.cpp` |
| `tests/run_overnight.sh` | `airflow_validation.cpp` |
| `tests/run_stability_sweep.sh` | `plot_stability.py` |
| `tests/run_labverify.sh`, `tests/package_labverify.sh`, `tests/re_test.sh` | `airflow_validation.py`, `viz_airflow.py`, `re_compare.py` |

**Fix:** one line at the top of each — `cd "$(dirname "$0")/.."` — then prefix the test
sources with `tests/` and keep `-I.`. This is exactly the fix already applied to and
verified on `build.sh`. Not done for the rest here: these drive the custom engine through
multi-hour runs, so a blind path edit is unverifiable in this environment and you would not
find out it was wrong until an overnight batch failed. It is a 20-minute job on the lab box
where each can actually be smoke-tested.

### 2.3 `PROJECT_MAP.md` — two factual errors
- It documents `_to_delete/` and its contents as if present; the directory did not exist
  until this pass (the compiled binaries and four superseded docs it lists were already
  gone from the uploaded snapshot).
- It labels `param_space_density.py` "(current)" and `param_space.py` "earlier". In fact
  `param_space.py` is the one `run_optimization.py` imports — see 3.2.

### 2.4 Fourteen docs referencing pre-move test paths
`AIRFLOW_VALIDATION_NOTE.md`, `CHANGES.md`, `GPU_KERNEL_AUDIT.md`, `INDOOR_FILTRATION_NOTE.md`,
`OBJECTIVE_NOTE.md`, `OCCUPANCY_NOTE.md`, `OPENLB_MIGRATION_PLAN.md`, `TEST_SUITE.md`,
`VALIDATION_ROSTER.md` and others cite e.g. `airflow_validation.cpp` where the file is now
`tests/airflow_validation.cpp`. Mechanical and harmless individually; collectively it makes
every roster in the repo untrustworthy. **Fix:** a scripted `tests/` prefix pass, verified by
re-running the existence check that produced this list.
`CHANGES.md` additionally points at `TECHNICAL_STATUS_AND_ROADMAP.md`, which was deleted.

### 2.5 `CHANGES.md` is a patch note, not a changelog
It reads "Drop these 7 files into your tree" — it documents one specific hand-off of three
fixes, not the project's history. Worth retitling (`FIXES_2026_REGULARIZED_WALL_SWEEP.md`)
so it is not mistaken for a running changelog, or folding its three fix write-ups into
`LBM_SOLVER.md` / `WALE_MODEL.md` where the models live.

---

## Bundle 3 — DECIDE (real duplication; your call, not mine)

### 3.1 `exposure_receptor.h` — documented as used, actually unused
`OPENLB_MIGRATION_PLAN.md` §2 puts it in the Stage-A architecture diagram, §4 puts it on the
KEEP list, `README.md` shows it in the Stage-A box, and `PROJECT_MAP.md` gives it a row. **No
code includes it.** The receptor field is built inline instead — `forward_city.cpp:202` and
`gen_openlb_geom.cpp:107`, the latter commented "ported from forward_city.cpp". So there are
two receptor implementations and the documented one is the dead one.

It is not obviously the worse implementation: it is building-anchored via 6-connected
component labelling of solid cells, which is what `EXPOSURE_METRIC.md` actually specifies,
whereas the inline versions are per-envelope-cell. **Either wire it in (and delete the
inline duplicate) or delete it (and fix four docs).** Leaving it is the one option that
keeps costing — every reader of the plan believes Stage A calls it.

### 3.2 Three parameter spaces, all live
| File | Imported by | Status |
|---|---|---|
| `param_space.py` (9-D) | `run_optimization.py`, `tests/{sweep_audit,exploit_scan,test_opt_logic}.py` | **the one the optimizer actually uses** |
| `param_space_density.py` (12-D) | `sensitivity_density.py` | current per the docs, but not wired to the optimizer |
| `param_space_v2.py` (13-D) | `gen_city_v2.py` only | the v2 archetype line; a closed pair with its one consumer |

The docs call the density space "current", but the optimizer runs on the 9-D one. Either
`run_optimization.py` should move to the density space or the docs should stop calling it
current — as it stands, `PARAM_SPACE_NOTE.md` and `run_optimization.py` disagree about what
the project is optimizing. `param_space_v2.py` + `gen_city_v2.py` retire together or not at
all.

*(Note: `run_optimization.py` imports `bayesopt`, which is not in this bundle —
`HANDOFF.md` §9 says it lives outside. The optimizer is therefore not runnable from a fresh
clone regardless.)*

---

## Bundle 4 — KEEP (looks retirable; is not, yet)

- **The whole custom engine** — `lbm_solver.{h,cpp}`, `lbm_kernels{_cpu.cpp,.cu}`,
  `lbm_gpu.h`, `main_cpu.cpp`, `abl_inlet.h`, `deposition.h`, `forward_city.cpp` and their
  tests. `OPENLB_MIGRATION_PLAN.md` §5 retires these only "once steps 3–5 pass". Step 3 has
  not passed; `urban_flow.cpp` has never been compiled. This is still the only working
  engine and retiring it now would leave the project with no solver at all.

- **`adjoint_transport.h` — blocked by a shallow coupling, not by policy.** You asked
  specifically about old adjoint methods, so: the GPU half is gone (Bundle 1), but the CPU
  header cannot follow it yet. `reverse_objective.h` — which the migration plan explicitly
  KEEPS for its `w`-contraction half — reopens `namespace adj` and depends on the `FLUID`
  enum and the `Field` struct that `adjoint_transport.h` defines. I verified this by
  compiling `reverse_objective.h` with the include stripped; it fails on `FLUID` at two
  sites. It does *not* use the actual adjoint machinery (`fwd_step`, `adj_step`).
  **So the retirement is a ~20-line extraction**: move `Field`, the cell-type enum and the
  index helpers into a small `transport_types.h`, include that from both, and
  `adjoint_transport.h` becomes free-standing and retirable with its two tests
  (`tests/adjoint_recip_test.cpp`, `tests/adjoint_test.py`). Worth doing when the adjoint is
  formally dropped — not before, since those two tests are live validation of the engine
  that is still in production.

- **The `gen_*` / `render_*` figure generators** (`citydemo.py`, `gen_city_gallery.py`,
  `pattern_strategies.py`, `rd_coarse.py`, `render_{examples,larger,parks,roughness,paramsheet,zoning_demo}.py`
  and their `gen_*.cpp` partners). These look like clutter — 13 files, each invoked by
  nothing — but they are the reproduction scripts for figures in the topic notes, they reuse
  `render_city.py`'s draw functions rather than duplicating them, and each targets a
  distinct note. Deleting them would cost figure reproducibility and save nothing but
  directory noise. If you want them out of the top level, `figures/` is the move, not
  `_to_delete/`.

- **`fix_gpu_build.sh`** — looks like a spent one-off, but it is an idempotent in-place
  patcher for an `nvcc`/`std::function` failure on the A4000, and the custom GPU path is
  still live. Retires with the custom engine, not before.

---

## What did NOT turn up

No duplicated data files, no orphaned CSVs (every `.csv` is read by a named consumer), no
compiled binaries checked in (the earlier pass caught those), and no genuinely redundant
pair of source files beyond §3.1. The directory's real problem is not junk files — it is
that **the documentation describes a repository layout and an architecture that the code no
longer matches**, which is Bundle 2's whole content.
