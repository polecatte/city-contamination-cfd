# urban_openlbm — exposure-driven urban CFD

A parametric urban-CFD pipeline for **exposure-driven city design**: a parametric city
builder generates geometries, a Lattice-Boltzmann wind solver drives a scalar pollutant-
transport + deposition model, and a parameter-space / sensitivity / optimization layer
searches designs against a population-exposure objective.

Governing functional: **`J = (1/|Ω|) Σ_x w(x)·Θ(x)`** — the receptor/occupancy weighting
`w` (where people are, by time of day) contracted with the cumulative dose `Θ = ∫C dt`
from the flow + transport solve, released as a burst over the source set Ω (every outdoor
ground cell). Lower `J` = a lower-exposure design; compare `J` **across** designs.

---

## Status: mid-migration to OpenLB

The project is **replacing its custom D3Q19/D3Q7 solver with an OpenLB 1.8 engine** while
keeping the real IP (city builder, receptor/exposure model, parameter space, viz). See
**`OPENLB_MIGRATION_PLAN.md`** for the full plan and **`OPENLB_STEP12_STATUS.md`** /
**`OPENLB_STEP3_STATUS.md`** for progress.

The pipeline is a three-stage swap of the *middle* box; Stages A and C are kept verbatim:

```
 STAGE A  scene build      city_builder7.h + city_zoning.h + voxelize.h
 (C++, no OpenLB)          + exposure_receptor.h/occupancy.h  →  material_map.dat,
                           source_mask.u8 (Ω), receptor_w.f32     [gen_openlb_geom.cpp]
        │
 STAGE B  flow + transport  ── being migrated ──
        │   NOW (works):  custom solver  lbm_solver + lbm_kernels (+CUDA)   [forward_city.cpp]
        │   NEXT (OpenLB): D3Q19+WALE + AD D3Q7 + deposition               [urban_flow.cpp]
        │
 STAGE C  exposure + viz    J = ⟨w,Θ⟩/|Ω|   +   visualize_forward.py, render_domain.py, city3d.py
```

### Migration status by step (plan §7)
- **Step 2 geometry bridge — DONE, gated.** `openlb_geometry.h` + `gen_openlb_geom.cpp`
  stamp OpenLB material numbers from the voxel grid; counts reconcile with the voxelizer
  at 4 m and 2 m. `geometry_loader.h` reads them back into a SuperGeometry.
- **Step 3 airflow — inlet verified, engine scaffolded.** The ABL/RFG inlet is ported and
  numerically verified (mean 0.10% vs log law, divergence 5.2%; `tests/abl_inlet_verify.cpp`).
  `urban_flow.cpp` + `abl_inlet_olb.h` are the OpenLB 1.8 airflow scaffold — **compile on
  your box** (flags marked `CONFIRM 1.8`); not yet built/gated.
- **Steps 1, 4–7 — pending** (OpenLB 1.8 install on the A4000; scalar+deposition; retire).

---

## Current production run (custom engine — still the working one)

```sh
# build the density-city forward solve (CPU/OpenMP)
g++ -O3 -std=c++17 -fopenmp -DCELL_SIZE_M=2.0 -c lbm_kernels_cpu.cpp -o kc.o
g++ -O3 -std=c++17 -fopenmp -DCELL_SIZE_M=2.0 -c lbm_solver.cpp      -o so.o
g++ -O3 -std=c++17 -fopenmp -DCELL_SIZE_M=2.0 forward_city.cpp kc.o so.o -o forward_city

COLL=hrr SCALAR=quick OUT_DIR=forward_city_out ./forward_city
python3 visualize_forward.py forward_city_out       # figures from the stored fields
```
`run_forward_city.sh` wraps the full build+run. `DUMP_GEOM_ONLY=1` stops after writing the
geometry/plan rasters (for the domain image). Every field the viz needs is written as a
self-describing binary (5-int header `[nx,ny,nz,dx*1000,ncomp]`) before any plotting;
`meta.txt` lists them.

## Generate a city only

```sh
g++ -O3 -std=c++17 -I. gen_density_city.cpp -o gen_density_city   # density-field zoning (current)
g++ -O3 -std=c++17 -I. gen_openlb_geom.cpp  -o gen_openlb_geom    # + OpenLB material map / Ω / receptor
```

---

## Layout

- **Active source** in the project root: city builder + zoning + voxelizer, receptor /
  exposure / infiltration / occupancy model, the custom LBM engine (kept as reference
  during migration), the OpenLB Stage-A/B files, the parameter space + optimizer, and the
  production viz suite.
- **`tests/`** — all unit tests, validation drivers, diagnostics, demos, and their plot /
  analysis scripts. Compile from the project root with `-I.` (e.g.
  `g++ -O3 -std=c++17 -I. tests/test_poiseuille.cpp -o test_poiseuille`).
- **`_to_delete/`** — quarantine for dead build artifacts and superseded docs. Review and
  `rm -rf` it yourself.

## Docs

`OPENLB_MIGRATION_PLAN.md` (the migration), `ARCHITECTURE.md` (full math), `PROJECT_MAP.md`
(file-by-file map), `CHANGES.md`, `HANDOFF.md`, and the topic notes: `LBM_SOLVER.md`,
`WALE_MODEL.md`, `HRR_ARCHITECTURE.md`, `EXPOSURE_METRIC.md`, `ADJOINT_SOLVER.md`,
`FORWARD_LIVE.md`, `INFILTRATION_MODEL.md`, `PARK_POROSITY.md`, `PARAM_SPACE_NOTE.md`, and
others. Test-suite docs: `TEST_SUITE.md`, `VALIDATION_ROSTER.md`, `VERIFICATION_AUDIT.md`.
