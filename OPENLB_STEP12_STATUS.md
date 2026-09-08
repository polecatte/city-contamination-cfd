# OpenLB migration — Steps 1–2 status

This note records progress on the first two (non-destructive) steps of
`OPENLB_MIGRATION_PLAN.md` §7. Nothing in the retire list (§5) was touched.

## Step 2 — Geometry bridge  ✅ COMPLETE (gate passed)

New files (Stage A → Stage B):

- **`openlb_geometry.h`** — stamps OpenLB `SuperGeometry` material numbers directly
  from the existing `VoxelGrid` (cell-for-cell, no STL round-trip). Writes
  `material_map.dat` (5-int header `[nx,ny,nz,dx*1000,1]` + `int32` payload — the
  project's existing self-describing convention). Includes a reconciliation routine
  that is the Step-2 gate.
- **`gen_openlb_geom.cpp`** — Stage-A driver. Replicates `forward_city.cpp`'s scene
  build exactly (same density city, buffers, solid-building voxelization, and Ω rule),
  then emits `material_map.dat`, `source_mask.u8`, `geom_type.u8`, `meta_geom.txt`.
- **`render_material_map.py`** — visual check (plan slices + vertical section).

Material scheme (NSE / airflow lattice):

```
0 donothing   1 fluid   2 wall(ground+solid buildings)
3 inlet       4 outlet  5 slip(lateral+top)            6 porous(park canopy)
```

**Design decision — Ω is not a material.** The burst-release source set Ω stays in its
own `source_mask.u8` array, consumed by the advection–diffusion lattice's source
post-processor in Stage B. This lets a single cell be both an NSE velocity-inlet cell
and a scalar-release cell without conflict, and keeps `material_map.dat` a pure
statement of the airflow geometry. (Matches the plan: Stage B "reads material_map +
source_mask" as two independent inputs.)

**Gate result — material counts match the old voxelizer**, at both resolutions:

| identity | dx=4 m (2.63 M cells) | dx=2 m (21.05 M cells) |
|---|---|---|
| WALL = ground + solid shells | 91 087 = 91 087 ✅ | 626 531 = 626 531 ✅ |
| POROUS = park shells + indoor | 4 725 = 4 725 ✅ | 31 500 = 31 500 ✅ |
| fluid+inlet+outlet+slip = FLUID | 2 534 939 = 2 534 939 ✅ | 20 391 694 = 20 391 694 ✅ |
| total cells conserved | 2 630 751 ✅ | 21 049 725 ✅ |

Build & run:

```bash
g++ -O3 -std=c++17 -DCELL_SIZE_M=2.0 gen_openlb_geom.cpp -o gen_openlb_geom
OUT_DIR=geom_out ./gen_openlb_geom            # exits 0 iff the gate passes
python3 render_material_map.py geom_out       # -> geom_out/material_map.png
```

Wind direction is honoured: `WIND_DEG` snaps the inlet/outlet to the nearest axis
(0°→inlet x=0, 90°→y=0, …). All `forward_city` knobs (`CITY_M`, `POP`, `PARK_FRAC`,
`PEAK_H`, buffers, …) are env-overridable so the geometry can be made identical to any
production run.

## Step 1 — OpenLB 1.8 environment  ⛔ BLOCKED in this cloud session

The cloud container's egress proxy allowlists only package registries + github.
`openlb.net`, `zenodo.org`, and `gitlab.com` are all hard-blocked (403 CONNECT tunnel /
connection reset), so **OpenLB 1.8 cannot be downloaded here**. The only reachable copy
is the stale GitHub mirror `github.com/openLB/openLB` (last commit 2019-12-02, ~v1.3) —
it has `nozzle3d`, WALE, and CPU LBM, but predates the unified `Platform::GPU_CUDA`
backend that the plan's GPU-on-A4000 story requires, and its dynamics/geometry API
differs enough from 1.8 that custom operators written against it would need rewriting.
So it is not a useful stand-in for the 1.8 gate.

This is the environment reality the plan's §8 anticipated: the loop is **you fetch +
build 1.8 on the A4000, I write + CPU-verify the bridge code here**. The geometry
bridge above is written to be version-independent, so it is ready regardless.

### Recipe to clear the Step-1 gate on the A4000 (or any box with real internet)

```bash
# 1. get OpenLB 1.8 (no registration needed)
curl -L -o olb-1.8r0.tgz "https://zenodo.org/records/15270117/files/olb-1.8r0.tgz?download=1"
tar xzf olb-1.8r0.tgz && cd olb-1.8r0

# 2. configure config.mk — GPU build for the A4000 (Ampere, sm_86):
#      CXX            := nvcc
#      CC             := nvcc
#      PLATFORMS      := CPU_SISD GPU_CUDA
#      CUDA_ARCH      := 86
#      PARALLEL_MODE  := NONE        # or MPI for multi-GPU
#    (a CPU-only smoke build instead: CXX:=g++, PLATFORMS:=CPU_SISD, drop CUDA_ARCH)

# 3. build the library + the canonical turbulent example
make -j
cd examples/turbulence/nozzle3d && make && ./nozzle3d
#   GATE: nozzle3d builds and runs on both CPU and the A4000 GPU build.
```

Confirm the exact `config.mk` variable names against the OpenLB 1.8 User Guide (Aug
2025) — I could not fetch it from this session, so treat the block above as a template,
not a verified transcript. Once `nozzle3d` runs, Step 3 (airflow-only: NSE + WALE +
ABL/RFG inlet, gated on `Xr/H` and ABL drift) is the next move, feeding on the
`material_map.dat` this bridge already produces.

## What I need from you

Whether OpenLB 1.8 is already installed on the A4000 and at what version (the plan's
step-1 blocking question), and — separately — if you want me to also build & smoke-test
the 2019 CPU mirror here just to exercise the toolchain (low value given the API drift,
so I skipped it by default).
