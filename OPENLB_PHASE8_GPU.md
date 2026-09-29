# OpenLB port — Phase 8: the GPU build

Date: 2026-09-29. Branch `claude/gracious-lovelace-3wlllj`. Status: **compiles for CUDA and is
verified on CPU; not yet run on a GPU** (the cloud container has none). The lab box's A4000 is
the first GPU it will touch; `GPU=1 ./lab_openlb.sh parity` is the test that decides it
(`LAB_RUNBOOK_OPENLB.md` §7).

## 1. What changed

Phases 5–7 added per-step work that ran as host loops over the lattice: WALE velocity-gradient
refresh, rough-wall floor, Richards–Hoxey top stress, the ABL/RFG inlet's `defineU` sweep, the
time mean, and the Step-4 flux / inject / deposit / Θ / settling loops. On a GPU build those
loops would read stale host copies (OpenLB keeps the lattice on the device) or force a full
device↔host copy every step.

`urban_ops.h` re-implements each of them as an OpenLB operator (`any_platform`, so the same code
is an OpenMP loop on CPU and a CUDA kernel on GPU), run through custom stages
(`executePostProcessors(ustage::…)`) in exactly urban_flow's order. Each is a line-for-line
transcription of the host loop it replaces, registered on exactly the same cells (the host
builds the cell lists once — `RoughWall::init`, `TopStress::init`, `step4::Ops::init` — and
registers them per cell).

| host reference (`urban_flow.cpp`) | operator (`urban_ops.h`) |
|---|---|
| `setBoundaryValues` (inlet `defineU`) | `InletOp` — RFG modes in a device array, `cell.defineU` |
| `VeloGradRefresh` (2 sweeps) | `WaleVelOp` → `WaleGradOp` (8th-order / 2nd / 1st, same branches) |
| `RoughWall::apply` (snapshot, remap + stress) | `RwSaveOp` → `RwApplyOp` |
| `TopStress::apply` | `TopOp` |
| `TimeMean::sample` | `TMeanOp` (sums on device; pulled once at the end) |
| `Ops::faceFlux / inject / deposit / accumulate / settle` | `AdFluxOp`, `AdInjectOp`, `AdDepositOp`, `AdThetaOp`, `AdSettleOp` |

Both paths stay in the binary. `HOST_OPS=1` (default on CPU) runs the host loops — they are
~2× faster on CPU, because OpenLB's CPU operator dispatch resolves dynamic fields per access —
and `HOST_OPS=0` (the only choice on GPU) runs the operators.

**Data residency.** After start-up the device is the truth. The host only pulls
(`setProcessingContext(Evaluation)`) at check/report steps and at the end: divergence guard,
time-series sums, exports. It never pushes lattice state while stepping, because OpenLB's
`Simulation` context copies whole arrays host→device with no dirty tracking — a push after the
device has advanced would silently roll it back. Start-up fields are pushed per field
(`setProcessingContext<Array<FIELD>>`); parameters (`setParameter`) upload synchronously and are
set only after their operator is registered (the OMEGA lesson of the CPU port, which bit again
here: inlet parameters set first reached no operator and every inlet cell went NaN).

## 2. Verification done here

**CPU, operators vs host loops, bitwise** — `tests/device_parity.sh` with `STRICT=1`, single
thread, four cases that between them execute every operator:

| case | what it exercises | outputs compared | max rel. diff |
|---|---|---|---|
| abl | inlet, WALE gradient (+ its check against OpenLB's functor), rough wall, top stress | live field, peak \|u\| | **0** |
| cube | + time mean | live field, time mean + variance | **0** |
| box7 | frozen wind: flux, inject, deposit, Θ | Θ, deposition, all time-series columns, budget | **0** |
| boxlive | live NSE + coupling + settling + scalar | as box7 | **0** |

37 comparisons, all identical to the last bit.

**Single thread, because** a multi-threaded CPU run is not bit-reproducible even against
itself: two 4-thread runs of the same path differ at ~1e-12 in roughly 1 run in 8. Per-material
checksums put the first difference every time in `MAT_OUTLET` at the last x-plane, which none
of our code writes: it is OpenLB's `InterpolatedPressure` outlet post-processor, whose
neighbour reads race under OpenMP. Rounding-level, OpenLB's own, harmless — but it is why the
bitwise test is single-threaded. (The operator path happened never to show it in 12 runs.)

**GPU compile** with CUDA 12.6 (nvcc + g++ 13, `CUDA_ARCH` 86 = A4000, double precision): clean.
Re-compiled with OpenLB's suppression of "host function called from device code" (20014/20011)
removed: no such call reached from any of our operators on the device path.

**Tolerance calibration.** GPU and CPU round differently: nvcc contracts `a*b+c` into fused
multiply-adds, and device sin/cos/log differ from glibc in the last ULPs. As a stand-in, the
same CPU code was built with `-ffp-contract=off` (g++ contracts to FMA by default in C++:
10 068 `vfmadd` in the normal binary, 0 in this one) and run on the operator path against the
normal build's host loops, 4 threads each — the way the lab runs CPU vs GPU:

| case | worst output | max rel. diff | tolerance |
|---|---|---|---|
| abl (400 steps) | live field | 2.4e-8 | 1e-4 |
| cube (600 steps) | live field / time mean | 3.3e-8 / 2.5e-8 | 1e-4 |
| box7 (linear scalar) | Θ | 9.5e-14 | 1e-9 |
| boxlive (NSE + scalar) | live field; Θ; budget terms | 3.6e-6; 6.3e-8; ≤ 3.7e-7 | 1e-4 |

So rounding alone moves these short runs by ≤ 4e-6; the tolerances leave ~30× (flow) and
~10⁴× (linear) headroom for the GPU's different math library, while a real device bug —
a stale field, a missed cell, a wrong parameter — shows up as O(1e-2) or worse.

## 3. Toolchain findings (all handled by `olbconfig.sh gpu` / `lab_openlb.sh`)

1. **CUDA 12.0 cannot compile OpenLB 1.8** — not ours, OpenLB's stock examples too:
   `fields.h(36): call to consteval function std::source_location::current did not produce a
   valid constant expression`. CUDA 12.6 works. Ubuntu 24.04's `nvidia-cuda-toolkit` is 12.0,
   so the lab script installs CUDA 12.6 from conda-forge (micromamba, no root) when `nvcc` on
   PATH is older than 12.4.
2. **OpenLB's GPU link line has no arch**, so with `-rdc=true` nvcc device-links for its default
   sm_52 and the executable holds **no sm_86 kernels** (`nvlink warning: SM Arch ('sm_52') not
   found`; `cuobjdump --list-elf` shows only an empty sm_52 image). It would fail at the first
   kernel launch. `olbconfig.sh gpu` adds `--generate-code=arch=compute_$(CUDA_ARCH),code=sm_$(CUDA_ARCH)`
   to `LDFLAGS`; `lab_openlb.sh` checks the built binary with `cuobjdump`.
3. The `gpu_only.mk` template is single precision; set to double (the app computes in double
   and the CPU reference is double).
4. A non-system CUDA gets an rpath, so the binary loads its own `libcudart`, not the distro's
   older one.
5. A namespace-scope constant table (`RW_DH`) is host memory, unreadable in a kernel; replaced
   by a `constexpr` function.

## 4. What is still unverified, and how the lab settles it

Nothing here has executed on a GPU. The risks that only a GPU run can clear: device-side
dynamic-field lookup, the RFG-mode device array, kernel-launch ordering between stages (OpenLB
synchronises at every `executePostProcessors`, and the code relies on that), and GPU memory
(~5 GB for the city in double; the A4000 is shared with the desktop).

`GPU=1 ./lab_openlb.sh parity` runs the four cases with the CPU build as reference and the GPU
build as test: `box7` must agree to 1e-9, the flow cases to 1e-4, and the WALE gradient check
must PASS on the GPU (it compares the device-computed gradient against OpenLB's own functor).
If parity passes, the GPU gates and city runs follow and `./lab_openlb.sh compare` puts their
numbers next to the CPU ones; the turbulent statistics (6a drift, 6b Xr/H, gate 8 J) should
agree within their seed-to-seed spread rather than bitwise.
