# OpenLB port — Phase 8: the GPU build

Date: 2026-09-29, updated 2026-10-29. Status: **runs on the lab's RTX A4000** (§5): gates 5, 6a,
6b (dx 4 and 2), 7b, 8, the wake check and the production city ran on the GPU build with the
same gate numbers as the CPU build, at ~100 MLUPS (BGK + WALE, double) against 11–19 MLUPS on
the lab CPU. `GPU=1 ./lab_openlb.sh parity` has **not** been recorded yet and is the first thing
to run on the current code (§6 changed the WALE arithmetic). §6 is the speed-up: an opt-in
float lattice and exact reductions of the FP64 work, with the tests that decide whether the
float build may be used.

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
3. The `gpu_only.mk` template is single precision; `olbconfig.sh gpu` sets double by default
   (the CPU reference is double), float with `PRECISION=single` (§6).
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

## 5. Lab results (2026-10, RTX A4000, double)

| | CPU build (lab, OpenMP) | GPU build |
|---|---|---|
| 6a drift at the compact face (40 m) | 10.9 % | 10.9 % |
| 6b Xr/H, dx 4 / dx 2 | 2.63 / 2.31 | 2.63 / 2.31 |
| 8 seed spread of J | 0.44 % | 0.51 % |
| wake reversed flow, last 15 % | 0.0 % | 0.0 % |
| throughput (Timer, MLUPS) | 11–19 | 97–102 (HRR: 38–41) |

Gate 6a is now decided at the production city face (500 m, `tests/gate6_analyze.py`), where the
lab's runs drift ~4–5 %; the 40 m compact face is reported with a warning. The identical 6a/6b
numbers are the time means of turbulent runs, so they are the strongest evidence so far that
the device path is right; the parity test is still the one that checks it directly.

## 6. Speed-up (2026-10-29)

**Where the time goes.** The A4000 runs FP64 at 1/32 of its FP32 rate (~0.6 vs 19.2 TFLOPS)
with 448 GB/s of memory bandwidth. A double D3Q19 step with WALE moves ~300 B per cell, which
bandwidth alone would allow at ~1.5 GLUPS; 100 MLUPS means the step is bound by FP64
arithmetic, not memory: per cell and step, three `pow` (each a log/exp pair) in the WALE
viscosity, nine divisions and up to 30 neighbour mask reads in the gradient, and the inlet's
100 RFG modes of sin/cos on the inlet plane. Six full-lattice passes per burst step (NSE
collide+stream, WALE velocity, WALE gradient, coupling, AD collide+stream, Θ).

**Changes** (all keep the double build's behaviour except where noted):

1. *WALE gradient stencil decided once.* Which difference `WaleGradOp` takes along each axis
   (8th-order, central, one-sided, zero) depends only on the static fluid flags, so
   `DeviceNSE::init` decides it (`stencilRule`) and stores it in `UF_MASK` bits 8–16. The
   operator reads one mask instead of up to 30 and each neighbour velocity once. Same branch,
   same arithmetic: bitwise identical (the CPU STRICT parity test must stay at 0).
2. *WALE powers through sqrt.* `x^2.5 = x² √x`, `x^1.25 = x √√x`, `x^1.5 = x √x` instead of
   `pow`. Changes the results by round-off only: `tests/wale_verify/pow_sqrt_check.cpp`, 182 k
   random gradients, ω_eff agrees to 3e-16 (double) and 1.6e-7 (float). Both paths share the
   collision, so CPU parity is unaffected; the GPU-vs-CPU tolerances are unchanged.
3. *Float lattice, opt-in.* `typedef FLOATING_POINT_TYPE T` (was `double`); `PRECISION=single
   ./olbconfig.sh gpu`, or `GPU=1 PRECISION=single ./lab_openlb.sh setup`, builds a float tree
   under `$WORK/gpu_sp`. Accuracy rests on OpenLB storing populations shifted by their weights
   (f − t_i), which keeps a float lattice close to double (Lehmann et al. 2022, *Phys. Rev. E*
   106, 015308). What must not be float stays double: Θ, deposition, the face fluxes and the
   time-mean sums are `TYPED_FIELD_BASE<double>` fields (a float Θ would round away its ~1e5
   small increments), and every host sum is double. The scalar is resolved to ~1e-7 absolute
   in float (C = Σf̃ + 1), so Q_RATE stays ~1 (J does not depend on it; a warning fires below
   0.1). Default stays double: the CPU reference and the parity test are double.
4. *Measurement.* `STAGE_TIMING=1` prints wall time per stage (device synchronised around each);
   every run prints spin-up MLUPS and burst ms/step and writes `precision`, `spinup_wall_s`,
   `burst_wall_s` to its meta files. `lab_openlb.sh speed` runs 2000 + 2000 steps of the
   production city with stage timing, per build.

**Not done, deliberately.** Per-field host pulls at report steps: the populations live in the
descriptor's own storage, which a per-field context switch can silently skip, so reports keep
the whole-lattice pull (one per 100 burst steps, `CLEAR_EVERY`); `speed` shows its share.
Fusing the WALE velocity pass into the coupling would save one pass in the burst but lags the
gradient by a step on the inlet plane; worth it only if `speed` says that pass matters.

**What decides whether the float build is used** (`LAB_RUNBOOK_OPENLB.md`):
- `GPU=1 ./lab_openlb.sh parity` (double) — the device path, after changes 1–2.
- `GPU=1 PRECISION=single ./lab_openlb.sh gates` — 5, 6a, 6b, 7a/b/c and the plume test in float.
- `GPU=1 ./lab_openlb.sh precision` — three designs at dx 8: double at two inlet seeds, float at
  one. A float run is another turbulent realisation after a few flow-throughs, so it is judged
  against the double build's own seed noise: each design within 3√2 σ, no systematic offset
  beyond 2√2 σ/√n, float budget closure < 1e-3 (`tests/precision_report.py`). It also prints
  the speed-up per design.
- `speed` for both builds.

None of this was compiled here (no OpenLB build in the cloud container); the lab build is the
compile check, and `send` collects the build errors if there are any.
