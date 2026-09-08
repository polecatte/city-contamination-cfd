# OpenLB port — status audit & lab-machine verification plan

Audit date: 2026-08-20. Scope: `OPENLB_MIGRATION_PLAN.md` §7 steps 1–4 as implemented in
`openlb_geometry.h`, `gen_openlb_geom.cpp`, `geometry_loader.h`, `abl_inlet_olb.h`,
`urban_flow.cpp`.

Method: (a) Stage A and the ABL inlet were **rebuilt and re-run from a clean tree** to
confirm the claimed gates reproduce; (b) every OpenLB API call in the Stage-B scaffold was
checked against the **OpenLB 1.8.1 Doxygen** (openlb.net, generated 2025-05-17), which is
the authoritative source — the 1.8 User Guide PDF is stale in several places and its code
listings do not compile against 1.8.

---

## 1. Verdict

**The port is at "Stage A done, Stage B written but never compiled."** That framing in
`OPENLB_STEP3_STATUS.md` is accurate and honest — but the gap between "scaffolded" and
"builds" is larger than the `CONFIRM 1.8` flags imply. `urban_flow.cpp` will not compile
against OpenLB 1.8 as written: roughly a dozen of its API calls refer to names that were
removed or renamed before 1.8, and two of them are not renames but genuine functionality
gaps that need a design decision.

Separately, three defects are **silent** — they would survive a successful compile and
produce plausible-looking but wrong numbers. Those are the ones that matter most, because
this project's whole output is a scalar ranking across designs; a wrong-but-finite `J` is
worse than a crash.

| Step (plan §7) | Claimed | Independently confirmed | Real state |
|---|---|---|---|
| 1 · Environment | blocked in cloud | — | **Not started.** No OpenLB on any machine you've told me about |
| 2 · Geometry bridge | done, gated | ✅ **rebuilt, gate PASSes, counts match to the cell** | **Done** |
| 3 · Airflow | inlet verified, engine scaffolded | ✅ inlet re-run, all 4 checks reproduce | **Inlet done; engine ~0% verified** |
| 4 · Scalar + deposition | scaffolded | n/a (needs OpenLB) | **Scaffolded, 3 substantive physics gaps** |
| 5–7 | pending | — | Pending |

### What I re-verified from scratch (all reproduced exactly)

```
gen_openlb_geom (dx=4 m)  → 177×167×89 = 2 630 751 cells
  MAT 1 fluid 2 445 872 · 2 wall 61 528 · 3 inlet 14 696 · 4 outlet 14 696
  5 slip 59 675 · 6 porous 4 725 · 7 ground 29 559        Ω = 22 812
  all 5 reconciliation identities PASS · exit 0
geometry_loader selftest  → round-trips every count · SELFTEST OK
abl_inlet_verify          → log-law 0.10% [PASS] · div 5.2% · σ 6.8% low · deterministic
```

The handshake files are also non-degenerate (checked, not just present):
`dep_vel.f32` 94 629 non-zero cells (max 2.0 mm/s), `receptor_w.f32` 34 781 non-zero
(max 0.513), `source_mask.u8` 22 812. Stage A is genuinely finished and trustworthy.

---

## 2. Defects found

### 2.1 Compile-blocking (mechanical — renames, ~2–3 h of work)

| # | Site | Problem | 1.8 fix |
|---|---|---|---|
| B1 | `urban_flow.cpp:429` | `olbInit(&argc,&argv)` — **removed** | `olb::initialize(&argc,&argv)` |
| B2 | `urban_flow.cpp:459-463` | `CuboidGeometry3D<T>` — **renamed** | `CuboidDecomposition3D<T>` |
| B3 | 8 sites (`:133,140,148,308,310,311,317`) | `instances::getBounceBack/getNoDynamics` + the `Dynamics*` overload of `defineDynamics` — the whole `olb::instances` namespace is **gone** | `sLattice.defineDynamics<BounceBack>(sGeom, MAT)` / `<NoDynamics>` |
| B4 | `:176,178,180,314` | `setInterpolatedVelocityBoundary`, `setInterpolatedPressureBoundary`, `setSlipBoundary`, `setZeroDistributionBoundary` — all **removed**, replaced by the declarative `olb::boundary` API | `boundary::set<boundary::InterpolatedVelocity<T,DESCRIPTOR>>(sLattice, sGeom, MAT_INLET)`; likewise `InterpolatedPressure`, `FullSlip`, `ZeroDistribution` |
| B5 | `:77` | `THETA` and `DEPOSIT` are used as descriptor fields but **never defined anywhere** | `struct THETA : public descriptors::FIELD_BASE<1> {};` (same for `DEPOSIT`), and drop the `descriptors::` qualifier at every `getField`/`setField` site |
| B6 | `:161` | `PorousBGKdynamics` on `WALED3Q19Descriptor` — that alias is `D3Q19<EFFECTIVE_OMEGA,VELO_GRAD>` and carries **no `POROSITY` field**, so the porous dynamics cannot instantiate | descriptor must be `D3Q19<EFFECTIVE_OMEGA,VELO_GRAD,POROSITY>` |
| B7 | `geometry_loader.h:81` | `block.get(x,y,z) = material` — `get()` returns `int` **by value** and is `const` | `block.set({x,y,z}, material)` |
| B8 | `:150-157` | `wallFunctionParam<T>` member names: `latticeWallDistance` is actually `latticeWalldistance` (lowercase d), `vonKarmanConst` is `vonKarman`, and **`z0` does not exist** | see §2.3 — not a rename |

Good news on the flags you were most unsure about: `WALED3Q19Descriptor`,
`WALEBGKdynamics`, `ConStrainSmagorinskyBGKdynamics`, `AdvectionDiffusionBGKdynamics`,
`PorousBGKdynamics`, `POROSITY`, `setWallFunctionBoundary`, `AnalyticalF3D<T,T>` with
`bool operator()(T[], const T[])`, the whole `UnitConverter` accessor set you used, and
`setProcessingContext(ProcessingContext::Evaluation)` are **all real, correct 1.8 names**.
The scaffold's instincts were right; it's the 1.4-era idioms that rotted.

### 2.2 Silent — compiles (or nearly), produces wrong numbers

**S1 · `getOrigin()` returns metres, not lattice indices.** Every block-iteration loop does

```cpp
const int gx0 = block.getOrigin()[0];   // ← physical SI metres, truncated to int
```

Doxygen for `BlockGeometry::getOrigin()`: *"the origin position given in SI units
(meter)."* With `dx = 4 m` every global index is off by a factor of ~4 and the geometry is
stamped scrambled. This pattern appears **7 times**: `stampSuperGeometry`, `exportLiveFlow`,
and all four `step4::` operators plus `gatherField`. In a single-cuboid serial run the
origin is `(0,0,0)` so it accidentally works — and then silently corrupts the moment you
go MPI or multi-block. Correct 1.8 idiom:

```cpp
auto& c   = cuboidDecomposition.get(iCglob);
auto& mc  = cuboidDecomposition.getMotherCuboid();
LatticeR<3> off = mc.getLatticeR(c.getOrigin());
```

**S2 · deposition velocity used in physical units where the closure needs lattice units.**
`urban_flow.cpp:381` computes `alpha = min(1, 8.0*vd)` with `vd` read straight from
`dep_vel.f32`, which `voxelize.h:65` documents as **m/s**. `CONTAMINANT_BC.md` specifies
α = 8·v_dep_**lb**. At the current operating point the conversion factor is dt/dx = 0.0125,
so the code over-deposits by **80×**: α = 1.6e-2 where it should be 2.0e-4. This inverts
the mass budget (everything deposits within metres of the source, nothing advects) and
would quietly destroy the exposure ranking. Note `w_s` two lines away *is* converted
correctly via `converter.getLatticeVelocity()` — so this is an oversight, not a convention.

**S3 · the AD relaxation rate is a placeholder that is never applied.**
`prepareScalarLattice` sets `const T omegaAD = 1.0; // placeholder` and then `(void)omegaAD`
— it is discarded. If the lattice ends up running at ω=1 the D3Q7 diffusivity is
D = (1/ω−0.5)/4 = 0.125 lu → **40 m²/s physical**, some 100× the turbulent diffusivity you
want (ν_t/Sc_t ~ O(0.1–1 m²/s)). The plume would be pure diffusion. ω must be derived from
D_eff = D_mol + ν_t/Sc_t, and since ν_t is spatially varying under WALE it has to be a
per-cell effective ω, not a lattice constant.

### 2.3 Design gaps — need a decision, not a fix

**G1 · There is no aerodynamic-roughness (z₀) wall function in OpenLB 1.8.** This is the
important one, because corrective C2 — the *stated fix* for the ~35 % ABL horizontal-
homogeneity drift the old solver suffered — is built on passing `z0` to
`wallFunctionParam`. That member does not exist. Neither does it exist in 1.8's newer
`WallModelParameters<T>` / `setTurbulentWallModel`, which is a smooth-wall Musker /
law-of-the-wall model. So the single largest physics justification for the migration is
**not available off the shelf**. Options, cheapest first:

1. Run `GROUND_BOUNCEBACK=1` (already wired), measure the actual drift, and only fix it if
   it exceeds the 10 % gate. The old solver's drift came from a no-slip floor at dx=4 m;
   OpenLB's is the same physics, so expect similar. Cheap to measure, ~1 h.
2. Use `setTurbulentWallModel` with `wallFunctionProfile` = Musker and tune
   `latticeWallDistance` / `samplingCellDistance` to reproduce a z₀=0.045 m log law at the
   first fluid node. A calibration, not new physics. ~1 day.
3. Write a custom rough-wall post-processor enforcing u(z₁) = (u*/κ)·ln(z₁/z₀). This is the
   physically right answer and is a genuinely small operator (~60 lines), but it must be
   GPU-side for the A4000. ~2 days.

Recommend (1) as the Phase-5 gate measurement, then (3) if it fails — (2) tends to
consume more time than (3) for a worse result.

**G2 · The Step-4 burst loop is architecturally CPU-only and prohibitively slow.**
As written, each burst timestep does **six full-domain host sweeps** (`couple`, `inject`,
`deposit`, `accumulateTheta` × block loops) over 2.63 M cells. At the default
`MAX_BURST_STEPS = 20 × 3540 = 70 800` steps that is ~10¹² host cell-visits, and on GPU
every sweep would need a full device→host→device round-trip per step. The header already
flags this (§6.3) — flagging it isn't enough; it means Step 4 **cannot be gated on the
city**, only on a small box, until the operators are rewritten as OpenLB post-processors.
`couple` in particular should just be
`NavierStokesAdvectionDiffusionVelocityCoupling` via `SuperLatticeCoupling`.

**G3 · The sponge layer can't get a different C_s.** `SmagorinskyBGKdynamics` takes its
Smagorinsky constant from `collision::LES::SMAGORINSKY`, which is a **lattice-global**
parameter set with `sLattice.setParameter<>()` — you cannot raise it on MAT_SPONGE alone.
Use `ExternalSmagorinskyBGKdynamics` (C_s from a per-cell field) so the fringe can be
graded, which is what corrective C5 actually describes.

### 2.4 Smaller items

- **The preflight (C1) is a false-negative sieve.** At the configured operating point
  τ = 0.5000001 — it passes `tau > 0.5` by 1.4e-7 while being, physically, *zero molecular
  viscosity*. BGK at τ→0.5 with no SGS contribution is unconditionally unstable. Tighten
  the gate to `tau > 0.505` and switch to `UnitConverterFromResolutionAndRelaxationTime`
  with τ ≈ 0.51–0.56 so dt is derived rather than guessed. (Re = 9.5e7 at charL = 356 m, so
  essentially all viscosity is subgrid — that's expected for urban LES, but it makes the
  operating point fragile rather than safe.)
- **The 1:1 index assumption is unverified and load-bearing.** `IndicatorCuboid3D(extent,
  origin)` + `CuboidDecomposition(indicator, dx, 1)` may yield nx or nx+1 nodes per axis.
  Everything downstream assumes exact 1:1. Use the explicit-extent constructor
  `CuboidDecomposition3D<T>(origin, dx, Vector<int,3>{nx,ny,nz}, 1)` and remove the
  ambiguity entirely.
- **No `superGeometry.communicate()` after stamping** — halo material numbers stay
  unstamped, so any neighbour test near a block edge is wrong under MPI.
- **Overlap is 2**; the 1.8 default is 3, and interpolated boundaries + WALE velocity
  gradients want it. Use 3 unless you have a reason.
- **`meta_flow.txt` mislabels the budget:** `mass_drained = emit − dep` double-counts the
  still-airborne mass. The CSV gets it right (`emit − dep − air`).
- **ν_t is exported as literal 0** in `umean_full.f32` component 4, so Stage C's eddy-
  viscosity panels will render blank.
- `dep_vel.f32` is written **twice** in `gen_openlb_geom.cpp` (lines 108 and 167) — harmless.
- **No Makefile exists** for the OpenLB app directory. Phase 1 below supplies it.
- **The Step-4 acceptance gate (§6.2 linearity guard) is not implemented anywhere** in the
  repo — not in `tests/`, not in `urban_flow.cpp`. It needs writing before Step 4 can pass.

---

## 3. Verification plan for the lab machine

Design principles: (i) **never debug two things at once** — get OpenLB itself green before
your code, get your code green on CPU before GPU, get physics green on a 40³ box before the
city; (ii) every phase has a numeric pass criterion, not "it ran"; (iii) each phase is
resumable over SSH — assume the connection drops, so everything long goes under `nohup`/
`tmux` with a log.

Assumed target: **OpenLB 1.8.1**, not 1.9. 1.9 (Dec 2025) refactored all 138 examples into
a new "case style" and its API is not covered by any published Doxygen yet; 1.8.1 is the
release everything above was verified against. Move to 1.9 after the port is green, if ever.

### Phase 0 — environment inventory (15 min)

```bash
ssh lab
nvidia-smi                       # GPU present? which arch?
gcc --version; nvcc --version    # need C++20-capable gcc; nvcc for GPU
mpirun --version                 # optional
df -h ~                          # need ~5 GB for the tree + build
curl -sI https://zenodo.org      # does the lab box have outbound internet?
nproc; free -g
```

**Gate:** you know whether the box has (a) a GPU and its SM number, (b) internet. If no
internet, download `olb-1.8.1` elsewhere and `scp` it in — the tarball is ~150 MB.

### Phase 1 — build OpenLB and prove it works before touching your code (1–2 h)

```bash
curl -L -o olb-1.8.1.tar.gz \
  "https://zenodo.org/records/15440776/files/release-1.8.1.tar.gz?download=1"
tar xzf olb-1.8.1.tar.gz && cd release-1.8.1     # (check the extracted dir name)

# CPU-ONLY first — always. Copy a template from config/ and edit:
cp config/default.mk config.mk   # or whichever CPU template ships
#   CXX := g++          CC := gcc
#   CXXFLAGS := -O3 -std=c++20
#   PARALLEL_MODE := NONE
#   PLATFORMS := CPU_SISD
#   FLOATING_POINT_TYPE := double
make -C external && make -j$(nproc) 2>&1 | tee ../build_lib.log

cd examples/turbulence/nozzle3d && make && ./nozzle3d 2>&1 | tail -20
```

**Gate 1:** `nozzle3d` builds and runs to completion, writes VTK, no NaNs.
*If this fails, stop — it is an OpenLB/toolchain problem and nothing downstream can
diagnose it.* Keep `build_lib.log`; a library that took 40 min to build is worth not
rebuilding.

Then create the app directory and confirm the build system reaches it:

```bash
cd $OLB_ROOT
mkdir -p examples/urban/urban_flow
cp examples/turbulence/nozzle3d/Makefile examples/urban/urban_flow/
# edit: OUTPUT := urban_flow   and fix ROOT := ../../..  to match the nesting depth
cd examples/urban/urban_flow
printf '#include "olb3D.h"\n#ifndef OLB_PRECOMPILED\n#include "olb3D.hh"\n#endif\nint main(int argc,char**argv){olb::initialize(&argc,&argv);return 0;}\n' > urban_flow.cpp
make && ./urban_flow && echo HELLO-GATE-OK
```

**Gate 2:** a trivial `main()` in *your* directory compiles and links. This isolates the
build-system question from the 12 API breaks — do not skip it.

### Phase 2 — Stage A on the lab box (20 min)

```bash
cd ~/urban_openlbm
g++ -O3 -std=c++17 -I. -DCELL_SIZE_M=4.0 gen_openlb_geom.cpp -o gen_openlb_geom
OUT_DIR=geom_out ./gen_openlb_geom | tee geom_gate.log
g++ -O2 -std=c++17 -DGEOMLOADER_SELFTEST -x c++ geometry_loader.h -o geomloader_test
./geomloader_test geom_out
```

**Gate 3 (exact numbers to match — I reproduced these today on a clean tree):**

```
2 630 751 cells · fluid 2 445 872 · wall 61 528 · inlet 14 696 · outlet 14 696
slip 59 675 · porous 4 725 · ground 29 559 · Ω 22 812 · all identities PASS · exit 0
```

Any deviation means a compiler/environment difference, not a code change — chase it now
while the answer is cheap.

### Phase 3 — mechanical API port to compile-green (half a day)

Apply B1–B8 from §2.1. Work in this order, rebuilding after each — the errors cascade and
fixing them in dependency order is much faster than batching:

1. B1 `olb::initialize`, B2 `CuboidDecomposition3D` — the file won't even preprocess otherwise.
2. B5 define `THETA`/`DEPOSIT` — needed before the AD descriptor instantiates.
3. B3 all eight `defineDynamics` sites, B6 the descriptor's `POROSITY` field.
4. B4 the four boundary calls → `boundary::set<...>`.
5. B7 `block.set({x,y,z}, mat)`.
6. B8 wall function → for now just take the `GROUND_BOUNCEBACK=1` branch and `#if 0` the
   wall-function block; G1 gets decided in Phase 5 on evidence.
7. Also here: switch to the explicit-extent `CuboidDecomposition` ctor, raise overlap to 3,
   add `superGeometry.communicate()` after stamping, and fix the seven `getOrigin()` sites
   (S1) using the `mc.getLatticeR(c.getOrigin())` idiom.

**Gate 4:** `make` produces `urban_flow` with zero errors. Comment out the Step-4 block
entirely for now (`#if 0` from line ~509 to the field exports) — airflow first.

### Phase 4 — geometry round-trip: the highest-value cheap test (1 h)

Before any physics, prove the material map arrived intact. Add ~10 lines after
`stampSuperGeometry`:

```cpp
superGeometry.communicate();
superGeometry.getStatistics().print();
for (int m=0; m<=8; ++m)
  clout << "MAT " << m << " olb=" << superGeometry.getStatistics().getNvoxel(m) << std::endl;
```

**Gate 5:** OpenLB's own per-material voxel counts equal the Phase-2 histogram **exactly**
(modulo the sponge cells carved out of FLUID: expect `fluid − 8·ny·nz` and `sponge = ` that
same number). Then run `python3 tests/render_material_map.py geom_out` and eyeball the
slices.

This one gate simultaneously catches the `getOrigin()` unit bug (S1), any nx-vs-nx+1
off-by-one, and any padding/overlap indexing error — the three things most likely to
silently poison everything downstream. **Do not proceed past a mismatch here.**

### Phase 5 — airflow physics, on isolated cases first (1–2 days)

Do *not* start with the city. Two targeted cases through the same code path:

**5a · Empty-domain ABL fetch (isolates the wall treatment).** Generate a geometry with
`PARK_FRAC=0 POP=0` or hand-build an all-fluid map with a ground plane. Run 3 flow-throughs.
Extract the mean streamwise profile at x = 0, ¼L, ½L, ¾L.

- **Gate 6a: mean-shear drift from inlet to city face < 10 %.** The inlet reproduces the
  target log law to 0.10 %, so *any* drift you see is the solver + floor treatment. Record
  the number under `GROUND_BOUNCEBACK=1` — that measurement is the input to the G1 decision.

**5b · Single cube, the check the old solver failed.** One cube of height H in an otherwise
empty domain, H/dx ≥ 10 (so at dx=4 m, H ≥ 40 m; or drop to dx=2 m). Measure wake
reattachment length.

- **Gate 6b: Xr/H in 1.4–1.8** (target band ≈1.6). This is the credibility gate for the
  whole migration — it is the specific failure that motivated leaving the custom solver.

Also verify at 5a: `converter.print()` shows τ ≥ 0.505 after the converter change, `maxU`
stays < 0.1 lu throughout, and the run doesn't trip the C4 guard.

**Only after 6a and 6b pass** run the full city (`3` flow-throughs ≈ 10 620 steps,
27.9 G cell-updates — minutes on GPU, ~20–30 min on a many-core CPU) and confirm
`umean_full.f32` opens in `visualize_forward.py`.

### Phase 6 — scalar + deposition, on a 40³ box (2–3 days)

Re-enable Step 4, but gate it on a **tiny synthetic geometry**, never the city — see G2.
Build a 40×40×40 all-fluid map with uniform velocity and no buildings.

1. Fix S2 first (`vd_lb = vd * dt/dx`) and S3 (derive ω_AD from D_eff; start with a
   *constant* D_eff = D_mol + ν_t̄/Sc_t and only make it per-cell once the constant case
   passes).
2. **Gate 7a — mass budget.** Release a burst, run to clearance.
   `|emit − (dep + out + air)| / emit < 1 %`. Fix the `meta_flow.txt` label while here.
3. **Gate 7b — linearity guard (§6.2, currently unimplemented — you must write it).**
   Release from a single cell → Θ₁. Release from the full Ω → Θ_Ω. Then for a random sample
   of ~20 receptor points, Θ_Ω(x) must equal Σᵢ Θᵢ(x) to within solver tolerance. The cheap
   version: run two single-cell releases and one two-cell release and check
   `Θ_{a+b} = Θ_a + Θ_b` to < 0.1 %. **This is the gate that protects the exposure metric** —
   if superposition doesn't hold, `J` is not the quantity `EXPOSURE_METRIC.md` defines.
4. **Gate 7c — deposition sanity.** With `v_d = 0` and `w_s = 0`, deposited mass must be
   *exactly* zero and all mass must exit the outlet. With the real `dep_vel.f32`, the
   deposited fraction should land in single-digit-to-low-tens percent, not ~100 %
   (which is what the S2 bug produces).

### Phase 7 — end-to-end and Stage C (1 day)

Full city, burst over Ω, then the contraction. **Gate 8:** `J = ⟨w,Θ⟩/|Ω|` is finite,
positive, and *stable* — run the same city with two different RNG seeds on the inlet and
confirm `J` reproduces to within a few percent. A metric whose seed-to-seed scatter exceeds
the design-to-design signal cannot rank designs, and `RANKING_STABILITY_NOTE.md` already
frames this concern for the old engine; it must be re-established for the new one.
Then `visualize_forward.py` + `render_domain.py` on the output directory.

### Phase 8 — GPU (only after Phase 7 is green on CPU)

```make
CXX := nvcc
CC  := nvcc
CXXFLAGS := -O3 -std=c++20 --forward-unknown-to-host-compiler
PLATFORMS := CPU_SISD GPU_CUDA
CUDA_CXX  := nvcc
CUDA_ARCH := 86        # A4000 = Ampere sm_86; check nvidia-smi for the actual box
PARALLEL_MODE := NONE  # MPI only for multi-GPU
```

`make -C external CXX=nvcc CC=nvcc` then rebuild. Rerun **Gates 5, 6a, 6b, 7a, 7b** on GPU
and diff against the CPU numbers — CPU/GPU drift is exactly the class of bug that bit the
custom engine, so treat parity as a gate, not a formality. This phase also requires the G2
rewrite: `couple` → `NavierStokesAdvectionDiffusionVelocityCoupling`, and `injectBurst` /
`deposit` / `accumulateTheta` → on-device post-processors.

---

## 4. Honest effort estimate

| Phase | Work | Elapsed |
|---|---|---|
| 0–2 | environment, library, Stage A | 0.5 day |
| 3–4 | API port to compile-green + geometry gate | 1 day |
| 5 | airflow gates (+ G1 decision) | 1–3 days |
| 6 | scalar/deposition gates + writing the linearity guard | 2–3 days |
| 7 | end-to-end + seed stability | 1 day |
| 8 | GPU + on-device operators (G2) | 3–5 days |

**≈ 2 weeks of focused work to a GPU-production port**, of which the genuinely uncertain
parts are G1 (roughness wall function) and G2 (on-device operators). Everything else is
bounded.

## 5. If you want to shorten the path

The single highest-leverage change to the plan: **do Phases 3–4 and Gate 6a before
committing to anything else.** Gate 6a tells you whether OpenLB's floor treatment actually
fixes the ABL drift that motivated the migration. If it does, the rest is engineering. If
it doesn't, you're facing G1's custom wall function anyway — and it is worth knowing that
on day 3 rather than day 10.
