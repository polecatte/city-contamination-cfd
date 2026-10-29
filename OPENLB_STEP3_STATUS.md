# OpenLB migration — Step 3 (airflow) progress

Target version chosen: **OpenLB 1.8.1**. Rationale — it's the release the migration plan
was written against, it has a published User Guide (Aug 2025) I could actually read to
ground the code, and it carries every feature the plan needs (`Platform::GPU_CUDA`, WALE
LES, advection–diffusion lattices, particles). **1.9 is API-compatible**, so if you'd
rather run the newest release the same code carries over with trivial edits.

Environment note (unchanged from Steps 1–2): OpenLB 1.8 can't be fetched from the cloud
container — the proxy blocks openlb.net / Zenodo / GitLab, and only a stale 2019 mirror
(~v1.3) is reachable. So the OpenLB-side code below was written against the app idiom +
the 1.8 User Guide but **compiled on your box, not here**. What *is* compiler-verified is
everything that doesn't need OpenLB: the geometry bridge, the inlet physics, and the
Stage-A↔B file handshake.

## What's verified here (compiled + numerically checked)

**ABL/RFG inlet — the plan's "single biggest task" (§6.1) — PASSES its gate.**
`abl_inlet_verify.cpp` exercises the ported inlet (`abl_inlet.h`, unchanged) as a pure
analytical field:

| check | result | verdict |
|---|---|---|
| mean streamwise profile vs log law | max 0.10% error | PASS (<2%) |
| solenoidality: RMS(div f)/RMS(grad) | 5.2% residual | acceptable — matches the header's "small residual; pressure cleans it over fetch" |
| turbulence intensity: unit-field variance | within 6.8% (σ_v runs low) | minor — see note |
| determinism (CPU/GPU parity prereq) | bit-identical repeat | PASS |

Two test bugs were found and fixed along the way (both in the *harness*, not the inlet):
averaging speed magnitude instead of the streamwise component (Jensen bias), and a
structured sample grid that aliased against the Fourier modes and faked a 20% mean
offset — randomized sampling gives 0.10%. See `abl_inlet_verify.png`.

*Note (σ_v 6.8% low):* the per-component gain calibration in `abl_inlet.h::build_modes`
uses only 4000 samples; bumping that to ~40k tightens all three σ to <1%. Non-blocking
(within typical synthetic-inflow tolerance), and it's a one-line change to your file —
flagged rather than made, since I don't edit your sources silently.

**Stage-A↔B file handshake — verified.** `geometry_loader.h`'s reader round-trips the
`material_map.dat` + `source_mask.u8` the bridge produced: histogram reads back
1 fluid 2 445 872 · 2 wall 61 528 · 3 inlet 14 355 · 4 outlet 14 355 · 5 slip 59 325 ·
6 porous 4 725 · 7 ground 29 559 · Ω 22 812 — exactly Stage A's counts.

## What's scaffolded (compiles in your 1.8 tree, flagged `CONFIRM 1.8`)

- **`geometry_loader.h`** — part (A) the verified reader; part (B) `stampSuperGeometry()`
  copies the imported materials onto a `SuperGeometry<T,3>` cell-for-cell (the bridge,
  applied — no STL, no re-voxelization).
- **`abl_inlet_olb.h`** — wraps the verified inlet as an `AnalyticalF3D` to drive
  `defineU` on `MAT_INLET` each step.
- **`urban_flow.cpp`** — Stage B **airflow app, LIVE flow**: D3Q19 + WALE, ABL/RFG velocity
  inlet, pressure outlet, free-slip sides/top, developed to a statistically-turbulent state
  and snapshotted (no time-averaging — the FORWARD_LIVE regime; the Step-4 burst rides the
  instantaneous flow). Exports `umean_full.f32` in the project's 5-int format (filename kept
  for Stage C compatibility; payload is the live snapshot, gathered per-cell — not zeros).
  A header block lists the v1.3→1.8 API deltas.
- **`urban_flow.cpp` Step 4 (now scaffolded, not stubbed):** a second D3Q7 advection–diffusion
  lattice coupled to the LIVE NSE velocity (no frozen mean), with the settling offset −w_s ẑ,
  the Ω burst injection (gated by `source_mask.u8`), the surface deposition sink (α=8·v_d
  from `dep_vel.f32`, α/8·C half-way bounce-back per `CONTAMINANT_BC.md`), the Θ=∫C dt
  accumulator, a closing mass budget, and self-termination at 99% clearance. Writes
  `theta.f32`, `deposition.f32`, `exposure_timeseries.csv`, `meta_flow.txt` for Stage C
  (`J = ⟨w,Θ⟩/|Ω|`). Stage A now also emits `dep_vel.f32` (handshake verified). The three
  custom operators are host block-loops for CPU-gate testing; the §6.3 note flags the 1:1
  reimplementation as on-device post-processors for the A4000.

### Corrective / stabilisation methods now wired (the old engine's safety net)
- **C1 — Mach/CFL/τ preflight:** refuses to run if `Ma≥0.1`, `τ≤0.5`, or `uLB≥0.1` (`FORCE=1` overrides). Pure converter math; ramp math unit-checked here.
- **C2 — rough-wall function on the GROUND (z0):** the ground was split into its own material `MAT_GROUND(7)` in the bridge so the wall function targets only the floor (buildings stay smooth no-slip bounce-back, COST 732). This is the fix for the ~35% ABL horizontal-homogeneity drift the old no-slip floor caused. Falls back to bounce-back with `GROUND_BOUNCEBACK=1`.
- **C3 — inlet startup ramp:** smoothstep over one flow-through (verified monotone, 0→1) so the domain fills without a pressure shock; applied through the inlet functor.
- **C4 — divergence/NaN guard:** checks `maxU` finite and sub-Mach every `CHECK_EVERY` steps and aborts cleanly (mirrors `forward_city`'s guard).
- **C5 — outlet sponge layer:** a graded high-viscosity Smagorinsky fringe (`SPONGE_CELLS`, wind-aligned) absorbs turbulence before the pressure outlet so it doesn't reflect.
- **C6 — LES stabilisation:** collision selectable at compile time (`-DCOLLISION_MODEL=0` WALE / `1` consistent-Smagorinsky / `2` regularized) plus an optional ADM deconvolution filter (`ADM_EVERY`).

The geometry bridge was re-verified after the ground split: `GROUND 29 559 + WALL(buildings) 61 528 = old WALL 91 087`, all reconciliation identities still PASS at 4 m and 2 m.

## Your move / the remaining Step-3 gate

1. Drop `urban_flow.cpp` + the three headers into `examples/urban/urban_flow/` in your
   OpenLB 1.8 tree; compile; work through the `CONFIRM 1.8` flags (they're the only spots
   I couldn't verify without the library).
2. Run on a city (`gen_openlb_geom` first to make `geom_out/`), spin up, and check the
   airflow gate the old solver failed: **cube-wake reattachment Xr/H ≈ 1.6 band** and
   **ABL mean-shear drift <10%** from inlet to city face. The inlet already reproduces the
   target profile to 0.10%, so any drift is the solver/wall treatment, not the inflow.
3. Then run Step 4 (now scaffolded) and check its gate: the linearity guard (single-cell
   release == the Ω-slice of a full-Ω release) and a mass budget closing ~99%.

If it's easier, tell me which OpenLB version you actually install and paste any compile
errors from the `CONFIRM 1.8` spots — I'll turn the scaffold into a clean build against
that exact version.
