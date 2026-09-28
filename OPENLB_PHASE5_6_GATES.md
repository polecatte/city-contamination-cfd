# OpenLB port — Phases 5–7: first runs, gate results

Date: 2026-09-27/28. Branch `claude/gracious-lovelace-3wlllj` (built on `claude/charming-allen-8pfsrc`,
Gate 5 green). Everything below was **run**, not reasoned about: OpenLB 1.8.1 (the GitLab
release tarball) built in a 4-core cloud container with the new `./olbconfig.sh cpu-mt`
(OpenMP, `CPU_SISD`, g++ 13.3). Numbers are from that box; the lab machine should reproduce
them to round-off apart from thread-order effects in the time means.

## 1. Scorecard

| Gate | What | Result |
|---|---|---|
| 3 | Stage A geometry, clean rebuild | ✅ counts identical to `OPENLB_PORT_STATUS…` §3 |
| 5 | OpenLB's own voxel counts = Stage A (+ sponge carve) | ✅ exact, all 10 materials |
| 6a | ABL drift inlet → city face < 10 % | ⚠️ **10.9 %** at the new default operating point (9.4 % at the old one); ≤ 7 % from ¼ L on (§4, §6) |
| 6a′ | peak lattice \|u\| < 0.1 throughout | ✅ **0.090** at the new default (τ 0.505, uLB 0.032); was 0.14–0.15 (§6) |
| 6b | cube reattachment Xr/H ∈ 1.4–1.8 | ❌ **2.60** (τ 0.51), **2.66** (τ 0.505) — not a viscosity effect (§5) |
| 7a | scalar mass budget closes < 1 % | ✅ **5e-11** (measured flux, not inferred) |
| 7b | linearity Θ_{a+b} = Θ_a + Θ_b | ✅ L2 **3.2e-8**, max pointwise **3.6e-6** |
| 7c | v_d = 0 → deposition exactly 0, all mass out | ✅ **0.0**, 99.97 % left downstream |
| 7c | deposited fraction vs reference | ✅ **−9.5 %** (2 mm/s), **−7.4 %** (2 cm/s) |
| 7a | …on the city (2.63 M cells, live flow) | ✅ **1.3e-14** (after the Ω/CV split, §3.6) |
| 8 | city J finite, positive, seed-stable < 5 % | ✅ **1.27 %** spread — but truncated at 388 s, 25 % still airborne (§8) |

## 2. The three requested changes

- **`olbconfig.sh cpu-mt`** — seeded from `config/cpu_gcc_openmpi.mk`, then `CXX/CC := g++/gcc`
  (the template names `mpic++`), `PARALLEL_MODE := OMP`. In 1.8 OpenMP parallelises each block's
  collide over `iX`, so it scales on single-cuboid runs. Measured 24 MLUPS on 4 threads for the
  bare city lattice (14 MLUPS with WALE active, §3).
- **τ floor + derived dt.** Preflight is now `tau >= 0.505`; the converter is
  `UnitConverterFromResolutionAndRelaxationTime`. **One correction to the plan as written:** the
  viscosity handed to it cannot be molecular air. At dx = 4 m and τ ≥ 0.505,
  ν = 1.5e-5 m²/s derives dt ≈ 3.6e3 s and uLB ≈ 3.6e3 — the Mach gate refuses. That is *why* the
  old point sat at τ = 0.5000001. The lattice carries a background eddy viscosity, `NU_EFF`,
  derived by default from `TAU` (0.51) and `LATTICE_U` (0.05): ν_eff = (τ−½)/3·dx·U/uLB =
  1.07 m²/s, Re_eff(charL) = 1335. dt is unchanged at 0.05 s, so step counts do not move.
- **Spin-up line** reports the computed spin-up and, separately, any `MAX_STEPS` override.

## 3. Defects that only a run could show

The compile-green port had four silent defects. Each would have produced finite, wrong
numbers (or NaN that the guard could not see):

1. **OMEGA and the LES constant were never set.** 1.8 dynamics read both as lattice
   *parameters*; unset they are 0, and ω = 0 wrote NaN into 11 390 bulk cells on the first
   step. Worse, they must be set **after** every `boundary::set`: a parameter set earlier does
   not reach the BGK mixin the boundary installs, and the outlet's finite-difference processor
   then reads `getOmegaOrFallback(signaling_NaN)` — found by bisecting `prepareLattice`.
2. **WALE was plain BGK.** 1.8's WALE reads a `VELO_GRAD` field that the *application* must
   refresh each step (`examples/turbulence/tgv3d` does). Never refreshed, ν_t ≡ 0. The stock
   refresh (`SuperLatticeVelocityGradientFD3D`) heap-allocates per cell and ran at < 1.5 MLUPS;
   `VeloGradRefresh` is the same stencil (8th-order central in the bulk) in OpenMP, at 14 MLUPS.
   `WALE_GRAD_CHECK=1` compares them: 14 300 cells, **max difference 0**.
3. **The C4 guard was blind to NaN.** `LatticeStatistics` keeps its max with `uSqr > tmpMax`,
   false for NaN; it reported 1.49e-154 (√DBL_MIN) on the NaN lattice. The guard now scans the
   cells and names the materials holding non-finite values.
4. **Every park was solid.** `POROSITY` was never defined, and 0 means solid. Now
   `PARK_POROSITY` (default 0.8 = voxelize.h's `PERM_PARK`; the C_d/LAD calibration is open).

Two hand-off defects on the Stage C side, found on the city output:

5. **Wind speeds would have plotted 69× too fast.** `visualize_forward.py` multiplies
   `umean_full.f32` by U_inlet/U_LB on the assumption that it holds lattice velocity, as
   `forward_city`'s did; urban_flow writes m/s. urban_flow now writes a `meta.txt` with
   `velocity_units ms` and the viz honours it (forward_city output unchanged). urban_flow
   also copies `geom_type.u8`, `source_mask.u8`, `receptor_w.f32` into its output directory.
6. **Stage-A Ω reaches past the scalar budget's control volume.** 165 ground cells at
   x = nx−2 release mass that leaves through the outlet without crossing a CV face; counted as
   emitted, they held the city budget 0.75 % open from step 0 (exactly 165/22 128) while the CV
   itself closed. Emission is now split by CV membership. (Also: 684 Ω cells sit on inlet,
   outlet or slip faces, where the scalar has boundary dynamics, and do not emit — 22 128 of
   Stage A's 22 812. J is normalised by what was actually released.)

## 4. Gate 6a — the floor (G1 decided on evidence)

Empty domain at the city's fetch and height (177 × 40 × 89, dx = 4 m), 4 flow-throughs, time
mean over the last 2. Drift = max_z |U_x(z) − U_inlet(z)| / U_inlet(40 m), laterally averaged —
`tests/airflow_validation.cpp`'s T_abl definition.

| floor / top | city face (40 m) | ¼ L | ½ L | ¾ L | U(4 m) inlet → ½ L |
|---|---|---|---|---|---|
| bounce-back / slip lid | **42.7 %** | 50.9 % | 53.2 % | 53.1 % | 3.88 → 0.67 m/s |
| RoughWall / slip lid | 10.8 % (at z = 348 m) | 9.7 % | 8.5 % | 9.0 % | 3.88 → 3.90 |
| RoughWall / TopStress | **9.4 %** (at z = 348 m) | 4.4 % | 4.5 % | 3.9 % | 3.88 → 3.88 |
| same, new operating point (§6) | **10.9 %** (z = 348 m; 10.3 % at z ≤ 100 m) | 7.0 % | 5.1 % | 4.1 % | 3.88 → 3.78 |

- **Bounce-back fails as the audit feared** (option 1 measured, fails). A no-slip wall one 4 m
  cell from the flow imposes a laminar stress ~ν·U₁/(dx/2), an order of magnitude above ρu*².
- **`RoughWall` (option 3).** After each stream: (i) the floor is made *specular* by remapping
  OpenLB's full-way bounce-back populations to their mirror partners — a pairwise swap within
  the first fluid layer, so mass is conserved exactly and the normal flux stays zero;
  (ii) ρu*² of tangential momentum is removed from that layer with u* = κ|U₁|/ln(z₁/z₀) from
  the local instantaneous velocity, by exact-difference forcing. Heights use the inlet's
  convention (z₁ = dx) so floor and inlet describe the same log law. Buildings stay bounce-back.
- **`TopStress`.** With the floor fixed, the residual was at the top cell under the slip lid —
  *identically* with either floor, so not a floor effect. A log-law ABL carries u*² at every
  height; a slip lid supplies none. Adding +u*² under the lid (Richards & Hoxey 1993) is the
  mirror of the floor operator. The remaining 9.4 % is the top cell 40 m from the inlet, still
  adjusting; everywhere downstream it is ≤ 4.5 % at every height.
- Both are host OpenMP loops over ~nx·ny cells; on GPU they become post-processors (Phase 8).
  `GROUND_MODEL=0` / `TOP_STRESS=0` restore the old behaviour for comparison.

## 5. Gate 6b — the cube

AIJ domain (5H up, 15H down, 5H each side, 5H headroom), H = 10 cells = 40 m, same floor/top,
4 flow-throughs, mean over the last 2 (≈ 6 shedding periods).

| τ | ν_eff (m²/s) | Re_H ≈ U_H·H/ν_eff | Xr/H | wake | peak \|u\|_lb |
|---|---|---|---|---|---|
| 0.51 | 1.07 | 225 | **2.60** | reversed to −1.6 m/s, horseshoe yes, roof reversal no | 0.142 |
| 0.505 | 0.53 | 450 | **2.66** | reversed to −1.3 m/s, horseshoe yes, roof reversal yes | 0.143 |

The wake has the right structure — upstream horseshoe reversal, a closed recirculation, a
roof separation at the lower viscosity — but the bubble is ~60 % too long. **My first
hypothesis, too little Reynolds number, is refuted by the second row**: halving the base
viscosity moved Xr/H by +0.06, inside the scatter of a 2-flow-through mean. The candidates
left, in the order I would test them:

1. **Resolution — now the prime suspect.** H/dx = 10 is the audit's floor. On a coarse grid the
   separated shear layer transitions late whatever ν is, which lengthens the bubble. Test: the
   same cube at dx = 2 m (H/dx = 20; 11 M cells, ~16× the cost — hours on the 32-core box, not
   feasible here).
2. **Inflow turbulence — measured, and not enough to explain it.** Xr falls with turbulence
   intensity at roof height, so this was the other candidate. One empty-domain run with the
   new second moments (`AVG_FT=2`, `gate6_analyze.py abl`), at the cube position 200 m
   downstream: I_u = 13.9 % at z = 40 m (roof) and 15.4 % at 20 m, against the neutral-ABL
   target σ_u = 2.5u* → 15.1 % and 17.2 %. About 8–10 % low, and it does not decay over the
   fetch (12.9 % at the inlet at 40 m). A 10 % deficit in I_u does not make a bubble 60 % long.
3. Only then the floor model inside the bubble (reversed flow under `RoughWall`), by rerunning
   with `GROUND_MODEL=0` for the cube alone.

## 6. The Mach sub-condition (6a′)

The one free choice at fixed dx is the split between τ and the lattice velocity:

    ν_eff = (τ − ½)/3 · dx · U_ref / uLB_ref

- **Mach:** the log-law inlet is fastest at the domain top and the resolved fluctuation rides
  on it; every run measured peak |u|_lb = 0.14–0.15 ≈ 2.9 × uLB_ref. Peak < 0.1 needs
  uLB_ref ≤ 0.034 (dt 0.034 s, +47 % steps).
- **τ floor:** τ ≥ 0.505. At uLB_ref = 0.034 that gives ν_eff ≥ 0.78 m²/s — within the range
  the two cube runs show Xr is insensitive to, so meeting the Mach condition should not cost
  6b anything. It costs ~1.5× wall time.

**Decided (2026-09-28): `TAU=0.505`, `LATTICE_U=0.032` are the defaults** — the more physical
choice on both axes: lower Mach and 22 % less added viscosity (ν_eff 0.83 vs 1.07 m²/s). The
city's measured peak ratio is 3.06, hence 0.032 rather than 0.034. dt = 0.032 s, 1.56× the
steps per flow-through.

Measured at the new point: peak |u|_lb **0.090** on the 6a fetch (6a′ passes); 7a/7b/7c all
still pass on the box (closure 1.4e-11, deposition −10.4 % vs reference, linearity holds).
The cost is at the city face: drift 10.9 % (9.4 % before). The deficit is the inlet-adjustment
zone — U(4 m) dips 3.88 → 3.25 m/s at 40 m, then recovers to 3.72 / 3.78 / 3.86 at ¼, ½, ¾ L —
and a less viscous flow adjusts over a sharper, deeper dip. The city face sits inside it
because `BUF_UP` is 40 m. Drift is ≤ 7 % from 176 m on, so an upstream buffer of ~150 m would
clear the gate. (COST 732 asks 5 H_max upstream, 440 m here.) Not changed: it also moves the
start of Ω, so it is a metric decision as much as a domain one.

## 7. Phase 6 — scalar + deposition (Step 4)

Rewritten against 1.8 and runtime-enabled (`STEP4=1`). D3Q7 BGK; OpenLB's
`NavierStokesAdvectionDiffusionVelocityCoupling`; Dirichlet C = 0 inlet, zero-gradient outlet,
bounce-back solids; S2 (v_d → lattice units) and S3 (ω_AD from D_eff = D_mol + ν_eff/Sc_t,
applied) fixed. Two traps met on the way: uninitialised OpenLB populations are C = 1 in the
shifted storage (every cell *and the padding* are now set to C = 0 — the padding otherwise
cycles a phantom through the boundary bounce-back cells with period 2), and the AD boundary
mixins read OMEGA too, so it is set after them.

The **budget is measured**: over the control volume x ∈ [1, nx−3] the outflow is the exact
lattice flux across its two faces, from post-stream ±x populations. So Gate 7a tests mass
conservation of the AD lattice and its boundaries — it is not an identity.

Gates on the 40³ box (`gen_gate6_geom CASE=box`, frozen 4 m/s wind, `STEP4_UNIFORM_U=4`):

- **7a** closure 5.4e-11 (with deposition), 5.7e-11 (without). Live-NSE smoke run: 8.1e-11.
- **7b** `tests/linearity_guard.cpp --run` (brought over from `openlb-integration-status`):
  relative L2 3.2e-8, max pointwise 3.6e-6 over 8 732 significant cells, deposition on at 10×.
- **7c** v_d = 0: deposited mass exactly 0.0; 99.97 % left through the outlet face, 0.10 %
  still airborne at the 99.9 % clearance stop.
- **7c** deposited fraction against a fine-grid 1-D reference (release where the lattice puts
  it, in the first cell): 0.697 % vs 0.770 % at 2 mm/s, 6.69 % vs 7.22 % at 2 cm/s. The
  residual −7…−10 % is first-cell resolution. (A closed-form *surface* release was ~30 % off,
  for the geometric reason, not a units one; S2's 80× could not pass either.)

**Known limitation — negative undershoots.** At τ_AD ≈ 0.52 and cell Péclet ≈ 10, BGK D3Q7
rings: ~1 % of the released mass sits in negative Θ, minimum −4 % of the peak. RLB: 0.9 %.
TRT (Λ = ¼, with the *correct* rate mapping — OpenLB's `collision::TRT` puts OMEGA on the even
part, and AD diffusion comes from the odd one) was worse: min −14 %, 2 % lost upstream. BGK is
kept. Superposition is unaffected (7b), so J is still the defined quantity; the undershoot is
an accuracy term to carry in the error budget, not a correctness defect.

## 8. Gate 8 — city end-to-end

*These runs predate §6's operating-point change and §9's 15 H domain; they are the evidence
that motivated both. The lab box reruns Gate 8 on the new domain.*

The production city (177 × 167 × 89, dx = 4 m), then-default operating point, rough wall + top
stress, 3 flow-through spin-up, then the burst over Ω (2 s pulse) riding the live flow. Two
runs differing only in the inlet seed (`ABL_SEED` 1000, 2000).

| | seed 1000 | seed 2000 |
|---|---|---|
| J = ⟨w,Θ⟩ / M_released | 7.579 | 7.774 |
| budget closure (CV) | 1.3e-14 | 1.7e-14 |
| deposited (by 388 s) | 2.79 % | 2.78 % |
| still airborne at stop | 25.4 % | 25.4 % |
| negative-Θ mass | 0.01 % | 0.01 % |
| peak \|u\|_lb (spin-up) | 0.153 | — |

J(t) seed difference along the curve: 0.2 % (50 s) · 0.3 % (100 s) · 1.1 % (150 s) ·
1.8 % (200 s) · 2.2 % (250 s) · 2.3 % (300 s) · 2.5 % (350 s). Spread about the mean at stop:
**1.27 %** (half the 2.5 % seed-to-seed difference) — Gate 8 passes as written.

**What this does not yet show.** The city clears slowly: 13 % leaves in the first 50 s, and the
late-time airborne mass decays with an e-folding time of ~263 s, so 99 % clearance is at
~1 240 s (~25 000 burst steps, ~5 h per seed on this 4-core box).
Both runs were capped at 8 000 steps (388 s) with a quarter of the release still airborne, so
J here is J(388 s), not J(∞), and the seed difference is still creeping up (+0.3 %/50 s at the
end). A full-clearance pair on the 32-core box is the remaining check; expect the final spread
at 3–4 %, which would still pass but with less margin than the design-to-design signal needs to
be judged against (`RANKING_STABILITY_NOTE.md`).

Two observations from the city fields (`visualize_forward.py`, now 8/8 figures on urban_flow
output): street-canyon speeds are 0–6 m/s at the first cell with near-stagnant courts, as
expected; and near-ground flow is *reversed* over the last ~100 m before the outlet. The city's
own wake reaches the outlet because the downstream buffer is 70 m (`BUF_DOWN`, inherited from
forward_city) against COST 732's ~15H. It does not break the budget (measured outflow stays
positive); §9 extends the buffer to 15 H.

## 9. Production domain: 15 H downstream

`gen_openlb_geom` now defaults `BUF_DOWN` to **15 × 88 m = 1 320 m** (88 m is the production
city's tallest building at the default knobs), a named constant rather than 15 × maxH of each
design: a domain that resizes with the design would make J step-discontinuous across designs.
It warns if a design is taller. The grid grows from 177 × 167 × 89 to **490 × 167 × 89**
(7.28 M cells, 2.8×). Two consequences, both handled:

- **Ω must not grow with the domain.** Stage A's rule (every open ground cell) turned the new
  buffer into release area: 75 083 cells instead of 22 812, which would dilute J with releases
  that never cross the city. Ω is now capped at the old footprint (`OMEGA_DOWN` = 70 m past the
  city). Checked: `source_mask.u8`, `receptor_w.f32`, `material_map.dat` and `dep_vel.f32` are
  cell-for-cell identical to the old domain over the shared region, and Ω and w are empty
  beyond it. Gate 5 is exact on the new map.
- **Cost.** 3 flow-throughs are now 45 939 steps (longer domain, smaller dt): ~7.7 h per
  spin-up on the 4-core container, so the wake check runs on the lab box (`lab_openlb.sh`
  gate `wake`: < 2 % reversed near-ground flow over the last 15 % of x; the old 70 m domain
  measures 68.9 % and fails it).

Small tests (`gen_gate6_geom`: abl, cube, box) are unaffected.

## 10. What I would do next

`./lab_openlb.sh all` on the lab box runs items 1–3 (runbook: `LAB_RUNBOOK_OPENLB.md`).

1. **Gate 6b.** The cube at dx = 2 m (H/dx = 20). Inflow turbulence is measured and ruled out as
   the main cause (§5), so resolution is the test that decides it.
2. **Gate 8 to full clearance** on the 15 H production domain, two seeds, plus the wake check.
3. **Upstream buffer** (§6): decide whether to lengthen `BUF_UP` to clear 6a at the city face.
4. **Phase 8 (GPU).** Every host operator added here — `VeloGradRefresh`, `RoughWall`,
   `TopStress`, the Step-4 inject/deposit/accumulate/flux loops — is a per-cell loop with no
   cross-cell writes (the specular remap reads a snapshot), so each maps onto an OpenLB
   post-processor one-to-one. That is the G2 rewrite, now with a known list.

## 11. Reproduce

All of this is scripted in `lab_openlb.sh`; the manual equivalent:

```bash
export OLB_ROOT=…/release-1.8.1 && ./olbconfig.sh cpu-mt && (cd $OLB_ROOT && make -C external)
# app dir: symlink urban_flow.cpp geometry_loader.h abl_inlet_olb.h abl_inlet.h; Makefile from nozzle3d
g++ -O2 -std=c++17 -I. gen_gate6_geom.cpp -o gen_gate6_geom
for c in abl cube box; do CASE=$c OUT_DIR=geom_$c ./gen_gate6_geom; done

# 6a / 6b
GEOM_DIR=geom_abl  OUT_DIR=g6a SPINUP_FT=4 AVG_FT=2 ./urban_flow && python3 tests/gate6_analyze.py abl  geom_abl  g6a
GEOM_DIR=geom_cube OUT_DIR=g6b SPINUP_FT=4 AVG_FT=2 ./urban_flow && python3 tests/gate6_analyze.py cube geom_cube g6b

# 7a / 7c / 7b (box, frozen wind)
export STEP4=1 STEP4_UNIFORM_U=4 GEOM_DIR=geom_box CLEAR_FRAC=0.001 MAX_BURST_STEPS=6000
OUT_DIR=g7a ./urban_flow && python3 tests/gate7_check.py budget g7a && python3 tests/gate7_check.py dep g7a geom_box
VD_SCALE=0 OUT_DIR=g7c ./urban_flow && python3 tests/gate7_check.py nodep g7c
CLEAR_FRAC=0 MAX_BURST_STEPS=1500 VD_SCALE=10 ./linearity_guard --run ./urban_flow geom_box lin7b

# 8 (city, two inlet seeds)
for s in 1000 2000; do STEP4=1 ABL_SEED=$s CLEAR_FRAC=0.05 MAX_BURST_STEPS=8000 GEOM_DIR=geom_out OUT_DIR=city_s$s ./urban_flow; done
python3 tests/stage_c_J.py geom_out city_s1000 city_s2000
```
