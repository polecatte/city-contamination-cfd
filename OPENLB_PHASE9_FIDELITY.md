# Phase 9: physical fidelity: HRR, corrected WALE, guideline domain, local diffusivity, ranking

Status 2026-10-06. Follows the physics review of the Phase 8 results (gates CPU = GPU, the
production city's intake fraction ~22 ppm against Apte et al. 2012's 14–52 ppm). That review
found five gaps; this phase fixes the four that are code and sets up the runs that test them.

| gap | effect | fix | where |
|---|---|---|---|
| lattice viscosity floor: tau 0.505 → nu ≈ 0.83 m²/s at dx 4 m, building Re_H ≈ 100–500 | wakes and canyon vortices of a low-Re flow (Snyder 1981: Re_H ≳ 1.1e4 for independence) | HRR collision at tau 0.5001 (nu ≈ 0.017 m²/s, cube Re_H ≈ 14 000) | `COLLISION_MODEL=3` |
| OpenLB 1.8's (and 1.9's) WALE has three slips | nu_t depends on the zz entries only: 0 for in-plane rotation, up to ~9× high in plane strain | `urban_les.h` | default for models 0 and 3 |
| compact domain: 40 m upstream, 35 m lateral | blockage 10.5 % (COST 732: < 3 %) | 5H / 6H / 5H / 15H with H_ref = 88 m: 2.79 % | `gen_openlb_geom` default |
| one scalar diffusivity everywhere: D = D_mol + nu_0/Sc_t | under HRR nu_0 is tiny: tau_AD = 0.50019, D ≈ 0.05 m²/s, no subgrid mixing | D(x) = D_mol + (nu_0 + nu_t(x))/Sc_t from the WALE nu_t | `D_LOCAL=1` |
| no dispersion validation, no grid convergence of J | | runs below | lab / H100 |

## 1. The WALE slips (OpenLB 1.8.1, `dynamics/collisionLES.h`, `detail::WaleEffectiveOmega`)

Nicoud & Ducros (1999):

    nu_t = (C_w Δ)² (Sd:Sd)^{3/2} / ((S:S)^{5/2} + (Sd:Sd)^{5/4}),
    Sd_ij = ½(g²_ij + g²_ji) − ⅓ δ_ij tr(g²),   S = ½(g + gᵀ)

The stock code (1.8.1 line numbers; **unchanged in 1.9.0**, lines 453/462/469, which only
replaced the zero-denominator guard by `+ 1e-12`):

- line 410: `G_ip = G[i][j]*G[i][j]` inside the double loop, `=` instead of `+=`, so Sd:Sd
  becomes Sd_zz² only;
- line 417: the same for S:S, which becomes S_zz² = (∂w/∂z)²;
- line 401: subtracts ⅓ Σ_i g_ii² instead of ⅓ tr(g²) = ⅓ g_kl g_lk, so the "traceless" part is
  not traceless.

The stock eddy viscosity is therefore a function of two numbers, (g²)_zz − ⅓Σg_ii² and ∂w/∂z,
whatever the rest of the gradient. `urban_les.h` (`urbanles::WaleCorrected<COLL>`) is the same
wrapper with the sums done properly. It also stores the effective omega in `EFFECTIVE_OMEGA`
for the scalar. `-DWALE_OPENLB` restores the stock model for comparison.

**Verification** (`tests/wale_verify/`): the stock function compiled verbatim, against an
independent numpy transcription of Nicoud & Ducros (1999) and the `urban_les.h` formula. Values
are tau_t = 3 nu_t (lattice units), C_w = 0.325, |g| ~ 0.01.

| velocity gradient | Nicoud & Ducros | urban_les.h | OpenLB 1.8.1 |
|---|---|---|---|
| pure shear ∂u/∂z | 0 | 0 | 0 |
| solid-body rotation (x-y) | 2.86e-3 | 2.86e-3 | **0** |
| axisymmetric strain | 4.77e-4 | 4.77e-4 | 6.73e-4 |
| plane strain in x-y | 2.76e-4 | 2.76e-4 | **2.59e-3** (9×) |
| shear + rotation in x-y | 6.79e-5 | 6.79e-5 | **0** |
| 6 random traceless tensors | — | = reference (all) | 0.2× to 8× the reference |

`urban_les.h` matches the reference to all printed digits. The stock value equals the
"(z,z)-entries only" prediction exactly in every case, which confirms the reading of the code.
I found no bug report, forum thread or changelog entry about it (searched openlb.net and the
1.9 release notes, 2026-10-06).

Correction to an earlier statement of mine: the stock model is *not* "near zero in a boundary
layer". In pure shear the correct WALE is also zero; that is the model's designed wall
behaviour. The stock error is in both directions, by large factors, depending on how the local
gradient is oriented relative to z:

- zero for any motion confined to the x-y plane (rotation or shear there);
- several times too large for horizontal plane strain.

Consequence for earlier results: every Phase 5–8 run had a subgrid viscosity that was wrong
cell by cell. Whether it was too high or too low on average in those flows has not been
measured. The cube gate can measure it: rerun with and without `-DWALE_OPENLB` at the same tau.
The tau 0.505 floor (ν ≈ 0.83 m²/s) was present in all of them either way.

## 2. HRR (model 3)

`dynamics::Tuple<…, momenta::BulkTuple, equilibria::ThirdOrder, WaleCorrected<collision::HRR>>`.
HRR is Jacob, Malaspinas & Sagaut (2018): a hybrid of regularised populations and a
finite-difference strain rate, with weight σ = `HRR_SIGMA` (default 0.98, as in the earlier
handmade solver).

- HRR reads the strain rate from `descriptors::TENSOR`. The WALE gradient operator now writes it
  there (`urbanops::writeStrain`) from the same velocity gradient WALE uses.
- Preflight accepts tau ≥ 0.50001 for model 3 (0.505 otherwise).
- Build: `MODEL=hrr ./lab_openlb.sh setup`. This builds `examples/urban/urban_hrr` with
  `-DCOLLISION_MODEL=3` and runs under `$WORK/hrr` at `TAU=0.5001`.

Validation (cube, gate 6b, dx 4, tau 0.5001, ν = 0.0167 m²/s), **not yet decided**:

- In the cloud container (4 threads, 3–5 MLUPs) the run was stable through 5 000 steps: the full
  inflow ramp to 86 %, peak lattice |u| 0.08.
- Both attempts were then killed by the container: a 30-minute task limit, then the container
  being reclaimed while idle. Neither was a solver failure.
- A full run (26 k steps, ≈ 2–3 h at that speed) belongs on the lab rig or the H100:
  `MODEL=hrr GPU=1 ./lab_openlb.sh gates`, whose gate6b_dx4 / gate6b_dx2 run at tau 0.5001.
- To compare: Xr/H with BGK 2.63 (dx 4) / 2.31 (dx 2), and experiments ≈ 1.4–1.6.

## 3. Domain (COST 732 / AIJ)

`gen_openlb_geom` now sizes the domain from a fixed H_ref = 88 m (the default design's tallest
building), not from the design's own height:

- 5H upstream;
- 6H to each side (5H gave 3.1 % blockage);
- 5H above;
- 15H downstream.

A fixed height keeps J continuous across designs.

| domain | dx | grid | cells | blockage |
|---|---|---|---|---|
| compact (old) | 4 | 490×167×89 | 7.28 M | 10.46 % |
| COST 732 | 4 | 590×414×133 | 32.5 M | 2.79 % |
| COST 732 | 8 | 295×207×67 | 4.09 M | 2.33 % |

`DOMAIN=compact` keeps the old extents. The lab script's gate and showcase geometries use it, so
earlier results reproduce. The guideline domain at dx 4 needs ≈ 20 GB of device memory (H100),
and at dx 8 it fits the A4000.

## 4. Release set Ω

- **Default (environs).** One release cell per plan cell over the city plus a ring of
  `ENVIRON_FRAC` (0.5) × city size around it: 1200 m × 1200 m, 22 500 cells at dx 8.
  - Releases are at ground level, or on the first fluid cell above the roof where a building
    stands.
  - The release is uniform per unit area and covers the same area for every design.
  - `release_zone.u8` tiles it 4 × 4 for the tail constraint and labelled tracers.
- **`OMEGA=legacy`** (the default with `DOMAIN=compact`) is the old rule: streets and parks only,
  over the city + 40 m / 35 m / 70 m. It reproduces the old `source_mask.u8` and
  `material_map.dat` byte for byte.
- `urban_flow` accepts every environs release cell (22 500 of 22 500, 1200 on roofs).

## 5. Local scalar diffusivity (`D_LOCAL=1`)

The scalar is advected by the resolved velocity. Only the unresolved mixing needs a diffusivity,
and the standard LES closure is ν_t/Sc_t with the flow's own subgrid viscosity.
`urbanles::VelocityEddyDiffusivityCoupling` replaces OpenLB's velocity coupling: identical velocity
copy, plus, per cell and per step,

    nu_0 + nu_t = (1/omega_eff − ½)/3          (EFFECTIVE_OMEGA from WaleCorrected)
    tau_D       = ½ + invCs2 · (D_mol + (nu_0 + nu_t)/Sc_t),   floored at TAU_AD_MIN (0.505)

This is written into the AD cell's OMEGA, which `dynamics::ParameterFromCell<OMEGA, AD_DYNAMICS>`
reads. Under TRT it is converted to the even rate as in `step4::prepare`.

- Cells without a valid EFFECTIVE_OMEGA (boundary dynamics, before the first collision) take ν_t = 0.
- The floor (D ≥ 0.625 m²/s at dx 4, U_lb 0.032) is a numerical-stability bound for BGK
  advection–diffusion at cell Péclet ~10–30. The run prints the share of cells sitting on it.
- `D_LOCAL=0` (default) is the constant-D path, bitwise unchanged: the bulk dynamics are not
  wrapped, and the coupling's velocity arithmetic is OpenLB's.
- Under model 0 with ν_t = 0, the local formula equals the constant one, so the two paths agree
  where the flow is laminar.

Checks (tiny compact city, 160 m, dx 8, HRR tau 0.5001, short spin-up):

- `D_LOCAL=0` with the new code is bitwise identical to the previous build (theta.f32 `cmp`,
  model 0).
- `D_LOCAL=1` is stable and the budget closes to 1e-14.
- At release, mean D = 1.25 m²/s, max 3.3 m²/s, and **99.9 % of the cells sit on the tau_AD
  floor**.
- That is expected in magnitude. In lattice units, the WALE ν_t ≈ C_w² |S|_lb ≈ 0.11 × 0.0064 ≈
  7e-4, so ν_t/Sc_t ≈ 1e-3, below the floor's 1.25e-3. Physically (C_w Δ)² |S| ≈ 6.8 m² × 0.1 s⁻¹
  ≈ 0.7 m²/s.
- So the subgrid scalar mixing is set by the stability floor, at about the size the closure would
  give anyway. The constant-D HRR path (tau_AD 0.50019, D ≈ 0.05 m²/s) is far below both.
- J's sensitivity to that floor is measured, not assumed: TAU_AD_MIN 0.505 / 0.51, Sc_t
  0.5 / 1.0, and AD TRT (`-DAD_TRT_MAGIC=0.25`) to allow a lower floor.

## 6. Validation and production runs

All with HRR (`MODEL=hrr`).

| run | where | what it decides |
|---|---|---|
| 6b cube dx 4 / dx 2 / dx 1 | lab, lab, H100 (`cube_dx1`) | Xr/H vs 1.4–1.6 and its grid trend; HRR's fix of the low-Re wake |
| 6a ABL | lab | profile drift with near-zero lattice viscosity |
| domain sensitivity: compact vs COST 732, same design, legacy Ω | lab (dx 8) / H100 (dx 4) | how much the 10.5 % blockage moved J |
| grid convergence of J: dx 8 / 4 (/ 2 on H100 at compact extents) | lab + H100 | whether dx 4 is converged for the ranking |
| D_LOCAL vs constant D; Sc_t 0.5 / 0.7 / 1.0; TAU_AD_MIN 0.505 / 0.51 | lab | closure sensitivity of J |
| MUST dispersion (FAC2 ≥ 0.5, \|FB\| ≤ 0.3, NMSE ≤ 4) | H100 | absolute validation of the concentration field (to be ported) |
| **ranking study** | lab (old model) + H100 (new) | next section |

## 7. Ranking study (`./lab_openlb.sh rank`, `tests/rank_report.py`)

The question: does a meaningful spread of geometries give a meaningful spread of exposure? And
would the old model have ranked them the same way?

Five contrasting designs from `gen_openlb_geom`, all ≤ 88 m with blockage < 3 % (dx 8 check):

| design | knobs | blocks | max H | blockage |
|---|---|---|---|---|
| base | defaults (60 m blocks, 20 m streets, 15 % park, peak 45) | 36 | 80 m | 2.33 % |
| core | PEAK_H 50, one employment centre, concentration 0.9 | 36 | 88 m | 1.77 % |
| low | PEAK_H 22, 28 m streets, 25 % park | 36 | 48 m | 1.45 % |
| fine | 36 m blocks, 12 m streets, PEAK_H 40 | 100 | 88 m | 2.33 % |
| coarse | 80 m blocks, 30 m streets, PEAK_H 35 | 25 | 80 m | 2.11 % |

There are two inlet seeds per design (the same seeds for every design, so differences between
designs are paired), run in two configurations:

- **old:** `./lab_openlb.sh rank`. BGK + WALE tau 0.505, compact domain, legacy Ω, constant D, dx 4.
- **new:** `MODEL=hrr GPU=1 ./lab_openlb.sh rank`. HRR + corrected WALE, COST 732 domain,
  environs Ω, D_LOCAL; dx 4 on the H100 (dx 8 elsewhere).

The report gives:

- per design, the intake fraction iF = B·J/dx³ in ppm (comparable across dx);
- the seed noise σ (pooled within-design);
- the spread (max − min)/mean;
- one-way ANOVA F and p;
- the number of distinguishable design pairs;
- Spearman ρ and Kendall τ between the configurations.

"Meaningful" means p < 0.05 and a spread above 3σ. With two seeds, a borderline F calls for
more seeds (`RANK_SEEDS="1000 2000 3000"`).
