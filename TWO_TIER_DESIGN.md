# Two-tier design: built form (outer) × zoning (inner)

Status: agreed specification, 2026-09-29. Nothing below is implemented yet except where noted.

## Question

Given a burst of particles released at a random location in a city's environs, which city
design minimises the population's exposure?

    minimise over design:  E over (release location x ~ p, wind direction ~ rose, time of day)
                           of the population dose from a unit burst at x

Particles, so dose is linear in concentration: the expectation over release locations equals
one run with the release spread over p(x) (Bennett et al. 2002 intake fraction; Apte et al. 2012
for distributed ground-level sources; Daniels et al. 2000 for the near-linear PM dose-response).

## Fixed (scenario and ground rules, never searched)

| item | value |
|---|---|
| City footprint | 600 m × 600 m |
| Environs | ring of width `ENVIRON_FRAC` × city size around it (start 0.5) |
| Release | burst, uniform per unit area over city + ring, independent of the design; ground level, or roof level where a building stands |
| Wind | **uniform 16-direction rose**, one reference speed, neutral stability. Sensitivity check only: single-lobed and two-lobed (sea-breeze) roses by reweighting the same 16 runs |
| Particles | 2 bins, ≈2.5 µm and ≈10 µm: own settling, deposition, size-dependent infiltration factor |
| Floor area | fixed plot ratio FAR 2.0 → GFA = 720 000 m² |
| Population | from GFA at ≈80 % occupancy with the builder's standards (35 m² per resident, 14 m² per job, labour rate 0.47): **P ≈ 13 800**, jobs ≈ 6 500 |
| Floor height | one value for all buildings (3.5 m), so a building's capacity does not depend on its use |
| Time budget | NHAPS: home 0.69, work 0.18, outdoors the rest; release time uniform over the day |
| Planning mode | new city: every building's use is free, no baseline plan |

Speed does not need a distribution: under neutral stability the dose from a fixed released
mass scales roughly as 1/U, which rescales every design equally.

## Outer level: built form (Bayesian optimisation on CFD)

| parameter | range | controls |
|---|---|---|
| block_size | 24–60 m | grain (≥ 6 cells at dx 4 m) |
| block_aspect | 0.5–2 | elongated blocks, canyon direction |
| street_width | 12–30 m | canyon aspect ratio H/W (≥ 3 cells at dx 4 m) |
| corridor | 0–2 wide streets, 30–50 m | ventilation corridors |
| height_concentration | 0–1 | uniform canopy ↔ concentrated core |
| height_cores, core_spread | 1–3, 0–1 | one tall core vs several |
| height_scatter | σ(ln h) 0–0.6 | towers among low-rise buildings |
| park_fraction | 0.05–0.30 | open space (porous canopy) |
| park_layout | 0–1 | one large park ↔ dispersed pocket parks |

Heights come from these shape parameters and are then scaled so total floor area equals the
fixed GFA (slenderness cap 7 × footprint as now). Population never sets height. Grid
orientation is not a parameter: under a uniform rose it has no effect.

Generator changes needed: a form-only mode in the builder (its pre-density `cbd_*`/`roughness`
path already gives population-independent heights; add the GFA normalisation, a single floor
height, full-footprint buildings only, no low-density housing type), and output of every
building's floor area, capacity and floor heights for the inner level.

## Inner level: zoning (exact integer programme per geometry)

- Variables per building b: use u_b ∈ {home, work}; occupancy o_b ∈ [0.5, 1], equal on all floors.
- Objective: Σ_b occupants_b × (0.69 if home else 0.18) × d̄_b + outdoor term, with d̄_b the
  building's floor-area-weighted expected dose (indoor through the size-dependent
  infiltration factor).
- Constraints: residents = P, jobs = 0.47 P, capacity from floor area, occupancy band.
- Event tail: CVaR₉₀ of population dose over (release zone × wind) ≤ bound, linear through the
  Rockafellar–Uryasev (2000) auxiliary variables; mean-plus-CVaR-constraint as in Krokhmal,
  Palmquist & Uryasev (2002).
- Per-person tail: no building occupied whose expected dose exceeds κ × the geometry's mean
  dose (κ ≈ 2): a bound on o_b. Equity framing: Levy, Chemerynski & Tuchmann (2006).
- Solver: HiGHS mixed-integer (`scipy.optimize.milp`); 60–150 buildings, seconds.

Outer objective: J\*(geometry), the best exposure any allowed zoning achieves. Also reported:
J at a reference zoning (uniform occupancy, use assigned at random with fixed proportions), so
that what form buys and what zoning buys can be separated (covariance decomposition).

The mean-plus-two-tails structure follows quantitative risk assessment's pairing of individual
and societal risk (Jonkman, van Gelder & Vrijling 2003; Vrijling et al. 1995; HSE 2001 R2P2).
Those criteria concern fatal accidents; here only their structure is borrowed.

## Dose data per geometry

- One live-flow run per wind direction.
- K labelled tracers per run, one per release zone (environs split into K = 16–32 tiles), for
  each particle bin. Transport is linear, so the tracers do not interact and their sum is the
  uniform release. Dose per building from its envelope cells, per floor height.
- Memory ≈ 1 GB per million cells at K = 32 × 2 bins: fits on the H100.

## Wind directions: move the inlet, keep the city on the grid

The city is never rotated. Every direction sees the same building voxels, so there is no
direction- and design-dependent staircase error (a 12 m street at 45° would be ~2 cells wide in
places, biasing exactly the street-width and block-size parameters being optimised).

- 0°, 90°, 180°, 270°: already supported: `openlb_geometry.h` puts the inlet on the matching face.
- The 12 oblique directions: velocity inlet on the two upwind faces, pressure outlet (with
  sponge) on the two downwind faces. The ABL mean profile already takes a wind angle
  (`abl_inlet.h`); the RFG fluctuation components must be rotated with it. The RFG is a
  space-time field, so evaluating it on both inflow faces gives turbulence that is consistent
  across the corner.
- Domain laid out per direction by translating the city: fetch + environs upwind on both
  upwind sides, the 15 H wake buffer on both downwind sides. Diagonal directions cost
  ~1.5–2× the cells of axis directions.
- Also to generalise: top-stress direction, the Step-4 mass budget (fluxes through all four
  side faces, not only x).

Validation before use:
1. Empty-domain ABL at 22.5° and 45°: profile drift along the diagonal, as gate 6a.
2. Mirror test: a city symmetric about the x-axis must give the same J at +22.5° and −22.5°.
3. Exact-rotation test: a 90°-rotated city at 90° vs the original at 0°.
4. Outlet stability with turbulent backflow on the oblique outflow faces (the old solver's
   oblique runs diverged, but it had symmetry planes and no lateral fetch).

## Fidelity and cost

- dx 8 m, 8 directions: screen ~150 designs. dx 4 m, 16 directions: the top 15–20.
- Per design at dx 4 m: 16 runs; estimated 10–20 min each on an H100 at showcase size (to be
  replaced by a measured H100 speed).
- Final check: new inflow seeds and the in-between directions for the chosen designs.

## Build order

1. **Done (2026-09-29):** `gen_form_city.cpp` (+ `render_form_city.py`, `form_city_examples.png`).
   - Parameters as in the outer-level table, via environment variables; fixed rules as above.
   - Every building has the same footprint (block_size² at the given aspect), so the GFA is met
     by distributing whole floors: 2 floors minimum, H_MAX = 100 m and the slenderness cap as
     maxima; achieved GFA within ±0.2 % over 332 test designs (the 32 corners of the five
     geometric knobs + 300 random), all feasible; no envelope cell shared between buildings.
   - Outputs: material map (inlet on x = 0 for now), `source_mask.u8` (one release cell per plan
     cell over city + environs, on the roof where built), `release_zone.u8` (4 × 4 tiles),
     `envelope.i32` (per-building air cells for dose), `dep_vel.f32`, `buildings.csv`
     (floors, floor area, resident and job capacity), `meta_geom.txt`.
   - Domain is design-independent: 610 × 318 × 101 = 19.6 M cells at dx 4 m (2.5 M at dx 8 m),
     set by the environs ring (300 m) and the 15 × H_MAX wake buffer (1500 m). The wake buffer
     is the largest single cost; lowering H_MAX to 80 m would cut it by a fifth.
   - Checked end to end at dx 8 m: `urban_flow` loads it, all release cells are accepted as
     sources, scalar budget closes to 6e-16.
   - Interaction to be aware of: park_layout 0 puts parks on the central slots, so a central
     height core then forms a ring of towers around a central park.
2. Oblique inlets (two-face inflow and outflow, per-direction domain layout) + the four
   validation tests above.
3. Labelled release-zone tracers and per-building dose output in `urban_flow`.
4. **Built ahead of step 3, tested on a proxy dose (2026-09-29):** `zoning/`.
   - `dose_table.py`: the programme's input, per-building dose per scenario (wind × release
     zone) with probabilities. `from_runs` builds it from `urban_flow` theta fields + the
     geometry's `envelope.i32`; `proxy` is a 2-D stand-in (not CFD) for testing.
   - `zoning_milp.py`: home/work per building + occupancy (50–100 %), residents = P, jobs =
     0.47 P, expected-dose objective, CVaR90 event-tail constraint (Rockafellar–Uryasev),
     per-person cap (no homes above κ × the mean resident dose); 200 random zonings as the
     exposure-blind reference; HiGHS, ~2 s for 122 buildings × 256 scenarios.
   - `plot_zoning.py`, `zoning_proxy_demo.png`. On the proxy: optimum 7.9 % below the random
     mean, but its CVaR90 is slightly WORSE than random (20.9 vs 20.4); along the front CVaR90
     drops 7 % for +6 % mean. The mean-optimal pattern (workplaces on the perimeter facing the
     environs, homes inside) comes from the proxy and is not a finding.
5. One geometry end to end, 16 directions, on the GPU; then the outer search.
