# The optimisation pipeline

Status 2026-10-13. Question: given a burst of fine particles released anywhere in a city's
environs, which design minimises the population's exposure? This file walks one evaluation from
parameters to the final answer and marks what is validated and what is not.

```
x in [0,1]^11 ─▶ gen_openlb_geom ─▶ urban_flow ─▶ J, iF ─▶ GP over (design, direction) ─▶ next (x, phi)
 (param_space_   (builder rev 9,      (HRR + WALE,    (opt/      (opt/driver.py)                 │
  density.py)     voxels, receptors,   burst, tail)    evaluate)                                  ▼
                  release set)                                              verify: top designs × 16 directions
```

## 1. Parameters (`param_space_density.py`)

Eleven knobs, unit cube. Wind direction is not a design knob: it is integrated over the rose.

| knob | range | sets |
|---|---|---|
| block_w, block_d | 24–100 m | grain; ≥ 6 cells at dx 4 |
| street_width | 12–30 m | canyon width; ≥ 3 cells at dx 4 |
| roughness | 0–0.8 | scatter of building heights (≈ std of ln h) |
| height_concentration | 0–1 | uniform canopy ↔ towers at the job/housing peaks |
| emp_centers | 1–4 (categorical) | number of job centres |
| emp_centrality, emp_concentration | 0–1 | where the centres sit, how sharp they are |
| res_gradient | 0–3 | housing falloff from the centre |
| jh_mix | 0–1 | segregated ↔ mixed use |
| park_fraction | 0.05–0.40 | open space |

Fixed scenario: 600 m × 600 m city, 20 000 people, 0.47 of them working in the city.

## 2. Build (`gen_openlb_geom`, builder revision 9)

- **Layout.** A street grid covering the city, clipped at its edge. Heights follow the density
  fields (shape set by `height_concentration`) and are scaled so total floor area houses the
  population (residents 35 m², workers 14 m²: 831 600 m²).
- **Height limit.** Each building is capped at min(slenderness 7, `H_MAX` = 100 m). A design that
  cannot fit is infeasible (exit 3).
- **Feasibility in practice.** A survey of 1 024 Sobol designs (`tests/builder_survey.py`): none
  infeasible; tallest building p50 52 m, uncapped p95 124 m. About 5 % of designs touch the limit,
  and their floors move to other buildings.
- **Domain.** COST 732 extents from H_MAX: 5 H upwind, 6 H lateral, 5 H above, 15 H downwind
  along the wind. That is 44.5 M cells at dx 4 and 5.6 M at dx 8; blockage ≈ 1.6 %.
- **Wind direction.** Two parts:
  - `ORIENT` 0–7 turns or mirrors the city by an exact grid symmetry (`tests/orient_check.py`:
    cell-for-cell exact for buildings, receptors and releases);
  - `WIND_DEG` 0–45 sets the inflow angle (inflow on x = 0 and y = 0, outflow on the far faces).

  `tools/wind_rose.py` maps a direction to the pair; a 16-direction rose needs inflow at 0°, 22.5°
  and 45° only.
- **Release set.** One release cell per plan cell over the city plus a 300 m ring, at ground
  level or on the roof (22 500 cells at dx 8).
- **Receptors.** Person-time per pathway (NHAPS):
  - indoor (home 0.69, other indoor 0.18 of P) on each building's envelope air, × infiltration 0.62;
  - park visitors (0.038 P) on the park's ground-level air;
  - street (0.093 P) on road cells next to the buildings, as `occupancy.h`.

  The outdoor pathways are 19.5 % of the weight.

## 3. Exposure (`urban_flow`)

- **Flow.**
  - Live LES: HRR collision + corrected WALE at τ 0.5001 (production), or BGK + WALE at 0.505.
  - ABL inlet with synthetic turbulence, rough-wall ground, top stress, sponge + pressure outlet.
  - 3 flow-throughs of spin-up.
- **Scalar.** A 2 s burst from every release cell at once; transport is linear, so this is the
  average over release locations. Per-cell D from the eddy viscosity (`D_LOCAL=1`) and
  deposition (2.5 µm).
- **Stop and tail.** Stop at 5 % airborne with the tail extrapolation (`TAIL=1`, TAIL_SPEC.md);
  1 % is the fallback.
- **Result.** J = ⟨w, Θ⟩/M; intake fraction iF = B·J/dx³ (ppm), the objective (comparable
  across dx).

## 4. Surrogate and search (`opt/driver.py`, BoTorch 0.18.1)

- **One run = one (design, direction, inlet seed).** The seed's scatter (≈ 0.5 % in iF) is the
  GP's noise.
- **GP over (design, direction).**
  - k = k_d + k_d′·k_θ: Matérn-5/2 on the continuous knobs × a categorical kernel on emp_centers;
    periodic in the direction.
  - The rose mean F(x) is the GP averaged over the 16 directions (AppendFeatures + Expectation;
    Cakmak et al. 2020).
- **Next design.**
  - Log expected improvement on F, with the best posterior mean as the incumbent (plug-in EI for
    noisy runs).
  - Candidates: the acquisition optimum for each emp_centers level, plus every design already run.
  - Infeasible designs are skipped by the generator's 10 ms pre-check.
- **Next direction for that design.** The one that most reduces the variance of its F:
  argmax_j Cov(F, J_j)² / (Var J_j + noise), exact for a GP.
- **Start.** 24 Sobol designs, one direction each (a shuffled cycle over the 16).
- **How directions get spent, as observed.** The optimizer does not re-run the same design at a
  new direction. It samples a contender through near neighbours run at other directions. On the
  synthetic test, the 10 runs nearest the final best design were within half the typical design
  distance and covered 5–7 directions.
- **Resumable.** `history.jsonl` plus a per-run cache (`evals/<key>/result.json`). Θ is kept,
  so J can be re-scored without new CFD.

## 5. Final result (`driver.py verify`)

The top designs by surrogate F are run at all 16 directions with new seeds. The reported F is that
direct 16-direction CFD average, not the surrogate. `verified.json` keeps the per-direction values,
so other wind roses can be applied afterwards.

## 6. Tests

| test | checks | result |
|---|---|---|
| `tests/test_wind_surrogate.py` | F from one direction per design; calibration | 0.45 % error vs 3.8 % at face value; 95 % coverage |
| `tests/test_driver_synth.py` | whole loop on a known function, 11-D, resume, verify | best design better than 100 % of 4 000 random ones; verify within 1 σ of the surrogate |
| `tests/orient_check.py` | ORIENT exact, oblique faces | 7 orientations cell-for-cell exact; faces right at 0 / 22.5 / 45° |
| `tests/builder_survey.py` | feasibility, height distribution, smoothness | 0 % infeasible; sweeps monotone apart from one-cell steps |
| `tests/tail_replay.cpp` | early stop + tail vs a 0.1 % run | −0.04 % at 5 % |
| oblique smoke run (22.5°, dx 8, small city) | stability, 4-face mass budget, in-solver tail | closure 3e-15; tail = fit prediction |

## 7. Weak points (status)

| | issue | status |
|---|---|---|
| 1 | receptor weighting | **fixed** |
| 2 | stages not connected | **connected** (this file); first real-CFD evaluation through the evaluator run here |
| 3 | flow physics not validated (cube wake, HRR, grid convergence of J, dispersion) | open: lab / H100 runs |
| 4 | parameter bounds, unrealistic heights | **fixed** (rev 9, H_MAX) |
| 5 | dx 8 as screen; builder discontinuities | builder smoothed (one-cell steps remain at dx 4, from the grid itself); dx 8 vs 4 decided by the ranking study |
| 6 | one wind direction | **all 16 possible**; oblique inflow runs but is not validated (ABL drift on the diagonal, mirror and rotation tests, outlet stability) |
| 7 | 1 % stop bias | **tail extrapolation** (bias was small in the one test city) |
| 8 | spread vs noise | ranking study (planned) |
| 9 | one particle size | kept on purpose (fine particulate); see the note in the chat record |
| 10 | GP vs discrete knobs | emp_centers categorical; block-count / park / business flips remain as GP noise |

New, found while building this:
- `infiltration.h`'s size-dependent penetration curve contradicts the ranges it cites
  (P = 0.15 at 2.5 µm). It is unused by this pipeline, so nothing is affected, but it should not
  be used as it stands.
