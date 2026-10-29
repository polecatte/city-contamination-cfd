# Toy geometry-then-zoning demo — results

`python3 toy_zoning_demo.py toy_zoning` (≈10 s). Full console output: `run_output.txt`.

**Toy model, not CFD.** The dose comes from a 2-D steady advection–diffusion–removal model
with street channelling, canyon trapping (W/H) and an assumed exponential decay of dose with
height (`LZ` = 20 m). Nothing is calibrated. What carries over to the real pipeline is the
method: fixed geometry → scenario set → dose per floor → zoning LP → held-out validation.

## Setup

- **Geometry (fixed):** 8 × 8 blocks, 60 buildings (9–110 m, taller core), 4 parks, 12 m
  streets, two 32 m boulevards; 526 floors, capacity 18 179; population fixed at 60 % of that.
- **Scenarios:** 16-direction wind rose (prevailing westerly) × release patterns: uniform over
  open ground (the project's Ω), along the boulevards ("traffic"), random clusters.
- **Zoning LP:** minimise expected dose per person; population fixed, each floor 30–100 %
  occupied, at most 30 % of people moved from the baseline (occupancy ∝ capacity).
- **Held-out tests:** the 8 wind directions between the training ones, crossed with
  - **A** the familiar release types (uniform, traffic), or
  - **B** three random-cluster releases never seen in training.

## Results

| zoning trained on | test A: new winds (flat mean) | test B: new winds + new releases (flat mean) | B scenarios made worse |
|---|---|---|---|
| naive: one wind (W), uniform release | 16.6 % | 6.7 % | 8 / 24 |
| robust: 8 winds × {uniform, traffic} | **23.0 %** | 4.2 % | 12 / 24 |
| robust + clusters: 8 winds × {uniform, traffic, 5 cluster draws} | 14.3 % | **10.1 %** | 5 / 24 |

(Flat mean = every wind direction weighted equally. With the rose weighting, the naive zoning
looks better than it is, because the rose is dominated by near-westerly winds, the one it was
trained on.)

1. **The naive zoning does what we feared:** it moves people to the upwind (west) half
   (figure 3). Its gain falls from +21–22 % for near-westerly winds to +10–13 % for easterly
   ones. The exploit is mild here only because moving people to upper floors helps in every
   wind, and that part carries it.
2. **Training over the wind rose generalises across wind directions** (test A: 23.0 % vs 16.6 %).
3. **It does not generalise to a kind of release it never saw.** Trained only on uniform and
   boulevard releases, the "robust" zoning made 12 of 24 cluster scenarios worse (test B).
   Adding other random draws of the same cluster family to training restored it (10.1 %,
   5 / 24 worse). **Robustness extends only as far as the scenario families in training**, and
   only the held-out test revealed that; the training numbers looked fine.
4. **Covariance decomposition holds exactly:** all zonings leave the geometry's mean dose
   unchanged (1.152e-3) and change only the alignment term (figure 4).
5. **In this toy, the robust gain is all vertical:** between floors +11–13 %, between
   buildings −2 to −4 % on test B. No horizontal rule (e.g. "wide streets") survived the
   held-out test; the street-W/H signal flips between strategies. That is the method correctly
   refusing to certify a horizontal pattern. The vertical result follows directly from the
   assumed `LZ` profile, so in the real pipeline it has to come from the CFD instead.
6. **Knowing the release would be worth a lot:** regret vs the per-scenario optimum ≈ 77 %.
   Pre-planned zoning captures only a small part of what a scenario-specific plan could.
7. **More freedom overfits:** held-out gain peaks at a 20 % relocation budget, then falls
   back (figure 4). Allowing more change buys more training-specific moves, not more
   robustness.

## Figures

1. `1_geometry_scenarios.png`: geometry, wind rose (train / held out), release patterns
2. `2_dose_fields.png`: ground-level dose for three scenarios
3. `3_zoning_and_validation.png`: where each zoning moves people; held-out results per scenario
4. `4_decomposition_budget.png`: covariance decomposition; gain vs relocation budget
