# Ranking-stability check (`ranking_stability.cpp`)

## Question
The diffusion benchmark showed the linear-upwind objective transport adds
numerical diffusivity D_num ≈ 0.05 (LU), ~49× van-Leer's, overlapping the physical
eddy-diffusivity band. That matters for *absolute* far-field concentrations. But
the optimizer only cares whether this changes **which designs look best**. This
test answers that directly.

## What it does
Scores N city designs two ways. IMPORTANT (corrected after inspecting the solver): they differ in TWO ways, not one — the production scalar advects on the LIVE resolved-turbulent LBM flow, while the linear-upwind objective runs on the FROZEN TIME-MEAN flow (what the reverse solver uses). So this compares the reverse-solver objective vs the production truth, conflating numerical scheme AND mean-vs-turbulent flow:
- **J_vanleer** — Σ w·TIAC from the production D3Q7 van-Leer scalar (the accurate
  reference, run as part of the LBM).
- **J_upwind** — Σ w·TIAC from the linear-upwind forward on the SAME frozen mean
  flow, SAME receptor w (effective inhabitance), SAME source.

The flow representation differs (live-turbulent vs frozen-mean) AND the scheme differs; receptor and source cell are shared. Then it reports Spearman ρ, Kendall τ, Pearson(log J),
and the top-k overlap of the best (lowest-J) designs, with a STABLE/SHIFT verdict
(STABLE if ρ>0.9 and the best designs fully overlap).

## Interpretation
- **STABLE** → the diffusion penalty is a quantified *absolute* bias (state it in
  the writeup, especially for far-field exposure) but it does not threaten the
  optimization; the cheap linear-upwind objective is safe to optimize against.
- **SHIFT** → numerical diffusion changes design preference; escalate to the
  frozen-limiter adjoint (Option 2), which keeps van-Leer accuracy while remaining
  adjoint-consistent.

## Honest caveats
- **Point source, not the uniform mask.** The production van-Leer is nonlinear and
  cannot cheaply produce the uniform-open-space-source objective the optimizer
  actually uses (it does not superpose point sources). So both schemes are scored
  from a single point source. This is a *conservative* proxy: the uniform source is
  a spatial average over release points and is expected to rank at least as stably.
  If point-source rankings are stable, uniform-source rankings almost certainly are.
  If they shift, escalate to a uniform-source comparison (requires a van-Leer
  forward driven by the distributed mask).
- **Passive tracer** (no settling/deposition) removes settling as a variable. The
  size-dependent settling is identical between schemes, so it does not affect the
  scheme-vs-scheme ranking comparison.
- **N flow solves.** The cost is dominated by one LBM warmup per design; the two
  scalar evaluations are negligible. Use the GPU build.

## Run
```
# fast: verify the statistics (no LBM)
bash build_diffusion.sh && ./ranking_stability selftest

# full test on real designs (GPU): one flow solve per design
bash build.sh            # or build_gpu.sh, to produce kernels.o + solver.o
bash build_diffusion.sh city
./ranking_stability run 6 512 60 4000     # N=6 designs, 512 m, 60 s release
```
Writes `ranking_stability.csv` (per-design params + J_vanleer + J_upwind + ranks).

## Verification
`ranking_stability selftest` checks the statistics on synthetic data: a monotone
perturbation returns ρ=1.000 / 100% overlap → STABLE; a scrambled ranking returns
ρ≈0.62 / 50% overlap → SHIFT; identity Kendall τ=1.000. (Verified.)
