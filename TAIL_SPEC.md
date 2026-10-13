# Tail extrapolation of the exposure integral (TAIL=1)

Status: implemented and replay-validated on one city (2026-10-13). Code: `tail_fit.h` (the rule),
`urban_flow.cpp` (sampling, stop, Θ correction), `tests/tail_replay.cpp` (offline replay).

## 1. Problem

J is the dose integral ⟨w, Θ⟩ / M with Θ(x) = ∫₀^∞ C(x, t) dt. A run cannot integrate to
infinity. It stops when the airborne mass in the control volume falls below `CLEAR_FRAC` × its
peak (default 1 %), and the dose still to come is dropped. Two consequences:

- **Bias.** J is underestimated by the missing tail. That would be design-dependent if
  slow-clearing (stagnant) designs lost more of it, which would compress the spread between
  designs.
- **Cost.** The last few percent of the mass take a long time to leave, so the burst phase of
  every run is longer than its information content warrants.

## 2. Method

After the release, the concentration field relaxes, at late times, onto the slowest flushing
mode of the city's flow: its shape stops changing and its amplitude decays like e^(−k t). The
receptor-weighted dose rate r(t) = d⟨w, Θ⟩/dt = ⟨w, C(t)⟩ then decays at the same rate, and

    ∫_{t_s}^∞ r dt = r(t_s) / k,     Θ(x) gains C(x, t_s) / k in every cell.

This is the standard extrapolation of an area under a concentration–time curve to infinity in
pharmacokinetics, AUC_∞ = AUC_t + C_last / λ_z (Gibaldi & Perrier, *Pharmacokinetics*, 2nd ed.,
1982). There the regulators' acceptance rule is that the extrapolated part stays a minor share
of the total, with ≤ 20 % commonly used. It is also the exponential tail of tracer-decay
ventilation measurements (Sandberg & Sjöberg 1983, *Build. Environ.* 18, 181). Nothing new is
built here, only the application.

## 3. Algorithm (`tail_fit.h`, identical in the solver and the replay tool)

1. **Sampling.** After the pulse, every `TAIL_EVERY` burst steps (default 20, rounded up to a
   multiple of `CLEAR_EVERY`), record (t, ⟨w, Θ⟩, airborne CV mass).
2. **Rate samples.** Difference quotients of ⟨w, Θ⟩ between consecutive samples, at the midpoints.
3. **Window.** Only pairs whose earlier sample already has airborne mass ≤ `TAIL_FROM` × peak
   (default 0.2), so the fast early modes are excluded.
4. **Fit.** Least-squares line through ln r(t): slope −k, R².
5. **Acceptance.** All four must hold:
   - at least 8 samples;
   - R² ≥ `TAIL_R2` (0.98): a clean single exponential;
   - the window saw at least a factor e^0.7 ≈ 2 of decay;
   - the extrapolated share of J ≤ `TAIL_MAXFRAC` (0.15).
6. **Stop.** At the first sample where airborne mass < `TAIL_CLEAR` × peak (default 0.05) and the
   fit passes. Otherwise the run continues, and `CLEAR_FRAC` (default 0.01) stays the fallback stop.
7. **Correction.** After the loop (both paths; on a GPU after the final fetch), add C(x, t_s)/k to
   Θ on the cells the solver integrates (fluid, porous, sponge). If the run ended at the fallback
   stop, the tail is still added whenever the fit passes the first three tests. Then Θ, J and
   Stage C all include it.

## 4. Interface

| variable | default | meaning |
|---|---|---|
| `TAIL` | 0 | 1 enables (off by default so existing gates reproduce) |
| `TAIL_CLEAR` | 0.05 | early stop level, airborne mass / peak |
| `TAIL_FROM` | 0.2 | fit window starts below this level |
| `TAIL_R2` | 0.98 | minimum R² of the ln-rate line |
| `TAIL_MAXFRAC` | 0.15 | maximum extrapolated share of J |
| `TAIL_EVERY` | 20 | sampling interval (burst steps) |

- **Log.** One `TAIL applied / NOT applied` line with k (and its e-folding time), R², the number
  of samples, the decay seen, airborne fraction at the stop, J before and after, and the share
  the fit predicted.
- **`meta_flow.txt`.** `tail_applied`, `tail_early_stop`, `tail_k_per_s`, `tail_r2`, `tail_n`,
  `tail_span`, `tail_air_frac_at_stop`, `tail_J_part`, `tail_fit_frac`.
- **Not extrapolated.** The deposition map: the mass still airborne at the stop is in the budget
  as airborne, not as deposited.

## 5. Validation

**Test case.** A 300 m city (builder rev 9, compact domain, dx 8, BGK + WALE) run to 0.1 %
airborne, time series every 50 steps. `tests/tail_replay.cpp` walks the time series as the
solver would and stops at each level. The reference is J at 0.1 % plus its own tail (0.004 %).

| stop when airborne < | burst steps | J, plain stop | J, with tail | extrapolated share |
|---|---|---|---|---|
| 10 % (first passing fit) | 8 050 | −0.69 % | **−0.08 %** | 0.6 % |
| 5 % (default) | 9 400 | −0.33 % | **−0.04 %** | 0.3 % |
| 2 % | 11 300 | −0.12 % | −0.00 % | 0.1 % |
| 1 % (old default, no tail) | 12 550 | −0.05 % | — | — |
| 0.1 % (reference) | 17 065 | 0 | — | — |

Fit quality: R² 0.998–0.999. The e-folding time is 112 s, about 2 min for a 300 m city at
4 m/s.

**Reading.**

- Here the plain 1 % stop was already within 0.05 % of the converged J. The receptor-weighted dose
  clears faster than the airborne mass, much of which lingers downstream of the city, away from
  people. So the truncation bias was small in this case.
- What the tail buys is time: the 5 % stop with the tail is as accurate as the 1 % stop without
  it, with 25 % fewer burst steps (36 % at the earliest passing fit).
- The burst is about half of a run, so that is roughly 12–18 % off every production run.

**Limits.**

- One city. Designs with stagnant zones (deep canyons, closed courtyards) may have a second,
  slower mode. Then the fit fails the R² or decay tests and the run falls back to `CLEAR_FRAC`;
  it does not extrapolate wrongly. That is why the acceptance tests exist.
- The ranking runs record their time series every 200 steps (`lab_openlb.sh rank`), so
  `tail_replay` can repeat this check on 20 more runs before production uses `TAIL=1`.

## 6. Production setting

`TAIL=1 TAIL_CLEAR=0.05 CLEAR_FRAC=0.01`. Expected error in J ≲ 0.1 %, below the inlet-seed
noise (≈ 0.5 %).
