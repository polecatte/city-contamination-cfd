# GPU kernel audit — `lbm_kernels.cu` (837 lines, read line-by-line)

Scope: the CUDA kernel only (CPU backend not audited, per request). Goal: find
bugs/inconsistencies, eliminate hidden hardcoded values we may want to tune, and
verify the code does what we intend.

## Verified correct (no change needed)
- **D3Q19 constants** CX/CY/CZ/OPP/W19 — standard; OPP pairs check out; weights
  1/3, 1/18, 1/36 correct. **MNRM2** moment norms and the **mrt_fwd/mrt_inv**
  transforms match d'Humières (2002). **meq_compute** equilibrium moments standard.
- **wale_nut** — WALE eddy viscosity (Nicoud & Ducros 1999); Cw passed in (already
  configurable); dx=1 absorbed into Cw² as documented.
- **SHELL scalar handling** — partial bounce-back `sigma=1−pm` with surface
  deposition `a`; for solid buildings (pm=0) this is full reflection + façade
  deposition, no penetration. Matches the intended solid-building model.
- **Deposition-velocity closure** (upload_geometry) — `α = v_dep_lb / W7_axial`,
  documented, clamped to [0,1] with a reported count. Sound.
- **export / results / time-averaging / checkpoint** functions — straightforward
  device→host copies and thrust reductions; no issues.

## Bugs found and FIXED
1. **Hardcoded viscosity/diffusivity floors inside the kernel** (was: flow
   `if(nu_eff<1e-3) nu_eff=1e-3`; scalar `if(D_eff<1e-3) D_eff=1e-3`). These
   silently overrode the configurable `nu_floor`/`D_floor` and capped the maximum
   reachable Reynolds number regardless of `NU_FLOOR`. This is the same bug class
   as the cosmetic-floor issue. **Fix:** removed both. `nu0`/`D0` are now floored
   once, in `lbm_solver.cpp` (single source of truth), and `nut≥0`, so
   `nu_eff,D_eff>0` (τ>0.5) is guaranteed without a second hidden floor.
2. **Data race on `I.nut`.** `kern_flow` *wrote* `nut_out[id]` while `kern_scalar`
   *read* `nut_in[id]` concurrently on separate streams (`sf`,`ss`) with no
   intervening sync — undefined behavior; the scalar's eddy diffusivity
   nondeterministically used this-step or last-step `nut`. **Fix:** WALE is now
   computed in a dedicated `kern_nut` on the default stream, after `kern_macro`
   and before the event that gates the concurrent kernels. Both `kern_flow` and
   `kern_scalar` now only READ the finalized `I.nut`. Deterministic; still
   one-step lagged by construction (gradients use stored velocity), but now
   lagged *deterministically* rather than by race.

## Hidden hardcoded value FIXED
3. **Bare `8.f` settling factor** (`sa = ws*8.f`) in two kernels — it is the D3Q7
   closure `1/W7_axial`, documented in upload_geometry but duplicated as a magic
   number. **Fix:** replaced with named `INV_W7_AXIAL` constant, so it stays
   consistent if the lattice weights change.

## Flagged (NOT changed — need a decision; not bugs)
- **`S_GHOST[]`** MRT ghost relaxation rates (1.19/1.4/1.2/1.98). Standard
  d'Humières values, but hardcoded; they govern high-Re stability and are the
  natural knob if we want to push Re. Candidate to expose as config. Left as-is.
- **Lateral walls = implicit zero-gradient.** The clamped pull
  (`max(0,min(nx-1,…))`) makes non-inlet/outlet/top faces a copy/Neumann
  condition. A modeling choice worth documenting; not a bug.
- **Legacy `adv==2` path** uses c_s²=1/3 in the scalar equilibrium, which has the
  known advection/diffusion normalization mismatch. The DEFAULT path (`adv≠2`)
  correctly uses operator-split advection + c_s²=1/4 TRT diffusion. Only relevant
  if someone selects the legacy mode.
- **Vestigial params:** after moving WALE out, `kern_flow` no longer uses its
  `Ux/Uy/Uz`/`Cw` arguments (still passed; harmless). Could be removed in a later
  cleanup; left to keep this diff minimal.

## Verification done here / still required
- Device code syntax/type-checks clean as host C++ (CUDA keywords neutralized) —
  catches typos/type errors in the kernel bodies, including the new `kern_nut`.
- **NOT verified:** nvcc acceptance and runtime behavior. `nvcc` is unavailable in
  the dev environment, so these changes ship UNTESTED on GPU. Rebuild on the A4000
  and confirm: (a) it compiles; (b) the Re sweep now produces THREE DIFFERENT
  md5s; (c) two identical runs match bit-for-bit (determinism, confirming the
  race fix). Then `re_compare.py` gives a real verdict.
