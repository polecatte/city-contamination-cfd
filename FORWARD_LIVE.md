# Live-flow forward dose regime (production default)

The production exposure path: a **linear forward** transport of a **source
distribution** over the **live (time-varying) turbulent flow**, using the **QUICK**
scalar scheme, accumulating a **cumulative population dose** over a defined exposure
window. This reverts to the original live-flow Phase-B forward scalar (rather than
the frozen-mean adjoint), with QUICK swapped in for van Leer.

## Why this regime

- **Turbulent fidelity.** Transport rides the instantaneous LES velocity, so plume
  meander and intermittency are resolved, not smoothed into a gradient-diffusion
  mean. This addresses the frozen-flow over-accentuation concern directly.
- **Linear ⇒ source distribution in one run.** QUICK is a fixed linear stencil (no
  limiter), so for a given flow realization the source→dose map is linear:
  J(Σ sᵢ) = Σ J(sᵢ) exactly (verified: flux(2C)=2·flux(C)). Releasing the whole
  candidate source distribution simultaneously in ONE forward run yields the summed
  (or, ÷|Ω|, the averaged) dose — no per-location sweep.
- **QUICK, not van Leer.** van Leer's limiter is nonlinear (solution-dependent), so
  it would break that superposition; QUICK keeps it exact while still suppressing
  the numerical diffusion that first-order upwind would add.

## What changed in the code (implemented + verified)

- **QUICK added to the D3Q7 forward scalar** (`faceflux`, mode 3, in
  `lbm_kernels_cpu.cpp`): Cface = (6C_U + 3C_D − C_UU)/8 on the existing 4-point
  stencil, with a first-order-upwind fallback where the far-upstream cell is a solid
  or off-domain (matching the adjoint's QUICK). Verified: exact on a linear ramp,
  correct boundary fallback, and linear. Compiles.
- **`SCALAR` env selector** in `lbm_solver.cpp`: `SCALAR=quick|vanleer|upwind|central`
  overrides the scheme and prints it in the banner. The production forward driver
  sets QUICK; the numdiff cross-check and other diagnostics keep their explicit
  choice (no silent override).
- **GPU mirror pending**: the same mode-3 branch must be added to `lbm_kernels.cu`
  and GPU-verified (as HRR was) before A4000/H100 production runs.

## The objective (cumulative dose, explicit window)

Per cell the solver already accumulates the time-integrated air concentration
(TIAC) over Phase B, Θ(x) = ∫₀ᵀ C(x,t) dt (`CINT`). The population dose is

    J = ⟨w, Θ⟩ = Σ_x w(x) Θ(x)

with w the F_inf-weighted effective inhabitance (INFILTRATION_MODEL.md). **T is a
physical exposure window** (the Phase-B release duration), NOT a wall-clock artifact
— the number means "population dose accumulated over an exposure period T," and the
paper must state T. (This is the T-dependence the steady-state path avoided; here it
is accepted and made explicit.)

## Caveats designed in

- **One run = one turbulent realization.** A live plume meanders; a single run is a
  sample with turbulent scatter. Make Θ meaningful by (a) accumulating over many
  eddy-turnovers (long T ⇒ effectively time-averaged), and/or (b) ensemble-averaging
  over release phases / flow states. A single short run is noisy.
- **Cost.** Live time-resolved transport is far heavier than the steady frozen
  solve, and (for turbulent robustness) may need an ensemble. This is the accepted
  price for turbulent fidelity.
- **No per-source attribution.** One combined forward run gives the aggregate dose,
  not the per-location breakdown the adjoint's φ field provided. If attribution is
  needed, the adjoint path (retained, verified) remains available and is cheaper for
  that purpose.

## Implemented (this build, compile-verified)

- **QUICK forward scalar** (mode 3) — linear, exact on a ramp, boundary fallback.
- **Burst release** (`Config.burst_release`) — impulse: full mass injected at the
  first Phase-B step, source off after, via solver-side Q_source control (no kernel
  surgery). Mass conserved; `total_emitted` set to the burst mass.
- **Mass-budget termination** (`Config.clearance_frac`, default 0.01) — Phase B
  stops when airborne mass < 1% of released (99% deposited/advected out), so Θ is
  self-terminating; warns if the cap is hit first.
- **Per-run RFG seed** (`Config.abl_seed`) — decorrelates turbulence between
  ensemble members.
- **Full TIAC getter** (`copy_tiac_to_host`) — the Θ=∫C dt field for the dose.
- **`forward_live.cpp` ensemble driver** — N bursts at N seeds → J=⟨w,Θ⟩ per member,
  reports mean ± std ± min/max. Builds and links against the solver.

## Remaining integration (architected)

1. **Simultaneous source DISTRIBUTION (per-cell mask).** The driver currently fires
   a single-cell burst per member. A burst over the whole Ω set at once needs a
   per-cell source mask in the kernel (an O(1)-per-cell `if(inject_now && mask[id])`
   check). By linearity (QUICK verified linear) the distribution dose equals the sum
   of per-cell bursts, so the aggregate is also recoverable as a loop over Ω without
   the mask — the mask is the efficiency option, not a correctness requirement.
2. **GPU mirror** of the QUICK scalar (mode 3) in `lbm_kernels.cu`, GPU-verified as
   HRR was, before A4000/H100 runs.
3. **Full-run validation.** The pieces compile and the logic is unit-checked, but a
   full live burst-to-clearance ensemble has not been run (needs the GPU / long
   compute). First real run should confirm the mass budget closes at ~99% and the
   ensemble spread is sane.

## Relationship to the retained adjoint path

The frozen-flow steady adjoint (ADJOINT_SOLVER.md) is not deleted — it stays as the
cheap, per-source-attribution, turbulence-averaged option, and as the validation
partner (the live-forward run is the high-fidelity reference the frozen-flow mean
should be checked against). The live-forward QUICK path becomes the default because
it resolves turbulence and computes the distribution dose in one linear run; the two
share the same QUICK operator, so they remain consistent.


## Spin-up before release

The live-forward burst path releases after **3 flow-throughs** of spin-up
(`Config.spinup_ft=3`; a flow-through = L/U, so the physical spin-up is
resolution-independent). No averaging phase is needed (unlike the frozen-flow
path) — release begins once turbulence is established. Three flow-throughs (vs the
solver default of 2) follows the Phase-0 finding that a bluff-body wake transient
was still clearing at 2; the adequacy is to be justified per-case by confirming the
near-source statistics have settled. Standard: COST 732 / Tominaga 2008 require
statistical stationarity / sampling-independence (a principle, not a fixed
flow-through count) — the 3-FT spin-up is our implementation of that requirement,
and the count + any per-case check should be reported. `SPINUP_FT` env overrides.

## Sources

- Leonard, B.P. (1979). A stable and accurate convective modelling procedure based
  on quadratic upstream interpolation (QUICK). *Comput. Methods Appl. Mech. Eng.*
  19:59–98.
- van Leer, B. (1974). Towards the ultimate conservative difference scheme II.
  *J. Comput. Phys.* 14:361–370. (The limiter QUICK replaces here, retained as an
  option.)
