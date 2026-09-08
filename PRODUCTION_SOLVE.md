# Production contaminant exposure solve — runbook

End-to-end recipe from a frozen mean flow to the single objective number the
optimiser consumes. This is the steady-state, linear-QUICK, rate-normalized path.
Driver: `exposure_solve.cpp` (`run_exposure.sh` to build + launch). Math and
citations: `ADJOINT_SOLVER.md`.

One command:

    COLL=hrr HRR_SIGMA=0.98 MAX_WARMUP=80000 bash run_exposure.sh 2681829 0.5 300000 2.0 [w.csv]

The stages below are what that runs.

## 0. Input: the frozen mean flow

The LBM has run (spin-up → Phase A2 time-averaging) and produced the **mean
velocity** ū(x) and **eddy viscosity** ν_t(x), plus the cell-type geometry. These
are held fixed for everything below (`copy_mean_flow_to_host`).

## 1. Assemble the transport operator M  (adj::Field)

Copy ū, ν_t into the transport field and set per-cell physics:
- effective diffusivity D_eff = D_mol + ν_t / Sc_t  (Sc_t ≈ 0.7);
- settling w_s (Stokes + Cunningham) and deposition v_d (Zhang) — 0 for a passive gas;
- cell types so scalar BCs match the LBM (open outflow streamwise, no-flux
  lateral/top, deposition at ground + building faces);
- transport step dt = `stable_dt(F)` (CFL over advection and diffusion).

This fixes the linear propagator **M**: one step is `C ← M·C` (QUICK advection +
central diffusion + deposition/settling), a race-free gather kernel.

## 2. Receptor field w  (effective inhabitance)

w(x) = the spatially/temporally resolved population weight — occupancy by land use
and floor area, over the home/work/park time budget, with indoor cells folded
through the infiltration factor (outdoor → indoor dose). Fixed per population
scenario. In `exposure_solve` this is loaded from a w-file (`x_m,y_m,z_m,weight`)
— **this is where `city_builder7`'s `eff_inh` plugs in** — or, absent a file,
defaulted to a pedestrian-level proxy (street-level fluid cells adjacent to
buildings).

## 3. Source specification

Either a concrete emission field s(x) (unit strength; everything is linear in s),
or — the production default — the **candidate set Ω** (outdoor ground cells),
representing "one emitter of unknown, uniformly-likely location." No explicit s in
that case.

## 4. The steady solve  (one iteration to a converged field)

M is a contraction (outflow + deposition drain the source, spectral radius < 1),
so a continuous source reaches a stationary field, found by fixed-point iteration.
Convergence is declared when the objective stops changing per probe interval —
"the exposure rate has stabilised," never a fixed step count — so the result is
**duration-independent**.

- **Production (source-anywhere):** `reverse_steady(F, w, φ)` iterates
  φ ← Mᵀ·φ + Δt·w to its fixed point. φ_ss(x) = ∂J/∂s(x) is the steady exposure-
  **rate** sensitivity to a unit source at *every* cell at once.
- **Known source:** `forward_steady(F, s, w, …, &C_ss)` iterates C ← M·C + Δt·s to
  the steady field C_ss, with (I−M)C_ss = Δt·s.

## 5. Form the objective number J

- Known source: **J = ⟨w, C_ss⟩ = ⟨φ_ss, s⟩** (equal by steady reciprocity,
  verified to ~1e-6). Units: exposure **per unit time** per unit emission —
  a rate, **lower is better**.
- Source-anywhere (Ω reduction): **J_ensemble = (1/|Ω|) Σ_{x∈Ω} φ_ss(x)** — the
  mean steady exposure rate over the uniform-prior single-source ensemble. This
  masked sum over the outdoor-ground set is the final scalar, from ONE reverse
  solve regardless of |Ω|.

## 6. (optional) multi-pathway, polydisperse, wind rose

- **Surface contact:** add the linear deposition term
  J_total = α⟨w_air, C_ss⟩ + β⟨w_dep, D_ss⟩ (D_ss = steady deposition-rate field,
  w_dep = surface-contact weight).
- **Polydisperse:** repeat 1–5 per size bin (each with its own w_s, v_d) and
  mass-weight-sum — coarse bins deposit, fine bins ventilate.
- **Wind rose:** M depends on wind direction, so repeat 1 + 4 per sector and
  combine by frequency, J = Σ_sector p_sector · J_sector. That, not a single
  direction, is the honest site objective.

## In the optimiser

Per city-morphology candidate: LBM flow solve → freeze mean flow → stages 1–5 →
one J. The expensive turbulence solve is paid once per candidate; the adjoint then
answers the entire source-placement question in a single additional solve
regardless of |Ω|. Everything stays **linear in the source**, which is what keeps
the adjoint exact and the Ω-reduction valid.

## Assumptions to keep honest

- **Linear exposure metric.** J = ⟨w, C_ss⟩ (and the α/β deposition terms) must
  stay linear in concentration; a threshold or saturating dose–response breaks the
  Ω-reduction and forces per-scenario forward solves.
- **Isotropic closure.** D_eff = D_mol + ν_t/Sc_t is scalar gradient-diffusion; an
  anisotropic ⟨u′ᵢu′ⱼ⟩ tensor (accumulated in Phase A2) would slot in at stage 1 if
  the MUST benchmark shows the plume shape needs it — it stays linear, so the
  adjoint survives.
- **Frozen mean flow.** Coherent mean structures can be over-accentuated vs the
  fluctuating reality; a small frozen-flow ensemble (average φ over several
  averaging windows) blunts this while keeping every solve linear/adjoint-exact.

## Outputs

`exposure_phi_zped.csv` (pedestrian-height φ slice) and `exposure_vel_z2.bin` for
the visualizer, plus the printed `J_ensemble`. Compare J *across designs*; the
absolute lattice value is not meaningful on its own.
