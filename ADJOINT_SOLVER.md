# The adjoint scalar-transport solver

## Purpose

The adjoint solver computes a population-weighted exposure objective and, in a single
additional solve, its **exact source–receptor sensitivity field** — the marginal
contribution of a unit emission at every location to total exposure. Run forward it
propagates concentration from a source to receptors; run in reverse (adjoint mode) it
propagates receptor sensitivity backward to candidate sources. The two directions are
discrete transposes of one another and agree to machine precision. This is the standard
backward/adjoint formulation of atmospheric transport used for source attribution,
sensor siting, and gradient-based design (Marchuk 1986; Pudykiewicz 1998;
Seibert & Frank 2004).

It operates on the **frozen mean flow** produced by the LBM: the time-averaged velocity
field **ū** and eddy viscosity ν_t from Phase A2 of the wind solve are held fixed, which
linearizes transport and makes the exact discrete adjoint available (see *Architecture fit*).

## Forward governing equation

Scalar concentration c(**x**,t) is advected by the frozen mean flow, mixed by gradient
diffusion, gravitationally settled, and lost to surfaces by dry deposition:

    ∂c/∂t + ∇·(ū c) + ∂(w_s c)/∂z = ∇·(D_eff ∇c) + s(x,t)

with

- **effective diffusivity** D_eff = D_mol + ν_t / Sc_t — molecular plus turbulent, the
  eddy viscosity from the LES closure divided by the turbulent Schmidt number
  (gradient-diffusion hypothesis; Sc_t ≈ 0.7, Tominaga & Stathopoulos 2007);
- **settling velocity** w_s from Stokes' law with the Cunningham slip correction,
  w_s = ρ_p d_p² g C_c / 18μ (Seinfeld & Pandis 2016);
- **source** s(**x**,t);
- **boundary conditions** matching the LBM scalar BCs: open (advective outflow) on the
  streamwise faces, zero-flux on the lateral/top faces, and a partially absorbing
  surface flux F = v_d c at the ground and building faces, where v_d is the dry
  deposition velocity (Zhang et al. 2001). A passive gas is the special case
  w_s = 0, v_d = 0.

## Discretization and the linear propagator

Advection uses the third-order upwind-biased QUICK scheme (Leonard 1979), with a
first-order upwind fallback where a building or domain boundary blocks the far-upstream
stencil cell; diffusion is central. Because the flow is frozen and the scheme is linear
(no solution-dependent flux limiter), one time step is an affine update

    c^{n+1} = M c^n + Δt · s^n ,

where **M** is a fixed sparse linear operator, implemented as a race-free gather kernel.
The production quantity of interest is the **steady concentration field** C_ss(**x**),
reached by advancing the continuous source until the field stops changing (the
exposure objective below). Equivalently, the time-integrated air concentration (dosage)

    Θ(x) = ∫₀ᵀ c(x,t) dt  ≈  Δt Σ_n c^n(x)

is used for the accidental-release (burst) variant, where it converges as the puff clears.

## Exposure objective (steady-state, rate-normalized)

The production objective is a **steady-state exposure rate**, not a time-integrated
dose. A continuous emission is advanced on the frozen flow until the concentration
field stops changing (loosely — exact steadiness is not required), giving the steady
field C_ss(**x**). Exposure is the inhabitance-weighted steady concentration,

    J = ⟨w, C_ss⟩ = Σ_x w(x) C_ss(x)     [exposure per unit time; lower is better],

where the receptor field w(**x**) is the **effective inhabitance** — the spatially and
temporally resolved population distribution supplied by the city model (home/work/park
time-activity budget). Because C_ss is the converged field, J is the rate at which dose
accumulates (dΘ/dt at steady state) and is **independent of run length**. This avoids the
arbitrariness of a fixed-window time-integrated dose Θ = ∫₀ᵀ C dt, whose value grows
with the (otherwise meaningless) simulation duration T. The steady field satisfies the
linear balance (I − M) C_ss = Δt · s, so J remains linear in the source. The fixed-point
iteration C ← M C + Δt·s converges geometrically because open (advective-outflow)
boundaries and surface deposition make M a contraction (spectral radius < 1): a steady
state exists precisely because the source drains through the domain rather than
accumulating. Convergence is declared on the objective itself — the relative change of J
per probe interval falling below tolerance — i.e. "the exposure rate has stabilised,"
never a fixed step count.

*(An accidental-release variant uses a burst of total mass M run to near-complete
clearance, for which Θ = ∫₀^∞ C dt converges and the mass budget M = M_out + M_dep
closes; deposited mass then enters as a separate linear penalty term ⟨w_dep, M_dep⟩.
The steady-state rate above is the default for chronic emitters.)*

## The adjoint (reverse) solve

Since M is fixed and linear, C_ss — and hence J — is linear in the source s, and the
discrete adjoint is the exact transpose Mᵀ. The steady adjoint field φ solves the
**steady transposed balance**

    (I − Mᵀ) φ = Δt · w ,   obtained by iterating   φ ← Mᵀ φ + Δt · w   to steady state,

which is the exact dual of the forward steady solve (rather than a fixed-window backward
accumulation). It yields the receptor-influence field

    φ(x) = ∂J / ∂s(x) ,

the sensitivity of the steady population exposure *rate* to a unit emission at **x** — the
source–receptor "footprint." Forward and reverse solves satisfy the duality identity

    J = ⟨w, C_ss(s)⟩ = ⟨φ, s⟩ ,

i.e. transporting the source forward to the receptors and transporting the receptors
backward to the source give the same objective (verified here to ~1e-6, limited only by
the steady-state tolerance). Because φ covers every candidate source cell at once, the
"source could be anywhere" ensemble over an outdoor set Ω reduces to a masked sum,
J_ensemble ∝ Σ_{x∈Ω} φ(x), from a single reverse solve. Physically, φ is the concentration
obtained by driving the *adjoint* (time-reversed, transposed) operator with the receptor
field w as its source (Marchuk 1986; Pudykiewicz 1998; Seibert & Frank 2004).

**Verification.** Correctness reduces to the adjoint identity ⟨M x, y⟩ = ⟨x, Mᵀ y⟩,
checked to machine precision by a dot-product (random-vector) test — the standard adjoint
consistency check (Errico 1997; Giles & Pierce 2000). This is far stronger than a
physical plausibility check and is what licenses using φ as an exact gradient.

**Scope of "exact."** Throughout, *exact* means the **discrete adjoint**: φ is the exact
transpose of the discretized forward operator M, so it is the exact gradient of the
*discrete* objective J (verified above). It is **not** claimed to be exact with respect
to the continuous advection–diffusion PDE — M carries the usual QUICK/central
discretization error — nor does exactness imply physical accuracy. The adjoint
differentiates the forward *model* exactly; the modelling approximations (frozen mean
flow, isotropic gradient-diffusion ν_t/Sc_t, and a metric that is linear in the source)
live in that forward model, not in the adjoint. The value of the discrete-adjoint
approach is precisely this separation: given the model, the source–receptor sensitivity
is obtained without additional approximation, so any error in J is inherited from the
forward model and not introduced by the reverse solve.

## Architecture fit

    ABL inlet ─▶ LBM wind solve (HRR + WALE)         [lbm_solver / lbm_kernels]
                    │  Phase A1 spin-up
                    │  Phase A2 time-average ─▶ frozen mean flow ū, ν_t, geometry
                    ▼
              adjoint transport solver                [adjoint_transport.h]
               ├─ forward: s ─▶ C_ss ─▶ J = ⟨w, C_ss⟩   (source → exposure rate)
               └─ reverse: w ─▶ φ = ∂J/∂s             (exposure → source footprint)
                    │
                    ▼
        source attribution · sensor siting · exposure-driven design

The wind field is solved once by the Lattice-Boltzmann core (HRR collision,
Jacob–Malaspinas–Sagaut 2018; WALE subgrid viscosity, Nicoud & Ducros 1999) and its
converged mean is frozen; every subsequent transport or adjoint solve reuses that field,
so the expensive turbulence solve is amortized across all source/receptor queries. The
production transport scheme (linear QUICK) is deliberately the one the adjoint transposes,
so the reverse solver is the exact dual of the forward model that is validated against
dispersion benchmarks — not a separate approximation. The forward path supplies
concentration and deposition fields; the reverse path supplies the exact objective
gradient with respect to source placement, which is the quantity consumed by the
exposure-driven optimisation layer (continuous source/vent placement in particular, where
the adjoint gradient is far cheaper than a surrogate).

## References

- Errico, R.M. (1997). What is an adjoint model? *Bull. Amer. Meteor. Soc.* 78, 2577–2591.
- Giles, M.B. & Pierce, N.A. (2000). An introduction to the adjoint approach to design. *Flow, Turbulence and Combustion* 65, 393–415.
- Jacob, J., Malaspinas, O. & Sagaut, P. (2018). A new hybrid recursive regularised BGK collision model for LES of turbulent flows. *J. Turbulence* 19, 1051–1076.
- Leonard, B.P. (1979). A stable and accurate convective modelling procedure based on quadratic upstream interpolation (QUICK). *Comput. Methods Appl. Mech. Eng.* 19, 59–98.
- Marchuk, G.I. (1986). *Mathematical Models in Environmental Problems.* Elsevier.
- Nicoud, F. & Ducros, F. (1999). Subgrid-scale stress modelling based on the square of the velocity gradient tensor (WALE). *Flow, Turbulence and Combustion* 62, 183–200.
- Pudykiewicz, J.A. (1998). Application of adjoint tracer transport equations for evaluating source parameters. *Atmos. Environ.* 32, 3039–3050.
- Seibert, P. & Frank, A. (2004). Source–receptor matrix calculation with a Lagrangian particle dispersion model in backward mode. *Atmos. Chem. Phys.* 4, 51–63.
- Seinfeld, J.H. & Pandis, S.N. (2016). *Atmospheric Chemistry and Physics*, 3rd ed. Wiley. (Stokes settling; Cunningham slip correction.)
- Tominaga, Y. & Stathopoulos, T. (2007). Turbulent Schmidt numbers for CFD analysis with various types of flowfield. *Atmos. Environ.* 41, 8091–8099.
- Zhang, L., Gong, S., Padro, J. & Barrie, L. (2001). A size-segregated particle dry deposition scheme for an atmospheric aerosol module. *Atmos. Environ.* 35, 549–560.
