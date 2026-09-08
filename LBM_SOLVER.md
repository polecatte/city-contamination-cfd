# The D3Q19 lattice-Boltzmann airflow solver

## Purpose and rationale

The flow solver computes the turbulent urban wind field, and its **time-averaged
mean** — mean velocity ū(x) and subgrid eddy viscosity ν_t(x) — is frozen and
handed to the scalar-transport and adjoint/exposure layers (`ADJOINT_SOLVER.md`).

Why a lattice-Boltzmann method rather than a classical Navier–Stokes solver? Three
properties matter for this problem. First, **locality**: the method never solves a
global pressure Poisson equation (the expensive, communication-heavy step in
incompressible NS solvers); pressure is local and algebraic, so every update touches
only nearest neighbours. That makes the solver embarrassingly parallel and a natural
fit for the GPU. Second, **geometry handling**: complex urban geometry is imposed by
simply *flagging* cells as fluid or solid — no body-fitted mesh generation — so a
new city layout costs nothing in meshing. Third, **the frozen-mean strategy**:
turbulent flow is expensive to resolve, but the downstream transport/adjoint problem
only needs the *mean* field. Solving the flow once, averaging it to stationarity, and
reusing that mean for every subsequent source/receptor query is what makes the whole
exposure-optimisation loop tractable — the flow solve is amortised across all of it.

## 1. The governing kinetic equation

Unlike a classical solver, the LBM does not discretise the Navier–Stokes equations
directly. It instead evolves a simplified **kinetic** (Boltzmann-like) equation for a
set of particle-distribution populations, whose statistical *moments* provably obey
Navier–Stokes in the continuum limit. This is a mesoscopic description — coarser than
tracking molecules, finer than the continuum fields.

The state is nineteen populations f_i(x,t), i = 0…18, each representing the density of
fictitious particles moving along a lattice velocity **e_i**. They evolve by the
**lattice Boltzmann equation**, which splits into two operations per time step:

    f_i(x + e_i Δt, t + Δt) = f_i(x,t) + Ω_i(f)                              (1)

- **Collision** (the right side, Ω_i): a purely *local* relaxation of the populations
  toward a local equilibrium — this is where viscosity and turbulence modelling live.
- **Streaming** (the left side): each population is copied, unchanged, to the
  neighbour cell its velocity points to — exact, local, and lossless.

The elegance is that the nonlinear advection which makes Navier–Stokes hard becomes
*exact linear streaming* on the lattice; the only nontrivial operation is the local
collision. All the physics is in Ω_i, and through a Chapman–Enskog expansion (§7) the
moments of (1) recover continuity and momentum in the low-Mach limit.

## 2. The D3Q19 velocity set

The name follows the standard DdQq convention: **D3** = three spatial dimensions,
**Q19** = nineteen discrete velocities. They are one rest velocity (e_0 = 0, for
stationary fluid), six face-normal velocities (±x, ±y, ±z, speed 1), and twelve
edge-diagonal velocities (speed √2). Each carries an equilibrium weight, and the set
satisfies the isotropy constraints that make the recovered equations correct:

    w_0 = 1/3,   w_face = 1/18,   w_edge = 1/36,   Σ_i w_i = 1
    Σ_i w_i e_i e_i = c_s² I ,     c_s² = 1/3                                (2)

The weights are not arbitrary: they are chosen so the second- and fourth-order moment
tensors of the velocity set are isotropic, which is exactly what the Chapman–Enskog
analysis needs to yield Galilean-invariant Navier–Stokes. D3Q19 is the usual 3D
choice because it is the minimal lattice with sufficient isotropy at acceptable cost
— D3Q27 is more isotropic but 40% more populations to stream and store, while D3Q15
is cheaper but noticeably less stable. c_s = 1/√3 is the **lattice speed of sound**;
the flow must stay well below it (low Mach) for the method to represent
near-incompressible air. The array OPP[i] gives the index of −e_i, used for
bounce-back walls (§6).

## 3. Macroscopic moments (kernel K1)

The continuum fields are not separate variables — they *are* low-order velocity
moments of the populations, which is how the macroscopic world emerges from the
kinetic one:

    ρ  = Σ_i f_i         (mass  = 0th moment)
    ρ u = Σ_i e_i f_i    (momentum = 1st moment)                             (3)

Pressure follows an isothermal equation of state, p = c_s² ρ — there is no separate
pressure solve. Incompressibility is therefore only *approximate*: density varies by
an amount of order Mach², which is why keeping the flow low-Mach matters. Kernel K1
evaluates (3) for every cell each step and writes ρ, u_x, u_y, u_z unconditionally;
kernel K2 then performs streaming, the turbulence model, collision, and boundaries.

## 4. Equilibrium and collision (kernel K2)

**The equilibrium distribution.** The collision relaxes the populations toward a
discrete local equilibrium — the lattice analogue of the Maxwell–Boltzmann
distribution, expanded to second order in the velocity (a truncated Gauss–Hermite
expansion):

    f_i^eq = w_i ρ [ 1 + (e_i·u)/c_s² + (e_i·u)²/(2 c_s⁴) − u²/(2 c_s²) ]     (4)

With c_s² = 1/3 the three coefficients become 3, 4.5, and 1.5 (as in the code). The
truncation at second order in u is precisely why the LBM is a low-Mach method: the
expansion is only accurate for |u| ≪ c_s.

**Viscosity from relaxation.** How fast f relaxes toward f^eq sets the fluid
viscosity. With relaxation rate ω = 1/τ,

    ν_eff = c_s² ( τ − 1/2 ) = c_s² ( 1/ω − 1/2 )   ⇒   ω = 1 / ( 3 ν_eff + 1/2 )   (5)

Because the effective viscosity ν_eff = ν_0 + ν_t includes the spatially-varying
eddy viscosity (§5), ω is computed *per cell*. Note the danger built into (5): as
ν_eff → 0 (high Reynolds number, the urban regime), τ → 1/2 and ω → 2, the edge of
the stable range. Controlling that limit is the entire reason for the choice of
collision operator.

**The collision operators** (selectable via env `COLLISION`) differ in how they
handle the non-hydrodynamic "ghost" modes — the higher moments of f that Navier–
Stokes does not constrain, and which grow and destabilise the scheme near τ = 1/2:

- **BGK** (single-relaxation): relax *every* population toward equilibrium at the one
  rate ω. Simplest and cheapest, but it lets the ghost modes relax at that same rate,
  so it becomes unstable at the low viscosities urban high-Re flow demands. Provided
  as a baseline.
- **MRT** (multiple-relaxation-time, mode 0): transform the populations into moment
  space (m = M f), relax each moment at its *own* tuned rate — the conserved
  hydrodynamic moments (ρ, ρu) untouched, the shear moments at ω(ν_eff), and the
  ghost moments at fixed damping rates — then transform back. Decoupling the ghosts
  and damping them independently buys substantial stability. (d'Humières 2002;
  Lallemand & Luo 2000.)
- **Regularized / projected-MRT** (mode 1, default): before relaxing, reconstruct the
  non-equilibrium part of f from its second-order Hermite projection, *discarding the
  ghost content altogether*, then BGK-relax the cleaned populations
  (f_i = f_i^eq + (1−ω) f_i^neq,reg). Achieves much of MRT's stability at lower cost.
  (Latt & Chopard 2006.)
- **HRR — hybrid recursive regularized** (mode 2): the production high-Re operator. It
  extends the regularized idea with a *recursive third-order* Hermite reconstruction
  of the non-equilibrium populations and then blends in a finite-difference estimate
  of the stress with weight σ, adding a small, targeted hyperviscosity that damps the
  aliasing which otherwise destabilises the τ → 1/2 limit. The reconstruction
  coefficients are 4.5 = 1/(2 c_s⁴) at second order and 13.5 = 1/(2 c_s⁶) at third (the ×3 iij multiplicity folded in);
  on D3Q19 only the six iij third-order Hermite polynomials are admissible (the
  others vanish identically on the velocity set). This is what lets the solver reach
  the low viscosities turbulent urban flow needs while staying stable. (Jacob,
  Malaspinas & Sagaut 2018; Malaspinas 2015; full derivation in `HRR_ARCHITECTURE.md`.)

## 5. Turbulence closure — WALE large-eddy simulation

The grid cannot resolve every turbulent eddy; large-eddy simulation resolves the
large, energy-containing eddies and *models* the effect of the unresolved
subgrid ones as an added eddy viscosity ν_t, so ν_eff = ν_0 + ν_t feeds (5).

The subgrid model is **WALE** (Wall-Adapting Local Eddy-viscosity; Nicoud & Ducros
1999). Its motivation is a specific failure of the classic Smagorinsky model:
Smagorinsky sets ν_t ∝ |S| (the strain-rate magnitude), but |S| is nonzero in *any*
shear — including the smooth, laminar shear next to a wall — so Smagorinsky
spuriously generates eddy viscosity at walls and requires ad-hoc near-wall damping
functions to behave. WALE is instead constructed from the velocity-gradient tensor
so that it **vanishes in pure shear** and **decays as y³ approaching a wall**, the
correct physical asymptotics, with no damping functions. This matters here because
the building boundary layers — where separation, recirculation, and reattachment set
the whole dispersion pattern — are exactly where Smagorinsky misbehaves.

From the resolved velocity-gradient tensor g_ij = ∂u_i/∂x_j (central differences):

    S_ij  = ½( g_ij + g_ji )                              (resolved strain rate)
    Sᵈ_ij = ½( g²_ij + g²_ji ) − ⅓ δ_ij g²_kk ,  g²_ij = g_ik g_kj  (traceless sym. square)
    ν_t = (C_w Δ)² · (Sᵈ_ij Sᵈ_ij)^{3/2} / [ (S_ij S_ij)^{5/2} + (Sᵈ_ij Sᵈ_ij)^{5/4} ]   (6)

C_w ≈ 0.325 is the model constant and Δ = 1 is the filter width (the grid spacing, in
lattice units, folded into C_w). The numerator uses Sᵈ, the traceless symmetric part
of the *square* of the gradient tensor — the construction that gives WALE its
correct shear and near-wall behaviour. A floor ν_eff ≥ 10⁻³ (τ ≥ 0.503) guards the
relaxation against the unstable limit, and the same gradient tensor is reused by the
HRR collision so it is computed only once per cell.

## 6. Boundary conditions

Cells carry a type: **FLUID** (0), **GROUND** (1, solid floor/wall), **SHELL** (2,
permeable building or park envelope), **INDOOR** (3, enclosed air that carries
inhabitance for the exposure model). Boundaries are applied during K2.

- **Inlet.** The incoming wind is prescribed. With `inlet_profile = 1` it is an
  atmospheric-boundary-layer (ABL) **log-law** — real wind is sheared by ground
  friction, not uniform, and the profile shape governs the whole downstream flow:

      U(z) = (u_* / κ) · ln( (z + z_0) / z_0 ) ,   u_* = κ U_inlet / ln( (z_ref+z_0)/z_0 )   (7)

  Here κ = 0.41 is the von Kármán constant, z_0 is the aerodynamic roughness length
  (larger for rougher terrain), z_ref is the reference height at which U = U_inlet,
  and u_* is the friction velocity, fixed by requiring the profile to pass through
  (z_ref, U_inlet) (Richards & Hoxey 1993). A known difficulty — the *horizontal
  homogeneity* or "ABL-drift" problem — is that this profile tends to decay between
  the inlet and the buildings unless the inlet, the turbulence model, and the ground
  wall treatment are mutually consistent; sustaining it is a recognised challenge in
  urban CFD (and the source of the drift seen in the validation runs).

  A smooth inlet also carries *no turbulence*, whereas the real ABL is strongly
  turbulent, and turbulence injected only by the geometry develops too far
  downstream. So fluctuations are superimposed as a sum of random Fourier modes
  (synthetic-eddy / random-flow generation) with a prescribed integral length scale
  (`abl_Lturb`), number of modes (`abl_nmodes`), and anisotropic component
  intensities σ_u:σ_v:σ_w — giving a turbulent inflow from the first cell.
  (Smirnov, Shi & Celik 2001; Kraichnan 1970.) `inlet_profile = 0` gives a uniform
  plug for idealised tests.

- **Outlet.** A zero-gradient / convective outflow lets the flow (and its turbulent
  structures) leave without reflecting back into the domain.

- **Solid walls (GROUND, building faces).** No-slip is enforced by **bounce-back**: a
  population arriving at a wall is reflected back along its opposite velocity
  (f_i ← f_OPP[i]), which pins the velocity to zero approximately halfway between the
  fluid and solid nodes. Bounce-back is purely local and needs only a cell flag,
  which is what makes arbitrary urban geometry cheap. An optional rough-wall log-law
  floor (`wall_model = 1`) applies a halfway moving-wall slip tuned to sustain the
  ABL profile along the ground. Free-slip (zero-shear) faces, used for the domain top
  and sides when desired, use mirror reflections that preserve the tangential
  components instead of reversing them.

- **SHELL (permeable façade).** A partial bounce-back interpolates between solid and
  open by the local permeability β: f_d = (1−β) f_OPP + β f. β = 0 recovers a solid
  wall (wind goes cleanly around the building), while β > 0 lets a controlled
  fraction pass — used for porous elements such as vegetation or park envelopes.

- **Oblique wind.** A non-axis-aligned wind direction θ is imposed by treating the
  upwind faces as inlets and the downwind faces as outlets (a per-face inlet),
  leaving the geometry aligned to the grid. Rotating the *inflow* rather than the
  *geometry* avoids the staircasing errors that rotating a voxelised city would
  introduce.

## 7. The recovered macroscopic equations

A Chapman–Enskog analysis — a multiscale expansion of (1) in the Knudsen number
(the ratio of lattice spacing to flow scale) — shows that the moments of the lattice
Boltzmann equation satisfy, to second order, the weakly-compressible incompressible
Navier–Stokes system:

    ∂ρ/∂t + ∇·(ρu) = 0                                                        (8)
    ∂(ρu)/∂t + ∇·(ρ u u) = −∇p + ∇·[ ρ ν_eff ( ∇u + ∇uᵀ ) ] ,   p = c_s² ρ    (9)

The kinematic viscosity in (9) is exactly the ν_eff of the relaxation relation (5) —
this is the sense in which "faster relaxation = lower viscosity." Because pressure is
the isothermal p = c_s² ρ, the method is *artificially compressible*: it admits small
density fluctuations of order Mach², and it reproduces true incompressible flow only
in the low-Mach limit where those fluctuations are negligible. In short, the LBM (1)
is a solver for (8)–(9), and the time-average of its solution is the frozen wind
field the transport layer advects the contaminant on.

## 8. Solver phases and output

`run()` proceeds in three phases, reflecting that a turbulent flow is chaotic in
detail but statistically stationary in the mean:

- **A1 — spin-up.** Evolve the flow for `SPINUP_FT` domain *flow-throughs* (a
  flow-through is the time L/U for a parcel to traverse the domain), so the transient
  from the initial condition washes out and turbulence, seeded by the synthetic
  inflow, becomes fully developed throughout the domain.
- **A2 — averaging to stationarity.** The instantaneous field never settles, but its
  running time-mean does. The solver accumulates the mean velocity ū(x) and the mean
  eddy viscosity ν_t(x) and monitors the running average until its relative change
  falls below `avg_threshold`. The converged mean is the **frozen mean flow** — a
  RANS-like statistically-steady field obtained *from* a resolved (LES) run rather
  than from a RANS turbulence model.
- **B — scalar release.** A passive scalar is injected and transported (on a separate
  D3Q7 advection–diffusion lattice) for direct/validation dispersion. The production
  exposure path bypasses this: it freezes A2's mean and hands it to the linear adjoint
  solver, which is far cheaper and gives exact source–receptor sensitivities.

**Output.** The mean velocity ū(x) = (u_x, u_y, u_z) and eddy viscosity ν_t(x), plus
the geometry, at roughly 289 bytes/cell. Downstream, ν_t enters scalar transport as
the turbulent diffusivity D_t = ν_t / Sc_t — the gradient-diffusion closure detailed
in `ADJOINT_SOLVER.md`. That hand-off (ū, ν_t) is the sole coupling between the flow
solver and the rest of the pipeline.

## 9. Symbols

| symbol | meaning | units |
|---|---|---|
| f_i(x,t) | discrete particle population along lattice velocity e_i | lattice |
| e_i (≡ c_i) | i-th D3Q19 lattice velocity (i = 0…18; written c_i in HRR/Hermite context) | cells/step |
| w_i | equilibrium weight of direction i (1/3, 1/18, 1/36) | – |
| c_s² | lattice sound speed squared, = 1/3 | (cells/step)² |
| Ω_i | collision operator (relaxation toward equilibrium) | lattice |
| ρ | fluid density (≈ const; sets pressure via p = c_s²ρ) | lattice |
| u = (u_x,u_y,u_z) | macroscopic (resolved) velocity | cells/step |
| p | pressure, p = c_s² ρ | lattice |
| f_i^eq | discrete equilibrium distribution, eq. (4) | lattice |
| τ | relaxation time | steps |
| ω | relaxation rate 1/τ, eq. (5) | 1/step |
| ν_0 | molecular (lattice) kinematic viscosity | cells²/step |
| ν_t | WALE subgrid eddy viscosity, eq. (6) | cells²/step |
| ν_eff | ν_0 + ν_t (the viscosity in the recovered NS) | cells²/step |
| σ | HRR hybrid FD-stress blend weight (0–1) | – |
| g_ij | resolved velocity gradient ∂u_i/∂x_j | 1/step |
| S_ij | resolved strain-rate tensor ½(g_ij+g_ji) | 1/step |
| Sᵈ_ij | traceless symmetric part of g², eq. (6) | 1/step² |
| C_w | WALE model constant, ≈ 0.325 | – |
| Δ | LES filter width (= grid spacing, 1 in lattice) | cells |
| OPP[i] | index of −e_i (bounce-back partner) | – |
| β | SHELL permeability (0 solid … 1 open) | – |
| U(z) | ABL inlet mean speed at height z, eq. (7) | cells/step |
| u_* | friction velocity | cells/step |
| κ | von Kármán constant, 0.41 | – |
| z_0 | aerodynamic roughness length | m (→ cells) |
| z_ref | reference height where U = U_inlet | m (→ cells) |
| Sc_t | turbulent Schmidt number (scalar coupling) | – |
| L/U | domain flow-through time (spin-up unit) | steps |

Lattice units use Δx = 1 cell and Δt = 1 step; physical velocities convert by the
factor dt_phys/dx (`velocity_phys_to_lattice`), and physical viscosity/diffusivity
by dt_phys/dx².

## Sources

- Qian, Y.H., d'Humières, D. & Lallemand, P. (1992). Lattice BGK models for
  Navier–Stokes equation. *Europhys. Lett.* 17(6):479–484. (D3Q19 / LBGK.)
- Chen, S. & Doolen, G.D. (1998). Lattice Boltzmann method for fluid flows.
  *Annu. Rev. Fluid Mech.* 30:329–364. (Review; Chapman–Enskog, moments.)
- Krüger, T., Kusumaatmaja, H., Kuzmin, A., Shardt, O., Silva, G. & Viggen, E.M.
  (2017). *The Lattice Boltzmann Method: Principles and Practice.* Springer.
  (Comprehensive reference for §§1–7.)
- d'Humières, D. et al. (2002). Multiple-relaxation-time lattice Boltzmann models in
  three dimensions. *Phil. Trans. R. Soc. A* 360:437–451. (MRT.)
- Lallemand, P. & Luo, L.-S. (2000). Theory of the lattice Boltzmann method:
  dispersion, dissipation, isotropy, Galilean invariance, stability.
  *Phys. Rev. E* 61:6546. (Ghost modes and stability.)
- Latt, J. & Chopard, B. (2006). Lattice Boltzmann method with regularized
  pre-collision distribution functions. *Math. Comput. Simul.* 72:165–168.
  (Regularized collision.)
- Malaspinas, O. (2015). Increasing stability and accuracy of the lattice Boltzmann
  scheme: recursivity and regularization. *arXiv:1505.06900.*
- Jacob, J., Malaspinas, O. & Sagaut, P. (2018). A new hybrid recursive regularised
  Bhatnagar–Gross–Krook collision model for large-eddy simulation of turbulent flows.
  *J. Turbulence* 19(11):1051–1076. (HRR.)
- Nicoud, F. & Ducros, F. (1999). Subgrid-scale stress modelling based on the square
  of the velocity gradient tensor (WALE). *Flow Turbul. Combust.* 62:183–200.
- Hou, S., Sterling, J., Chen, S. & Doolen, G.D. (1996). A lattice Boltzmann subgrid
  model for high Reynolds number flows. *Fields Inst. Commun.* 6:151–166. (LES-LBM.)
- Richards, P.J. & Hoxey, R.P. (1993). Appropriate boundary conditions for
  computational wind engineering models using the k-ε turbulence model.
  *J. Wind Eng. Ind. Aerodyn.* 46–47:145–153. (ABL log-law inlet; homogeneity.)
- Smirnov, A., Shi, S. & Celik, I. (2001). Random flow generation technique for large
  eddy simulations and particle-dynamics modeling. *J. Fluids Eng.* 123:359–371.
  (Synthetic inflow turbulence.)
- Kraichnan, R.H. (1970). Diffusion by a random velocity field. *Phys. Fluids*
  13:22–31. (Random-Fourier-mode turbulence.)
- Tominaga, Y. et al. (2008). AIJ guidelines for practical applications of CFD to
  pedestrian wind environment around buildings. *J. Wind Eng. Ind. Aerodyn.*
  96:1749–1761. (Urban-CFD best practice.)
- Franke, J. et al. (2007). *Best Practice Guideline for the CFD Simulation of Flows
  in the Urban Environment.* COST Action 732.
