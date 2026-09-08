# The WALE subgrid turbulence model

The large-eddy-simulation (LES) closure used by the flow solver: the **Wall-Adapting
Local Eddy-viscosity** model of Nicoud & Ducros (1999). It supplies the subgrid eddy
viscosity ν_t that is added to the molecular viscosity (ν_eff = ν_0 + ν_t) and fed to
the collision relaxation rate. Implemented in `wale_nut` (`lbm_kernels_cpu.cpp` and
its device mirror).

## What it is for

LES resolves the large, energy-containing turbulent eddies on the grid and *models*
the effect of the smaller, unresolved ("subgrid") ones. That effect is represented as
an extra dissipation — an eddy viscosity ν_t(x) that drains energy from the resolved
scales as the real subgrid cascade would. WALE is the specific recipe for computing
ν_t from the resolved velocity field.

## Governing equation

WALE builds ν_t from the resolved velocity-gradient tensor g_ij = ∂u_i/∂x_j (central
differences on the lattice):

    ν_t = (C_w Δ)² · (Sᵈ_ij Sᵈ_ij)^{3/2} / [ (S_ij S_ij)^{5/2} + (Sᵈ_ij Sᵈ_ij)^{5/4} ]   (1)

with the two tensors

    S_ij  = ½( g_ij + g_ji )                                   resolved strain rate      (2)
    Sᵈ_ij = ½( g²_ij + g²_ji ) − ⅓ δ_ij g²_kk ,   g²_ij = g_ik g_kj                       (3)

Sᵈ is the **traceless symmetric part of the square of the velocity-gradient tensor**.
The contractions Sᵈ:Sᵈ = Sᵈ_ij Sᵈ_ij and S:S = S_ij S_ij are scalars (the double-dot
product A:B = Σ_ij A_ij B_ij — a full contraction over both indices, i.e. the sum of
elementwise products). C_w ≈ 0.325 is the model constant and Δ is the
filter width (the grid spacing; Δ = 1 in lattice units, folded into C_w). When the
denominator vanishes (uniform flow), ν_t is set to 0.

## Why WALE rather than Smagorinsky

The classic Smagorinsky model sets ν_t = (C_s Δ)² (2 S_ij S_ij)^{1/2} ∝ |S| — it keys
the eddy viscosity to the strain-rate magnitude alone. Two consequences make it a poor
fit for building flows, both of which WALE fixes:

1. **Spurious near-wall / laminar-shear viscosity.** |S| is nonzero in *any* shear,
   including the smooth laminar shear next to a wall or in a laminar boundary layer, so
   Smagorinsky produces subgrid viscosity where there is no turbulence, and it needs
   ad-hoc van-Driest damping functions to suppress it near walls. WALE's operator (3)
   is built so that ν_t **vanishes in pure shear** and, critically, **decays as y³**
   approaching a wall — the correct asymptotic behaviour — with no damping functions.
2. **Sensitivity to strain *and* rotation.** Because Sᵈ derives from g² = (∇u)·(∇u), it
   responds to both the strain-rate and rotation-rate tensors, so it detects the
   genuinely turbulent regions (where both are present) and stays quiet in irrotational
   straining. This is exactly the discrimination needed around separating/reattaching
   building wakes, where getting the turbulent-region location right sets the
   dispersion pattern.

So WALE is chosen because the physics that matters here — near-wall boundary layers and
separated shear layers on bluff bodies — is precisely where Smagorinsky misbehaves.

## How it couples into the solver

ν_t from (1) is added to the molecular viscosity, and the sum sets the local collision
relaxation:

    ν_eff = ν_0 + ν_t ,      ω = 1 / (3 ν_eff + ½)         (lattice; c_s² = 1/3)

so a more turbulent cell (larger ν_t) relaxes faster toward equilibrium — more subgrid
dissipation, exactly as intended. A floor ν_eff ≥ 10⁻³ (τ ≥ 0.503) guards the τ→½ stability limit.
The **same** ν_t is then handed to the scalar transport as the turbulent diffusivity via
the gradient-diffusion hypothesis, D_t = ν_t / Sc_t (Sc_t ≈ 0.7), which is the coupling
detailed in `ADJOINT_SOLVER.md`. And the same gradient tensor g_ij computed here is
reused by the HRR collision's finite-difference stress, so it is evaluated once per cell.

## Implementation notes

`wale_nut` takes the nine velocity derivatives, forms the nine components of g²_ij
(g2ij), subtracts the trace to build the six independent components of Sᵈ, and evaluates

    A = Sᵈ:Sᵈ = Sᵈ_00² + Sᵈ_11² + Sᵈ_22² + 2(Sᵈ_01² + Sᵈ_02² + Sᵈ_12²)

then ν_t = C_w² · A^{3/2} / [ (S:S)^{5/2} + A^{5/4} ], returning 0 if the denominator
underflows. Because S and Sᵈ are symmetric, only six components of each are stored.

## Symbols

| symbol | meaning | units |
|---|---|---|
| ν_t | subgrid eddy viscosity (the WALE output) | cells²/step |
| ν_0 | molecular kinematic viscosity | cells²/step |
| ν_eff | ν_0 + ν_t | cells²/step |
| g_ij = ∂u_i/∂x_j | resolved velocity-gradient tensor | 1/step |
| g²_ij = g_ik g_kj | matrix square of g | 1/step² |
| S_ij | resolved strain-rate tensor, eq. (2) | 1/step |
| Sᵈ_ij | traceless symmetric part of g², eq. (3) | 1/step² |
| S:S, Sᵈ:Sᵈ | double contractions (scalars) | (1/step)², (1/step²)² |
| C_w | WALE constant, ≈ 0.325 | – |
| Δ | LES filter width (grid spacing; 1 in lattice) | cells |
| Sc_t | turbulent Schmidt number (scalar coupling) | – |

## Sources

- Nicoud, F. & Ducros, F. (1999). Subgrid-scale stress modelling based on the square of
  the velocity gradient tensor. *Flow, Turbulence and Combustion* 62(3):183–200.
  (The WALE model; eqs. 1–3, C_w, the y³ near-wall scaling.)
- Smagorinsky, J. (1963). General circulation experiments with the primitive equations:
  I. The basic experiment. *Mon. Weather Rev.* 91(3):99–164. (The baseline eddy-viscosity
  model WALE improves on.)
- Germano, M., Piomelli, U., Moin, P. & Cabot, W.H. (1991). A dynamic subgrid-scale eddy
  viscosity model. *Phys. Fluids A* 3(7):1760–1765. (Context: dynamic determination of
  the constant; an alternative to a fixed C_w.)
- Hou, S., Sterling, J., Chen, S. & Doolen, G.D. (1996). A lattice Boltzmann subgrid
  model for high Reynolds number flows. *Fields Inst. Commun.* 6:151–166. (LES within the
  lattice-Boltzmann framework — ν_t entering the relaxation time.)
- Pope, S.B. (2000). *Turbulent Flows.* Cambridge University Press, ch. 13. (LES and
  eddy-viscosity closures, general reference.)
