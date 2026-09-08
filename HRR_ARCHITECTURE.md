# HRR collision — construction and application

The production high-Reynolds collision operator: **Hybrid Recursive Regularized**
(Jacob, Malaspinas & Sagaut, *J. Turbulence* 19:1051, 2018). This document explains
exactly how the operator is built and applied, matching `rr_collide` in
`lbm_kernels_cpu.cpp` (and its device mirror in `lbm_kernels.cu`).

## Status

Complete in both backends. The RR core and the σ-blended FD hybrid are both
implemented and unit-tested on CPU: equilibrium is an exact fixed point, mass and
momentum are conserved to round-off, and the recovered shear viscosity is exact at
σ=1 and σ=0.98 (stress-recovery test). Wired through `lbm_solver.cpp` and `Config`
as `COLLISION=hrr` (collision_mode=2) with `HRR_SIGMA`. GPU-verified on the A4000
(banner σ=0.980, ran the full lab-verify battery without divergence).

## The key idea in one sentence

**HRR does not change τ.** τ (equivalently ω = 1/τ) is fixed by the viscosity
relation ω = 1/(3ν_eff + ½) — identical for BGK, MRT, regularized, and HRR. What HRR
changes is the *operand* of the relaxation: instead of relaxing the raw
non-equilibrium populations, it relaxes a **reconstructed** non-equilibrium field
that (a) has the destabilising "ghost" content removed and (b) carries a small,
σ-controlled hyperviscosity in its third-order part only. The final relaxation step
is ordinary BGK at that same ω.

So every operator in the family has the shape

    f_i  =  f_i^eq  +  (1 − ω) · f_i^neq

and they differ **only** in how f_i^neq is built. HRR builds it as described below.

## Why (the Phase-0 decision)

Phase 0 established three independent ways that the regularized (projected-MRT)
operator cannot run the production flow: a steady stability wall at ν = 5e-3
(τ = 0.515, Re ≈ 365 — still laminar); turbulent (synthetic-inflow) breakdown at
5e-3; and divergence at 5e-3 on the fine grid (the wall tightens with resolution).
reg cannot reach a viscosity low enough to *be* turbulent. HRR is the literature-
standard fix: it injects a **tunable hyperviscosity** that stabilises exactly this
high-Re/low-ν regime while leaving the recovered Navier–Stokes viscosity exact.

## The construction, step by step

Inputs to one cell's collision: the streamed populations f_i; the macroscopic ρ and
u = (u,v,w) from the moments (K1); the relaxation rate ω from the viscosity (WALE
included); the blend σ; and the resolved strain-rate tensor S_ab (central-difference
velocity gradients, reused from the WALE step).

**Step 1 — equilibrium and raw non-equilibrium.** Build the 2nd-order equilibrium
and subtract it:

    f_i^eq  = w_i ρ ( 1 + 3 c_i·u + 4.5 (c_i·u)² − 1.5 u² )
    f_i^neq = f_i − f_i^eq

The raw f_i^neq is *not* relaxed directly — it contains ghost (non-hydrodynamic)
content that destabilises the τ→½ limit. It is only used to measure the stress next.

**Step 2 — measure the projected second-order coefficient (the stress).** The
non-equilibrium second-order Hermite coefficient is the non-equilibrium momentum flux
Π_neq:

    a²_ab = Σ_i c_ia c_ib f_i^neq                 (a,b ∈ {x,y,z})

(The −c_s²δ_ab term drops because Σ_i f_i^neq = 0.) This **projected** a² is what
sets the physical viscosity, so it is used *unmodified* for the second-order part of
the reconstruction — that is why the recovered Navier–Stokes shear viscosity is
exact regardless of σ.

**Step 3 — the hybrid blend (only if σ < 1).** Form a second stress estimate from
finite differences via the leading-order Chapman–Enskog stress–strain relation,

    a²_FD = −(2ρ / 3ω) · S        (= −(2ρc_s²/ω) S, with c_s² = 1/3)

and blend it with the projected stress:

    b_ab = σ · a²_ab + (1 − σ) · a²_FD,ab

b is the **blended** stress. Crucially, b feeds *only* the third-order term (Step 4);
the second-order term keeps the pure projected a². At smooth flow the two stress
estimates agree and b ≈ a²; at sharp gradients (the aliasing-prone scales) they
differ, and that mismatch is what injects the hyperviscosity — concentrated at high
wavenumbers, and routed exclusively through the third-order channel so it never
touches the second-order viscosity. σ = 1 disables it (pure RR).

**Step 4 — reconstruct the third-order coefficient recursively.** Rather than
*measuring* a³ from f_i^neq (which would reintroduce ghost content), it is rebuilt
from the (blended) stress and the velocity by the Malaspinas recursion,

    a³_abg = u_a b_bg + u_b b_ag + u_g b_ab

On D3Q19 only the six "iij" components are admissible (Step 6), and they evaluate to

    a³_xxy = 2u b_xy + v b_xx      a³_xxz = 2u b_xz + w b_xx
    a³_yyx = 2v b_xy + u b_yy      a³_yyz = 2v b_yz + w b_yy
    a³_zzx = 2w b_xz + u b_zz      a³_zzy = 2w b_yz + v b_zz

**Step 5 — assemble the regularized non-equilibrium populations.** Project the
Hermite coefficients back onto populations using the 2nd- and 3rd-order Hermite
polynomials H² , H³ evaluated at each c_i:

    f_i^neq,reg = w_i ( 4.5 · H²(c_i):a²  +  13.5 · H³(c_i):a³ )

with, explicitly,

    H²(c_i):a² = Σ_ab a²_ab ( c_ia c_ib − c_s² δ_ab )
    H³(c_i):a³ = Σ_iij a³ ( c_i-components of the iij Hermite polynomial )

The coefficients are the inverse Hermite norms: 4.5 = 1/(2 c_s⁴) at second order and
13.5 = 1/(2 c_s⁶) at third (the ×3 multiplicity of each iij multiset folded in).
Note the split: **second order uses the projected a² (exact viscosity); third order
uses the blended a³ (the σ-hyperviscosity).**

**Step 6 — relax (ordinary BGK at the unchanged ω).**

    f_i = f_i^eq + (1 − ω) f_i^neq,reg

This is the only place ω enters, and it enters exactly as in plain BGK. HRR's work is
all in *what* f_i^neq,reg is; the relaxation itself is untouched.

## Why each piece is there

- **Reconstruct rather than relax raw f^neq** — the raw non-equilibrium carries
  higher moments Navier–Stokes does not constrain (ghosts); relaxing them at ω is
  what makes BGK blow up near τ = ½. Rebuilding f^neq from only the physical
  coefficients removes them.
- **Recursive a³ (not measured)** — measuring a³ from f^neq would put the ghost
  content straight back; the recursion regenerates only the physically-implied
  third-order part from a² and u (Malaspinas 2015).
- **Hybrid blend in the third-order term only** — this is the crux of the design.
  Putting the FD/projected stress mismatch solely in the third-order channel adds
  dissipation at the small scales that alias and destabilise, while the second-order
  moment — and therefore the resolved shear viscosity ν_eff — is left exactly as τ
  prescribes. σ tunes stability; τ sets physics; they are orthogonal.

## What is exact, what is tuned

- **Exact (independent of σ):** mass and momentum conservation; the equilibrium as a
  fixed point; the recovered Navier–Stokes shear viscosity (second-order moment uses
  the unmodified projected a²). Verified numerically.
- **Tuned (via σ ∈ (0,1]):** the small-scale hyperviscosity that buys stability at
  low ν. σ = 1 → pure recursive-regularized, no added dissipation; σ slightly below 1
  (production: 0.98) → enough high-wavenumber damping to hold the τ→½ limit without
  visibly altering the resolved flow.

## D3Q19 admissibility (why the subset is automatic)

Only the six "iij" third-order Hermite polynomials contribute on D3Q19; the others
vanish identically on this velocity set, so there is no hand-picked subset to get
wrong. The fully-diagonal H³_iii vanishes because c³ = c on the lattice and
c_s² = 1/3 make H³_iii(c) = c_i(c_i² − 3c_s²) = c_i(1 − 1) = 0; the fully-mixed
H³_xyz vanishes because no D3Q19 velocity has all three components nonzero. Hence
the reconstruction (Step 4) needs only the six iij terms and is exact for this
lattice by construction.

## Symbols and coefficients

| symbol | meaning |
|---|---|
| f_i, f_i^eq, f_i^neq | population, equilibrium, non-equilibrium (raw) |
| f_i^neq,reg | reconstructed (regularized + hybrid) non-equilibrium — the relaxed operand |
| c_i | D3Q19 lattice velocity (≡ e_i in LBM_SOLVER.md; CX,CY,CZ in code) |
| w_i | lattice weight (W19) |
| c_s² = 1/3 | lattice sound speed squared |
| ω = 1/τ | relaxation rate, from ν_eff (unchanged by HRR) |
| a²_ab | projected 2nd-order non-eq. Hermite coeff. = Π_neq (stress) |
| a²_FD | finite-difference stress, −(2ρ/3ω) S |
| S_ab | resolved strain-rate tensor (central-difference gradients) |
| b_ab | blended stress σ a² + (1−σ) a²_FD (3rd-order input only) |
| a³_abg | recursive 3rd-order coeff., u_a b_bg + u_b b_ag + u_g b_ab |
| σ | hybrid blend / hyperviscosity knob (1 = pure RR; 0.98 = production) |
| 4.5 = 1/(2c_s⁴) | 2nd-order Hermite reconstruction weight |
| 13.5 = 1/(2c_s⁶) | 3rd-order Hermite reconstruction weight (iij multiplicity folded in) |

## Sources

- Jacob, J., Malaspinas, O. & Sagaut, P. (2018). A new hybrid recursive regularised
  BGK collision model for large-eddy simulation of turbulent flows. *J. Turbulence*
  19(11):1051–1076. (HRR; the σ hybrid.)
- Malaspinas, O. (2015). Increasing stability and accuracy of the lattice Boltzmann
  scheme: recursivity and regularization. *arXiv:1505.06900.* (Recursive a³.)
- Latt, J. & Chopard, B. (2006). Lattice Boltzmann method with regularized
  pre-collision distribution functions. *Math. Comput. Simul.* 72:165–168.
  (The regularization the RR core builds on.)
- Coreixas, C., Wissocq, G., Puigt, G., Boussuge, J.-F. & Sagaut, P. (2017).
  Recursive regularization step for high-order lattice Boltzmann methods.
  *Phys. Rev. E* 96:033306. (Recursive-regularized formulation and stability.)
