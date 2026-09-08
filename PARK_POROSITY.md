# Park porosity — model, provenance, and honest scope

## What the code does (verified)

Parks (Usage `PARK`) are voxelised as **SHELL** cells with a permeability
`PERM_PARK = 0.800` (`city_builder7.h`). In the collision kernel a SHELL cell
applies **partial ("grey") bounce-back**: the reflected fraction is

    σ = 1 − perm            (lbm_kernels_cpu.cpp)

so each population arriving at a park cell is split into a **reflected** part σ and
a **transmitted** part (1 − σ) = perm. With `perm = 0.8`, 80 % of the momentum is
transmitted and 20 % reflected — a weak, distributed resistance rather than a solid
wall. The same mechanism spans the built environment as a single permeability knob:

    PERM_BIZ = 0.005   (dense commercial, near-solid)
    PERM_RES = 0.010   (apartments)
    PERM_RES_LOW = 0.020   (wood-frame houses)
    PERM_PARK = 0.800   (tree canopy, very high porosity)
    PARK_CANOPY_H = 10 m   (canopy depth applied to park cells)

So a park is physically an obstacle that *slows* the wind in proportion to
(1 − perm), extracting momentum through the canopy depth — the correct qualitative
behaviour for vegetation.

## Provenance — the honest status of PERM_PARK = 0.8

**It is a physically-motivated value, not a calibrated one.** The code comment reads
"trees: very high porosity"; there is no link in the source to a leaf-area density
(LAD), a drag coefficient C_d, a measured canopy pressure drop, or a velocity-
attenuation target. `perm = 0.8` was chosen as a reasonable "mostly-open" fraction
consistent with the ordering (park ≫ house > apartment > commercial), **not** derived
from or calibrated against a vegetation measurement. This must be stated plainly:
the value is a defensible order-of-magnitude choice, and the transmitted fraction is
not yet mapped to a physical canopy property.

## Verification method, and the verification-vs-validation distinction

**What was done (verification of provenance), and how.** The claims above were
established by reading the source, not by running a physics comparison:

1. Located the parameter — `grep PERM_PARK city_builder7.h` returned
   `constexpr float PERM_PARK = 0.800f; // trees: very high porosity`, alongside the
   full permeability ladder (BIZ 0.005, RES 0.010, RES_LOW 0.020, PARK 0.800) and
   `PARK_CANOPY_H = 10 m`.
2. Traced its use into the kernel — in `lbm_kernels_cpu.cpp` the SHELL write block
   computes `float sigma = 1.f - pm[id];` and applies partial bounce-back with that
   reflected fraction, confirming the σ = 1 − perm mechanism and that park cells
   (PARK → SHELL, perm 0.8) transmit 80 % of the momentum.
3. Searched for any physical linkage — grepping for LAD, leaf-area, drag, C_d,
   attenuation, or calibration returned **only the PERM_PARK definition line
   itself**, confirming the value is standalone, not derived from a canopy property.

This is **verification**, in the formal CFD sense: it confirms the implementation
matches its documented intent (the code really does apply the stated partial-
bounce-back closure with the stated value). It is a source-level check, cheap and
exact, and it is the correct basis for the provenance statements above.

**What was *not* done (validation).** No **validation** of the park model has been
performed. Validation would ask whether the resulting park flow matches physical
reality — comparing the modelled velocity attenuation or pressure drop through the
canopy against wind-tunnel or field data for a vegetation stand of known leaf-area
density (e.g. the Gromke & Ruck canopy cases). That comparison has not been run, and
`perm = 0.8` has not been calibrated to any such measurement.

**Why the distinction matters for the writeup.** Verification ("did we implement the
intended model correctly") and validation ("is the model physically right") are not
interchangeable. It is accurate to state that the park porosity model is *verified*
(the code does what it claims) but *not validated* (its output has not been checked
against canopy data). Claiming the park model is "validated" on the basis of the
source check would be an overclaim; the honest status is **verified implementation,
uncalibrated and physically unvalidated parameter** — which is why the calibration
step in the previous section is the prerequisite for any validation claim.

## Precedent — cite by component, not as one standard

There is no canonical paper that specifies "porous bounce-back for parks at
permeability β." The model is a sound *combination* of two separately-established
pieces, and the writeup should cite them as such (the same component-not-whole
framing used for the inlet and subgrid model):

- **Vegetation as a porous momentum sink (the physics).** The accepted CFD
  representation of trees/hedges is a canopy drag sink, F = −ρ C_d a |u| u, with a
  the leaf-area density: Green (1992); Liu et al. (1996); Sanz (2003) for the k–ε
  canopy closure; and the Gromke & Ruck wind-tunnel/CFD series for trees in urban
  canyons. These justify treating a park as a distributed momentum-absorbing medium.
- **Partial (grey) bounce-back (the numerics).** Representing intermediate
  permeability by reflecting a fraction of the populations is a recognised
  lattice-Boltzmann technique: Dardis & McCloskey (1998) (the original partial-
  bounce-back scheme for porous media); Walsh, Burwinkle & Saar (2009) (partially-
  solid cells / variable permeability). These justify the σ = 1 − perm closure.

What is **not** citable as standard is the specific pairing (grey bounce-back used
for a park canopy). Most urban-vegetation CFD uses the explicit drag-sink term
rather than partial bounce-back, precisely because the drag term ties resistance to
measured LAD and C_d, whereas a bounce-back fraction is a more abstract parameter.

## The gap, closed — calibration performed

The permeability has now been calibrated against canopy drag. Method, result, and
verification below; the calibration harness is `canopy_calib.cpp` (D1Q3 channel
using the exact SHELL closure) plus the analytical momentum law.

### Step 1 — exact drag law (verified to machine precision)

Applying the real closure `fd[i] = σ·fp[OPP[i]] + (1−σ)·fp[i]` (σ = 1−perm) to a
population, the post-closure momentum is

    ρu_out = (1 − 2σ) ρu_in = (2·perm − 1) ρu_in                              (★)

because Σ_i c_i fp[OPP[i]] = −ρu (bounce-back reverses momentum). Verified
numerically on the D3Q19 lattice to ~1e-16 (perm=0→−1 reversal, 0.5→0 full stop,
0.8→+0.6, 1→+1 no obstacle; mass conserved). So the closure removes a **fixed
fraction 2σ of the momentum per cell per step, independent of |u|** — a *linear*
(Darcy) drag, whereas a vegetation canopy is *quadratic*, F = ρ C_d a |u| u. This
functional mismatch is the central caveat (see Step 4).

### Step 2 — target from canopy-drag literature

The canopy sink is F = ρ C_d a u², giving a 1-D pressure-loss coefficient over the
canopy depth of **Δp/(ρu²) = C_d · a · L**. With representative urban-tree values

- C_d ≈ 0.2  (sectional drag coefficient of vegetation; Wilson & Shaw 1977;
  Katul et al. 2004 — range 0.15–0.3),
- a ≈ 1.0 m⁻¹ (leaf-area density of an in-leaf urban tree crown; Gromke & Ruck
  2007/2012 — range ~0.5–2),
- L = 10 m (PARK_CANOPY_H),

the target is **C_d·a·L ≈ 2.0**.

### Step 3 — calibration result

The D1Q3 channel gives a loss coefficient (channel-confirmed and analytically
matched) of Δp/(ρu²) = 2σ·N_cells / u_lb, with N_cells = L/dx the canopy depth in
cells. Setting this to C_d·a·L and simplifying (N_cells = L/dx) gives a
resolution-explicit calibration:

    σ = ½ · C_d · a · dx · u_lb ,     perm = 1 − ½ · C_d · a · dx · u_lb       (§)

where dx is the cell size (m) and u_lb the lattice canopy-level velocity. Confirmed
against the channel (formula perm=0.996 at dx=2 m reproduces the loss=2.0 crossing).
Representative calibrated values (C_d=0.2, a=1.0, u_lb≈0.05):

    dx = 4.0 m :  perm ≈ 0.980
    dx = 2.0 m :  perm ≈ 0.990
    dx = 0.5 m :  perm ≈ 0.9975

**The current PERM_PARK = 0.800 is ~20× too resistive** — at these depths it yields
a loss coefficient of 35–44 versus the target 2.0, i.e. it behaves almost like a
solid wall, not an open canopy. The calibrated per-cell reflected fraction is
σ ≈ 0.002–0.02 (0.2–2 %), not 0.20.

### Step 4 — honest caveats on the calibrated value

- **Velocity-specific.** Because the closure is linear in u and the canopy is
  quadratic, the two can be matched at only one speed. Verified directly: the loss
  coefficient halved as u doubled (1.97 → 0.98 → 0.49 for u_lb = 0.01/0.02/0.04),
  whereas a true canopy's would be u-independent. So (§) holds at the chosen
  reference u_lb; below it the model over-drags, above it under-drags. Calibrate at
  the pedestrian/canopy-level speed where exposure matters most.
- **Resolution-dependent.** perm must scale with dx (via §); a single constant is
  wrong across grids — the drag compounds per cell, so a fixed perm gives 4× the
  loss at dx=0.5 m that it does at dx=2 m.
- **Still isotropic, single-layer, momentum-only** (as in the scope section).

### Verification vs validation (of this calibration)

The calibration is **verified**: the momentum law (★) matches the real closure to
machine precision, the channel reproduces formula (§), and the limits/monotonicity/
velocity-scaling all behave as derived (checks logged in the calibration run). It is
**not yet validated** against measured canopy data — C_d and a are literature
central values, not fitted to a specific park, and no comparison to a wind-tunnel
velocity-attenuation profile (e.g. a Gromke & Ruck CODASC case) has been run.
Validation would fit (C_d·a) to such a profile and confirm (§) reproduces it.

## Scope caveats to keep with the model

- **Isotropic resistance.** σ = 1 − perm reflects equally in all directions, so the
  park resists cross-flow and vertical flow the same as streamwise flow. A real
  canopy is anisotropic (mostly vertical LAD gradient); an explicit drag term can be
  directionally weighted, partial bounce-back as implemented cannot.
- **Single value, no vertical LAD profile.** Real canopies are sparse at the trunk,
  dense in the crown; here a park is a uniform-perm block of depth PARK_CANOPY_H.
- **Momentum only.** The park attenuates wind; it does not add the turbulence
  production/dissipation a canopy k–ε closure (Sanz 2003) would, nor deposition to
  foliage — both are second-order for the dispersion mean but worth noting.

## Sources

- Wilson, N.R. & Shaw, R.H. (1977). A higher order closure model for canopy flow.
  *J. Applied Meteorology* 16:1197–1205. (Sectional drag coefficient C_d ≈ 0.2.)
- Katul, G.G., Mahrt, L., Poggi, D. & Sanz, C. (2004). One- and two-equation models
  for canopy turbulence. *Boundary-Layer Meteorology* 113:81–109. (Canopy drag C_d.)
- Dardis, O. & McCloskey, J. (1998). Lattice Boltzmann scheme with real numbered
  solid density for the simulation of flow in porous media. *Phys. Rev. E* 57:4834.
- Walsh, S.D.C., Burwinkle, H. & Saar, M.O. (2009). A new partial-bounceback
  lattice-Boltzmann method for fluid flow through heterogeneous media.
  *Computers & Geosciences* 35:1186–1193.
- Sanz, C. (2003). A note on k–ε modelling of vegetation canopy air-flows.
  *Boundary-Layer Meteorology* 108:191–197.
- Green, S.R. (1992). Modelling turbulent air flow in a stand of widely-spaced
  trees. *PHOENICS Journal* 5:294–312.
- Liu, J., Chen, J.M., Black, T.A. & Novak, M.D. (1996). E–ε modelling of turbulent
  air flow downwind of a model forest edge. *Boundary-Layer Meteorology* 77:21–44.
- Gromke, C. & Ruck, B. (2007, and subsequent). Influence of trees on the dispersion
  of pollutants in an urban street canyon. *Atmospheric Environment* / *Boundary-
  Layer Meteorology* series. (Wind-tunnel + CFD canopy-in-canyon reference set.)
