# Reverse (adjoint) solver and source-distribution field

## Purpose
Compute the effective-exposure objective from the **source side**, so that one
solve scores any release distribution, and the source can be anonymized. The
objective is linear in concentration:

    J = Σ_x w(x)·C(x)            (forward: release s, weight by receptor w)
      = Σ_x F(x)·s(x)            (reverse: footprint F from w, contract with s)

The two are identical by source–receptor reciprocity. The reverse form needs
only ONE adjoint solve (forced by the fixed receptor w) to produce the footprint
F(x) = effective exposure per unit release at x; thereafter J for any source
field s is a dot product.

## What the reverse solver is (and is not)
- It is the **discrete adjoint of the scalar transport** on the FROZEN mean flow.
- It is **not** an adjoint of the airflow. The contaminant is passive, so the
  airflow is an input; the flow solver and the production van Leer D3Q7 scalar are
  left completely intact. The reverse solver reads the time-averaged velocity and
  eddy viscosity (`Solver::copy_mean_flow_to_host`) and never re-runs the flow.

## Eliminating disagreement between the solvers (the nonlinearity plan)
A discrete adjoint is an exact transpose only of a **linear** forward operator.
The production forward uses the van Leer limiter, which is nonlinear; its discrete
adjoint loses consistency exactly where the limiter activates or upwinding
switches (Liu & Sandu 2008). We therefore adopted **Option 1 — a consistent
linear scheme**: the objective is evaluated with a matched pair of *first-order
linear upwind* forward `M` and adjoint `Mᵀ`. Because `M` is linear, `Mᵀ` is its
exact transpose and the forward and reverse objectives agree to machine precision
*by construction* — there is no nonlinearity to reconcile. The trade-off is more
numerical diffusion than van Leer; the van Leer D3Q7 scalar remains the tool for
physical-dispersion/validation runs, and the linear-upwind objective is to be
tested against physical standards (the agreed caveat).

## Numerics (adjoint_transport.h / adjoint_transport_gpu.cu)
- State is the **concentration field** C (not the D3Q7 distributions), advanced by
  an explicit finite-volume step `M = I + dt·(−∇·(uC) + ∇·(D∇C) − deposition)`:
  first-order upwind advection, central diffusion with `D = D0 + nut/Sc_t`, surface
  deposition, and gravitational settling (`w_eff = w − w_s` on the z-axis; the
  downward ground face also receives the settling flux).
- Boundaries (wind +x): x faces OPEN (advect out, zero inflow), y faces and z-top
  no-flux walls, z=0 ground (deposition), solid/building faces deposition.
- Both `M` (fwd_step) and `Mᵀ` (adj_step) are written as pure **gather** kernels —
  each cell writes only its own output — so the adjoint is race-free on GPU (no
  scatter, no atomics) and the CPU/GPU paths share identical per-cell math.
- The adjoint terms are the exact transpose of the forward gather coefficients,
  derived term-by-term:
  - identity → identity;
  - diffusion is self-adjoint → `kd·(P[j]−P[i])` (same form as forward);
  - advection is not self-adjoint → when the outward face velocity `vfn ≥ 0`,
    `Pn[i] += ka·vfn·(P[j] − P[i])` (gathers the adjoint from the downstream
    neighbour, i.e. reverse propagation);
  - deposition and open-boundary outflow are diagonal self-loss → transpose to the
    same self-loss.
- Time integration: TIAC = ∫C dt over the release window. The reverse footprint is
  the exact reverse-mode adjoint of that accumulation (`reverse_footprint`): each
  step `Cb += dt·w; sb += dt·Cb; Cb = Mᵀ·Cb`, giving `F = sb` with `J = Σ F·s`.

## Receptor and source fields (reverse_objective.h)
- **Receptor w(x)** — the §5 effective-exposure weighting, rasterized to a field at
  z=1: building `eff_inh` spread over facade-ring fluid cells, street weights on
  road cells, park `eff_inh` over park ground. Weights live on fluid cells (where
  the adjoint field is defined).
- **Source mask s(x)** — UNIFORM over open city spaces: z=1 fluid cells inside the
  city rectangle (roads + parks; buildings are solid → excluded; buffer fetch is
  outside the rectangle → excluded). Ground level only. Normalized to Σ=1 and
  scaled by the total emission Q.

## Verification
- `adjoint_recip_test.cpp` (prescribed sheared flow + solid blocks + ground):
  - **(A)** dot-product test `<M u, v> == <u, Mᵀ v>` to ~1e-7 = float epsilon for
    1, 8, and 40 composed steps → `Mᵀ` is the exact transpose (a transpose bug
    would give O(1) error).
  - **(B)** integrated objective identity: forward `Σ w·TIAC` == reverse `Σ F·s`
    to ~1.8e-7.
  - **(C)** physical sanity: a ground release advects downwind, mass is finite.
- `lab_test.cpp` §6 runs the reverse objective on the frozen LBM mean flow and
  asserts the same forward==reverse reciprocity as a standing regression
  (`reverse_objective.txt`).

## CPU / GPU mirror
- CPU: `adjoint_transport.h` (header-only), exercised and verified by the test.
- GPU: `adjoint_transport_gpu.cu` mirrors the gather kernels one-thread-per-cell
  with identical math; device drivers `reverse_footprint` / `forward_tiac` manage
  scratch and run the loops. **Untested on hardware** (no nvcc here); device-code
  syntax was checked with CUDA keywords neutralized. Rebuild on the target GPU and
  run the reciprocity check before production use.

## Known simplifications / next steps
- The wired demo uses one representative settling velocity (bin 0). The full
  mass-weighted footprint is `F = Σ_i frac_i·F_i` over PSD bins — loop the
  reverse solver per bin (each with its `w_s`/deposition), summed by activity
  fraction; this is the "propagation specified by density and size" path.
- Wall (building-face) deposition velocity is currently 0 (deposition via ground
  settling only); per-surface deposition velocities from the Zhang closure can be
  wired into the `vdep` field.
- Single wind direction; wind-rose aggregation remains future work.

## References
- Liu & Sandu (2008), *On the properties of discrete adjoints of numerical methods
  for the advection equation* — adjoint inconsistency localizes to limiter/upwind
  switches; motivates the linear-consistent choice.
- arXiv:2501.02161 — discrete vs continuous adjoint LBM; discrete adjoint gives
  consistent boundary conditions and exact sensitivities.
- SU2 `FROZEN_LIMITER_DISC`; arXiv:2410.05901 (freezing limiter nonlinearity);
  arXiv:1806.06117 (ICON artificial-source-term adjoint) — alternatives if van
  Leer must be retained in the objective forward.
- Marchuk; Pudykiewicz (1998); Keats et al. (2007) — adjoint source–receptor for
  atmospheric release/receptor problems.
