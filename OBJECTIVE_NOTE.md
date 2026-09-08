# Objective: concentration × effective inhabitance

## Definition
The optimization objective is the **effective-inhabitance-weighted concentration**:

    J = Σ_locations  C(location) · inh_eff(location)

the inner product of the dispersed concentration field and the effective
inhabitance map.

- **C** — the mass-weighted time-integrated air concentration (TIAC = ∫C dt),
  summed over the polydisperse size bins. It is the concentration the population
  is exposed to at each location.
- **inh_eff** (`eff_inh`) — the effective inhabitance: the building/park
  population already weighted by the NHAPS occupancy time budget (it is *where*
  and *how much* people are present, daily-averaged; no time-of-release
  assumption). Street occupants carry their own per-cell weight.

J is a raw exposure functional with units of concentration·person·time. There are
**no exposure weights, no infiltration factor F_inf, and no deposited-surface exposure term** —
it is purely the concentration·inhabitance product. The optimizer minimizes J.

## Where each population parcel samples the concentration
- **Building occupants** → facade-adjacent outdoor concentration (one-cell ring
  around the envelope), via `indoor_exposure(...).total_outdoor_exposure`
  (i.e. Σ eff_inh · C_facade, with F_inf NOT applied).
- **Street occupants** → local cell concentration, via `street_exposure(..., 1.0)`
  (Σ street_weight · C).
- **Park occupants** → footprint-average concentration × the park block eff_inh.

Summed, these are exactly Σ C · inh_eff over all effective inhabitance.

## Implementation
- `lab_test.cpp` §5 — replaces the former multi-pathway exposure (which used pathway
  weights W_AIRBORNE/W_DEPOSITED, F_inf, and a deposited-surface term) with the
  three concentration·eff_inh sums above; writes `objective` to `total_exposure.txt`.
- The concentration field itself (`dispersion_z1.bin`, mass-weighted TIAC) and
  deposition field are unchanged.

## What is NOT changed here
- The **source** is untouched — still the single point source at `srcIdx` from
  `source_x/source_z`. (Source treatment is a separate, later change.)
- `infiltration.h` (F_inf) and exposure weights remain in the tree for a future
  exposure-calibrated variant; they are simply not part of this objective.

## Caveats
- Dropping F_inf means building occupants are scored on outdoor facade
  concentration, which overstates indoor intake relative to an infiltration
  model — intentional for this raw concentration·inhabitance objective.
- Verified: the objective compiles and runs on the CPU backend; it is nonzero and
  well-formed at realistic city sizes (e.g. a 512 m city: population ~5200,
  inhabitance ~4700). It is trivially zero only for degenerate sub-threshold test
  cities that contain no inhabited buildings.
