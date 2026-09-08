# Building infiltration & indoor exposure — linear architecture

Replaces the (reverted) diffusion-through-wall model. Indoor exposure is a
**linear reduction of the outdoor concentration** via the infiltration factor
F_inf, applied at the receptor. No indoor transport is resolved; buildings remain
outdoor-flow obstacles. This preserves the exact linear adjoint and the
Ω-reduction, and it makes block geometry sufficient by construction.

## 1. The model (followed exactly)

**Primary reference — Riley, McKone, Lai & Nazaroff (2002), *Environ. Sci.
Technol.* 36(2):200–207**, "Indoor particulate matter of outdoor origin:
importance of size-dependent removal mechanisms." Their single-compartment
(well-mixed), no-indoor-source mass balance, per particle-size bin of diameter d:

    dC_in/dt = P(d)·a·C_out − ( a + k(d) )·C_in

At steady state (dC_in/dt = 0) this gives the infiltration factor exactly:

    C_in(d) / C_out(d)  =  F_inf(d)  =  P(d)·a / ( a + k(d) )                (1)

with
- **C_out(d)** outdoor concentration of size bin d (from our adjoint/CFD field),
- **C_in(d)** indoor (well-mixed) concentration,
- **a** air-exchange rate [1/time] (ventilation + crack infiltration),
- **P(d)** penetration factor [–], fraction surviving envelope passage,
- **k(d)** indoor deposition loss-rate coefficient [1/time].

Equation (1) is used verbatim; nothing is added to it. It is a **constant linear
multiplier** on the outdoor concentration, per building, per size bin.

## 2. Parameter sources (each sourced separately)

- **P(d)** — penetration factor: **Liu & Nazaroff (2001), *Atmos. Environ.*
  35(26):4451–4462**, "Modeling pollutant penetration across building envelopes"
  (crack-flow penetration physics). P ≈ 0.7–1.0 for 0.1–1 µm; falls for coarse
  (>2.5 µm, inertial/gravitational loss in cracks) and ultrafine (<0.1 µm,
  diffusional loss).
- **k(d)** — indoor deposition loss rate: **Lai & Nazaroff (2000), *J. Aerosol
  Sci.* 31(4):463–476**, "Modeling indoor particle deposition from turbulent flow
  onto smooth surfaces." Size-dependent, minimum (~0.1 1/h) near 0.1–0.5 µm,
  rising for coarse (settling) and ultrafine (diffusion).
- **a** — air-exchange rate: building-/climate-specific; residential central
  value ≈ 0.5 1/h (range ~0.1–2). Population distributions in **Allen et al.
  (2012), *Environ. Health Perspect.* (MESA Air)** and ASHRAE.
- **F_inf population anchor** — **Allen et al. (2012), MESA Air**: measured PM2.5
  F_inf = 0.62 ± 0.21 (community means 0.47–0.82). Used as the uniform fallback
  when building-specific P, a, k are unavailable.
- **Review / cross-check** — **Chen & Zhao (2011), *Atmos. Environ.*
  45:275–288**, I/O ratio, F_inf, and P.

## 3. Coupling to the adjoint exposure objective

A person's inhaled concentration depends on microenvironment (outdoors, or indoors
in building b): outdoors they breathe C_out(x); indoors they breathe
C_in,b(d) = F_inf,b(d)·C_out,b(d), where C_out,b is the outdoor concentration the
building intakes. Time-activity (fraction indoors f_in, outdoors f_out) comes from
**Klepeis et al. (2001), NHAPS, *J. Expo. Anal. Environ. Epidemiol.* 11:231–252**
(national means: f_in ≈ 0.87, f_out ≈ 0.075, in-transit ≈ 0.055).

The steady population **intake rate** (our objective J — an intake-fraction-type
quantity; Bennett et al. 2002) is then linear in the outdoor field, per size bin:

    J = Σ_d ω_d · ⟨ w_d , C_out,d ⟩                                          (2)

with the **effective receptor weight** applied to the OUTDOOR field at outdoor cells:

    w_d(x) = B · [ ρ_out(x)·f_out  +  Σ_b 1{x ∈ envelope(b)} · (N_b/|env(b)|)·f_in·F_inf,b(d) ]   (3)

where B = breathing rate, ρ_out(x) = outdoor-present population density, N_b =
occupancy of building b, env(b) = the outdoor fluid cells adjacent to building b's
envelope, |env(b)| its cell count, and ω_d the emitted mass fraction in bin d.

Because F_inf,b(d), f_in, f_out are constants, w_d is a fixed field and J is linear
in the source ⇒ the adjoint φ_ss = ∂J/∂s is exact and the Ω-reduction
J_ensemble = (1/|Ω|)Σ_{x∈Ω} φ_ss(x) is valid, unchanged. **No indoor cells, no
through-wall transport, no permeable façade.** Indoor occupants' weight sits on
the outdoor envelope cells, scaled by F_inf.

## 4. Why block geometry suffices

The outdoor flow/dispersion needs only each block's **external shape** (the plume
sees the obstacle's outside). The "building" enters solely as per-block attributes
— occupancy N_b, F_inf,b(d), floor area, and which outdoor cells form env(b).
Blocks are therefore the correct level of detail; real interiors are never needed.
city_builder7's block-type → occupancy (eff_inh) mapping already supplies N_b; this
model adds F_inf,b(d) and the envelope-cell set per block.

## 5. Assumptions — carefully attributed

**Inherited from Riley et al. (2002) [the model's own assumptions]:**
- R1  Each building interior is a single **well-mixed** zone.
- R2  **No indoor sources** — contaminant is outdoor-origin only.
- R3  P, a, k **constant** over the averaging period.

**Our steady-state specialization:**
- S1  We take the **steady limit** C_in = F_inf·C_out (Riley's t→∞ equilibrium).
      This drops the indoor response time constant τ = 1/(a+k) (~1–2 h). Valid for
      a chronic/steady exposure objective; NOT for a fast passing plume, where the
      indoor lag matters (that is the shelter-in-place regime of **Chan et al.
      2007, *Atmos. Environ.* 41:4962**, which would require the transient form).

**Extra assumptions we must make (discretization/coupling — NOT in Riley):**
- E1  **Intake location.** Riley uses one scalar C_out; we map it to the outdoor
      cells adjacent to the envelope and assume the building intakes the **mean**
      C_out over env(b). (Windward-face weighting is a documented refinement.)
- E2  **Receptor placement.** Indoor-occupant weight is placed on env(b) outdoor
      cells (not interior cells), the discrete realization of C_in = F_inf·C_out.
- E3  **Constant, wind-independent a** per building in the base model. Real a is
      wind/stack-driven and couples to the CFD envelope pressure (Chan et al.
      2007); a wind-dependent a(local U) is a linear-preserving refinement, left
      out of v1.
- E4  **Uniform time-activity** f_in, f_out (NHAPS national means); real fractions
      vary by demographic and land use.
- E5  **Per-building-type P, a, k**, or the uniform F_inf = 0.62 fallback (Allen
      2012) where building-specific values are unavailable.
- E6  **Bins independent** between outdoor and indoor (no coagulation/phase change
      across the envelope); F_inf applied per bin (consistent with Riley's
      size-resolved treatment).

## 6. Implementation — v1 (done) and the path to per-building

**Implemented** in `exposure_solve.cpp` (transport operator M untouched; all
changes are receptor-side). Per cell, on the OUTDOOR field the solver produces:

- **envelope cells** (outdoor fluid face-adjacent to a building, any height) get
  indoor weight `f_in · F_inf` — the discrete C_in = F_inf·C_out;
- **pedestrian cells** (outdoor fluid at z_ped, restricted to the buildings'
  bounding box + ~10 m, so empty upwind/downwind buffers carry no people) get
  outdoor weight `f_out`.

Then the existing `reverse_steady` runs on the unchanged operator and the
Ω-reduction gives J. Verified end-to-end: w-sums match `nEnv·f_in·F_inf` and
`nPed·f_out` exactly; indoor weight dominates outdoor, as expected.

Defaults (env-overridable `FINF`, `F_IN`, `F_OUT`): F_inf = 0.62 (Allen 2012),
f_in = 0.87, f_out = 0.075 (Klepeis 2001).

**v1 simplifications (beyond §5's E-list), to retire with the city_builder7 bridge:**
- V1a  **Uniform occupancy per envelope cell** ⇒ each building's occupancy scales
       with its envelope (surface) area, and indoor exposure samples the *sum* of
       C_out over the envelope. The strict eq.(3) form uses N_b/|env(b)| so the
       building samples the *mean* C_out (E1); that needs real per-building
       occupancy N_b (from eff_inh) and connected-component building labels.
- V1b  **Uniform F_inf** across all buildings (the Allen-2012 population mean);
       per-building P·a/(a+k) awaits building-type attributes.
- V1c  **Uniform outdoor pedestrian density** inside the built-up box.

**Path to per-building:** bring in city_builder7's per-block occupancy (eff_inh)
and labels → replace uniform weights with N_b/|env(b)|·f_in·F_inf,b(d) on each
building's envelope and the block's occupancy-weighted pedestrians, and make F_inf
per-bin (§2). No solver, adjoint, reciprocity, or Ω-reduction change — still only w.

Run: `COLL=hrr HRR_SIGMA=0.98 MAX_WARMUP=80000 bash run_exposure.sh 2681829 0.5 300000 2.0`
