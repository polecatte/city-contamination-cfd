# Indoor Filtration — Implementation Note

Completes the solid-building indoor-exposure pathway and adds indoor air
filtration as a protective-action lever. Every modeling decision cites its
source (in prose and in the code comments).

## What was built

**1. Indoor mass balance with filtration (`infiltration.h`).** Added the
steady-state indoor/outdoor ratio of the well-mixed indoor balance with no
indoor sources:

  C_in/C_out = ( P·a_inf + (1−η_mv)·a_mech ) / ( a_inf + a_mech + k + λ_filt )

- P = size-dependent envelope penetration (Liu & Nazaroff 2003; Stephens & Siegel 2012)
- a_inf = infiltration air-exchange rate; a_mech = mechanical outdoor-air supply (filtered at η_mv)
- k = indoor deposition loss (Lai & Nazaroff 2000)
- **λ_filt = indoor filtration loss rate** — the new term. For a portable cleaner λ_filt = CADR/V (Shaughnessy & Sextro 2006); for HVAC recirculation λ_filt = η_filter · recirc-rate.
- **Source for the balance:** Nazaroff (2004), *Indoor Air* 14(s7):175–183 ("Indoor particle dynamics"); Chen & Zhao (2011), *Atmos. Environ.* 45, 275–288.
- `cadr_to_lambda(CADR, V)` and `indoor_io_ratio(...)` added; `infiltration_factor(...)` is the λ_filt = a_mech = 0 special case.

**2. Solid-building voxelization (`voxelize.h`).** `voxelize(p, r, solid_buildings=true)` rasterizes non-park buildings as **solid full-bounce-back** (perm = 0) for correct aerodynamics (COST 732; Tominaga et al. 2008), retaining façade deposition; parks stay porous (Merlier et al. 2018). Default remains the legacy permeable shell, so existing callers are unaffected.

**3. Indoor exposure stage (`indoor_exposure.h`).** For each solid building, samples the outdoor concentration on the façade-adjacent ring, applies the I/O ratio, and accumulates occupant-weighted indoor exposure — the term that feeds airborne inhalation in `exposure_objective.py`. Park-goers (the time-budget's 6.4% park share) remain in the porous park cells and receive near-outdoor exposure, which is physically appropriate.

## End-to-end demonstration (`demo_indoor.cpp`, real solver run)

Small city → solid buildings → LBM with the ABL turbulent inlet → outdoor
concentration field → indoor exposure across a filtration sweep. The run is
physically sane (mass budget closes: 98.7% airborne, 0.3% deposited, 1.0%
outflow; concentrations positive). Indoor filtration as a protective action
(dp = 0.5 µm mass-mean, a_inf = 0.5/h):

| filtration | λ_filt (1/h) | I/O ratio | total indoor exposure |
|---|---|---|---|
| none | 0 | 0.53 | baseline |
| portable cleaner | 2 (CADR/V) | 0.15 | −71% |
| HEPA | 5 | 0.07 | −86% |
| strong | 10 | 0.04 | −93% |

See `indoor_filtration.png`.

## Important modeling caveat (carried forward)

Evaluate the I/O ratio at a PM class's **mass-mean diameter (~0.3–0.5 µm for
PM2.5), not its cutoff** (2.5 or 10 µm). The cutoff badly under-predicts class
infiltration because the mass sits well below it; the demo uses 0.5 µm
accordingly.

## Remaining wiring (one external file)

`exposure_objective.py` (not in this workspace) should call the indoor-exposure
stage: indoor airborne inhalation = Σ occupants · F_inf,filt · (façade TIAC);
deposited-surface exposure continues from the deposition field. λ_filt and a_inf are scenario
parameters (protective measures / building stock), not optimizer design
variables, so they don't reintroduce a reward-hackable knob.

## References (this change)
- Nazaroff (2004). Indoor particle dynamics. *Indoor Air* 14(s7), 175–183.
- Chen & Zhao (2011). Review of I/O ratio, infiltration factor, penetration factor. *Atmos. Environ.* 45, 275–288.
- Liu & Nazaroff (2001). Modeling pollutant penetration across building envelopes. *Atmos. Environ.* 35, 4451–4462.
- Liu & Nazaroff (2003). Particle penetration through building cracks. *Aerosol Sci. Technol.* 37, 565–573.
- Lai & Nazaroff (2000). Modeling indoor particle deposition. *J. Aerosol Sci.* 31, 463–476.
- Shaughnessy & Sextro (2006). What is an effective air cleaning device? (CADR/effectiveness). *J. Occup. Environ. Hyg.* 3, 169–181.
- Stephens & Siegel (2012). Penetration of ambient submicron particles into buildings. *Indoor Air* 22, 501–513.
- Merlier, Jacob & Sagaut (2018). LBM-LES of pollutant dispersion in street canyons incl. tree planting. *Atmos. Environ.* 195, 89–103.
- Tominaga et al. (2008). AIJ guidelines for practical applications of CFD to pedestrian wind environment. *J. Wind Eng. Ind. Aerodyn.* 96, 1749–1761.
