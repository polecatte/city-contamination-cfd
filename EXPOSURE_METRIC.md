# The exposure metric — microenvironmental formulation

The population exposure objective follows the **standard microenvironmental exposure
model** (Duan 1982; Ott 1982; Klepeis 2001). No new metric is introduced; this
document states the standard and how it is discretised on the urban model. It
supersedes the per-cell weighting for the population term (which claimed a spatial
resolution in the *people* that does not exist).

## 1. The standard: microenvironmental exposure

A person's exposure over a period T is the time-weighted sum of the concentrations
they experience across the microenvironments j they occupy (Duan 1982; Ott 1982):

    E = Σ_j t_j · C_j                                                         (1)

where t_j is time spent in microenvironment j and C_j its concentration. Population
exposure sums (1) over people. This is the accepted framework behind essentially all
population-exposure and health-impact assessment; our objective is an instance of it,
not a new construction.

For an accidental (burst) release the quantity a person accumulates is the
time-integrated concentration over the plume passage — the **dosage** Θ = ∫C dt — so
C_j in (1) is replaced by the microenvironment dosage Θ_j, and t_j by the fraction of
the exposure period spent there. We resolve **two microenvironments**: indoors and
outdoors (in-transit is not modelled; see §5).

## 2. Two microenvironments, building-anchored

Population is anchored to **buildings**, the resolution at which occupancy is actually
known (land use, floor area, census) — not to CFD cells. Each building b has an
occupancy N_b. Its occupants split their time by the time-activity budget
(Klepeis 2001, NHAPS): fraction f_in indoors, f_out outdoors (f_in ≈ 0.87,
f_out ≈ 0.075; the ~0.055 in-transit remainder is unmodelled).

- **Indoor microenvironment.** Occupants breathe infiltrated air, C_in = F_inf,b·C_out,b
  (single-zone infiltration; Riley et al. 2002, INFILTRATION_MODEL.md), where C_out,b
  is the outdoor concentration the building envelope sees. Their dosage is
  f_in·F_inf,b·Θ_out,b, with Θ_out,b the mean dosage over building b's envelope cells.
- **Outdoor microenvironment.** During their outdoor time, the same occupants breathe
  the outdoor air at street level adjacent to their building: f_out·Θ_street,b, with
  Θ_street,b the mean dosage over the outdoor (pedestrian-level) cells adjacent to b.

Both pathways read the **same outdoor airborne field** Θ_out the solver produces;
neither removes or relocates mass. Deposition on façades is a separate sink, not an
exposure term (no double-count — the two inhalation pathways are separated by the time
budget f_in + f_out ≤ 1).

## 3. The three reported quantities

Applying (1) building-by-building and summing over the population:

    E_indoor  = Σ_b N_b · f_in  · F_inf,b · Θ̄_env(b)                          (2)
    E_outdoor = Σ_b N_b · f_out ·           Θ̄_street(b)                       (3)
    E_total   = E_indoor + E_outdoor                                          (4)

Θ̄_env(b) and Θ̄_street(b) are the mean dosage over building b's envelope and adjacent
street cells respectively. **All three are reported**, because:

- **E_total** (4) is the standard combined microenvironmental exposure — the metric
  chronic-exposure and health-impact work expects, and it is indoor-dominated because
  f_in ≫ f_out (Klepeis 2001).
- **The pathway split** (2),(3) is standard for **acute / accidental** releases: the
  shelter-in-place literature (Chan, Nazaroff et al. 2007) keeps indoor and outdoor
  separate precisely because "is it safer in or out during the plume?" requires them
  un-combined, and because the acute street-level hazard (fast, intense, unattenuated)
  is the headline for a burst — which E_total, being indoor-dominated, would mask.

Reporting all three is therefore the best-supported choice for an accidental-release
scenario: the combined metric matches the chronic convention, the split matches the
acute convention, and both use only building-anchored occupancy data.

## 4. Discretisation (one forward run, three reductions)

The forward burst run produces the outdoor dosage field Θ(x)=∫C dt once. Per building
b (connected component of SOLID cells), the envelope set env(b) = outdoor fluid cells
face-adjacent to b, and the street set street(b) = outdoor fluid cells at pedestrian
height adjacent to b. The receptor weights

    w_in(x)  = Σ_b 1{x∈env(b)}    · N_b · f_in · F_inf,b / |env(b)|
    w_out(x) = Σ_b 1{x∈street(b)} · N_b · f_out          / |street(b)|

make E_indoor = ⟨w_in, Θ⟩, E_outdoor = ⟨w_out, Θ⟩, E_total = ⟨w_in+w_out, Θ⟩ — three
weighted sums of the *same* Θ, no extra solves. (For the adjoint path, φ is linear in
w, so φ_total = φ_in + φ_out likewise needs no extra solve.) Dividing by |env|/|street|
makes each building sample the *mean* dosage over its envelope/street, per (2)–(3).

## 5. Assumptions (stated, per microenvironmental convention)

- **M1 Building-anchored population.** Occupancy is resolved at the building, not the
  cell; per-cell distribution within a building is uniform over its envelope. Occupancy
  N_b comes from the city model (eff_inh); absent it, a floor-area proxy is used.
- **M2 Two microenvironments.** Indoor + outdoor only; the ~5.5% in-transit NHAPS
  fraction is unmodelled (a small, stated omission).
- **M3 Outdoor = residents' outdoor time.** Outdoor exposure represents building
  occupants during their f_out outdoor time, placed adjacent to their building.
  **Transient through-pedestrians** (people not belonging to a building in the scene)
  are NOT modelled — negligible for chronic, potentially relevant for an acute release
  on a busy thoroughfare; a pedestrian-density layer would be required and is out of
  scope. This is the honest limitation of the outdoor term.
- **M4 Single-zone infiltration**, constant per-building F_inf (Riley R1–R3, §
  INFILTRATION_MODEL.md); steady F_inf ignores the indoor air-exchange lag (fine for a
  dosage integral, not for indoor *timing*).
- **M5 Time-activity uniform** across the population (NHAPS national means).

## 6. Why not per-cell population

Velocity and concentration are known per cell; **occupancy is not**. A per-cell
"inhabitance" implies population precision that no data supports — it is a downscaling
rule, not measurement. Anchoring population to buildings matches the resolution of the
people-data to the exposure, and the per-cell field becomes merely the quadrature of a
continuous population-weighted integral E = ∫ρ_pop(x)·Θ(x) dx with ρ_pop a
building-derived density — which is the honest reading of the discretisation in §4.

## 7. Relation to intake fraction

E_total is proportional to the **intake fraction** (mass inhaled per mass emitted;
Bennett et al. 2002; Marshall & Nazaroff), the recognised source→exposure metric — so
the objective is a cell-resolved, microenvironmentally-weighted intake fraction, a
named standard quantity, not an ad-hoc sum.

## Sources
- Duan, N. (1982). Models for human exposure to air pollution. *Environment International* 8:305–309.
- Ott, W.R. (1982). Concepts of human exposure to air pollution. *Environment International* 7:179–196.
- Klepeis, N.E. et al. (2001). NHAPS time-activity. *J. Expo. Anal. Environ. Epidemiol.* 11:231–252.
- Riley, W.J., McKone, T.E., Lai, A.C.K. & Nazaroff, W.W. (2002). Indoor PM of outdoor origin. *Environ. Sci. Technol.* 36:200–207.
- Chan, W.R., Nazaroff, W.W., Price, P.N. & Gadgil, A.J. (2007). Effectiveness of urban shelter-in-place. *Atmos. Environ.* 41:4962–4976.
- Bennett, D.H. et al. (2002). Defining intake fraction. *Environ. Sci. Technol.* 36:207A–211A.
- Allen, R.W. et al. (2012). Residential infiltration of outdoor PM2.5 (MESA Air). *Environ. Health Perspect.* 120:824–830. (F_inf = 0.62.)
