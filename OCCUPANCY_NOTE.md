# Street Inhabitance + Sourced Occupancy — Implementation Note

Adds an explicit **street (outdoor pedestrian / in-transit) exposure pathway**
and replaces the ad-hoc time-budget constants with values sourced from measured
activity data. Sources are stated here and in the code comments.

## 1. Time budget — now sourced (Klepeis et al. 2001, NHAPS)

The builder previously used a constructed split (`TIME_HOME=0.780`,
`TIME_WORK=0.157`, `TIME_PARK=0.064`) with **no street/transit category** —
commuters and pedestrians were absorbed into "home."

Replaced with the **National Human Activity Pattern Survey** 24-hour
population averages — **Klepeis et al. (2001)**, *J. Expo. Anal. Environ.
Epidemiol.* 11(3):231–252 (n = 9,386, EPA-funded, built to feed exposure
models): residence-indoors **69%**, other-indoor **18%**, in-vehicle **5.5%**,
outdoors **7.6%** (≈87% indoors). Corroborated by Canadian CHAPS-2 (Matz et al.
2014: 88.9% indoor / 5.8% outdoor / 5.3% vehicle).

Mapped to the model's exposure pathways (`city_builder7.h`, `occupancy.h`):

| NHAPS microenvironment | % | model pathway |
|---|---|---|
| residence indoor | 69 | residential buildings (infiltration-reduced) |
| other indoor | 18 | business/mixed buildings (infiltration-reduced) |
| in-vehicle | 5.5 | **street** (full outdoor) |
| outdoors | 7.6 | park (porous) + **street**, split by `OUTDOOR_PARK_SHARE` |

Street fraction = in-vehicle + sidewalk share of outdoors ≈ **9.3%** (at
`OUTDOOR_PARK_SHARE = 0.5`). NHAPS does not subdivide "outdoors" into park vs
sidewalk, so that one split is an explicit assumption, flagged as such.

## 2. No time-of-release (diurnal) profile — by design

We do **not** model occupancy as a function of release hour. The event time is
unknown and must be treated as adversarial, so exposure is weighted uniformly
over all 24 hours — i.e., the daily-average occupancy, which is the expectation
over a uniformly-distributed release time. A diurnal profile would let the
layout overfit to one hour's population pattern (same robustness principle as
the fixed population and the wind-rose aggregation). NHAPS does provide
weekday/weekend and 3-hour-segment breakdowns (Tsang & Klepeis 1996,
EPA/600/R-96/148) if a specific scenario is ever mandated, but the default is
the 24-h average.

## 3. Street inhabitance mechanism (`occupancy.h`)

Street occupants are placed on the **road network** — ground-level (z = 1)
fluid cells inside the city footprint. Each non-park building sheds its occupant
count onto its façade-adjacent road cells, so pedestrian density tracks local
activity (the standard assumption in micro-environmental exposure models, e.g.
EPA APEX/SHEDS); the field is normalized to the total street population. Street
exposure = Σ(road-cell occupants × outdoor concentration), with **no
infiltration reduction** — an optional `vehicle_io < 1` can credit in-vehicle
cabin filtration (car cabins I/O ≈ 0.4–0.7) for the transit sub-fraction.

This matters because street occupants breathe undiluted plume-level air; at a
given location their I/O = 1 versus indoor I/O < 1, so per unit local
concentration they are the most exposed group.

`build_street_occupancy()` and `street_exposure()` are implemented and run
end-to-end on real solver output (`demo_occupancy.cpp`); the pedestrian field is
shown in `occupancy_street.png`.

## 4. Finding from the demo — occupant-conservation leak (please review)

Running the two pathways surfaced a latent bug: in the small test city
(which generated **zero business floor area**), the 18% "other-indoor/work"
population had no workplace to be placed in and was **silently dropped** —
placed occupants 287 vs population 349 (**18% lost**).

This is a population-conservation leak in `city_builder7.h::finalize`: when
`total_biz_fa == 0`, `work_people` is multiplied by `0` and never reallocated.
It is also mildly **exploitable** — an optimizer could shrink business zoning to
shed workers and lower the exposure, the same failure class as the retired
`target_density` depopulation hack (now closed by fixing total population). **Recommended fix:** when business floor area is unavailable, reallocate
`work_people` (e.g., fold into `home_people`, or distribute proportionally) so
population is conserved regardless of zoning. One-line guard at each of the two
`finalize` allocation blocks. Flagging rather than silently changing the
builder — say the word and I'll patch it.

## References (this change)
- Klepeis, Nelson, Ott, Robinson, Tsang, Switzer, Behar, Hern & Engelmann (2001). The National Human Activity Pattern Survey (NHAPS). *J. Expo. Anal. Environ. Epidemiol.* 11(3), 231–252.
- Tsang & Klepeis (1996). Descriptive statistics tables from a detailed analysis of the NHAPS data. EPA Final Report EPA/600/R-96/148 (time-of-day segments).
- Matz et al. (2014). Effects of age, season, gender and urban-rural status on time-activity: CHAPS-2. *Int. J. Environ. Res. Public Health* 11, 2108–2124.
- (Pedestrian/micro-environment exposure modeling) U.S. EPA APEX/SHEDS human exposure model documentation.
