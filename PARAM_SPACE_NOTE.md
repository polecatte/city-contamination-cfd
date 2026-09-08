# PARAM_SPACE_NOTE — the 9-D search space and why each bound is where it is

The live search space is defined in `param_space.py`. Units are metres unless noted;
the anchoring physics is in parentheses. `to_physical` maps the [0,1]^9 unit cube to
these ranges (linear except `cbd_decay`, which is log).

| # | parameter | bounds | scale |
|---|-----------|--------|-------|
| 1 | block_w | 24 – 72 m | lin |
| 2 | block_d | 16 – 40 m | lin |
| 3 | cbd_peak | 20 – 120 m | lin |
| 4 | cbd_decay | 1e-6 – 3e-5 m⁻² | log |
| 5 | patchiness | 0 – 1 | lin |
| 6 | park_centrality | 0 – 1 | lin |
| 7 | park_fraction | 0 – 0.40 | lin |
| 8 | roughness | 0 – 0.80 | lin |
| 9 | street_width | 8 – 40 m | lin |

**block_w [24, 72] m** — building/block footprint width (E–W). Fine urban grain
(24 m ≈ a small perimeter-block/townhouse parcel) to coarse superblock (72 m). Below
~20 m the footprint is only ~5 cells wide and the 4 m LBM cell can't resolve the
building or its canyon cleanly; past ~72 m blocks are monolithic and too few remain
to place zoning/parks. With `street_width` this sets the grid pitch and plan density λp.

**block_d [16, 40] m** — footprint depth (N–S). Deliberately a different range from
`block_w` so the optimizer can make elongated, anisotropic blocks (canyon orientation
matters once wind angle is active). 16 m ≈ a single building row (~4 cells, the
resolvable minimum). Narrower span reflects that building depth is physically more
constrained than street frontage — a modeling choice, not a hard law.

**cbd_peak [20, 120] m** — height added at the core, on top of the fixed 9 m base.
20 m ≈ a modest 6–7-storey downtown; 120 m ≈ a major high-rise CBD (~40 floors). Also
implicitly capped per block by `SLENDERNESS = 7 ×` the smaller footprint dimension.

**cbd_decay [1e-6, 3e-5], log** — Gaussian falloff `exp(−decay·r²)`. Read as the
e-folding radius L = 1/√decay: 1e-6 → ~1000 m (broad, gentle gradient across the whole
city), 3e-5 → ~183 m (tight, compact downtown). Log scale because it is a rate spanning
~1.5 decades and the interesting variation is multiplicative.

**patchiness [0, 1]** — business spread (unitless blend weight). 0 = concentric
business core; 1 = business follows a coherent noise field into off-centre patches.
[0,1] is literally the interpolation weight in `r_eff = (1−p)·r + p·noise·R`.

**park_centrality [0, 1]** — park placement bias (unitless). 0 = edge greenbelt ring,
0.5 = evenly distributed, 1 = central park cluster. Natural range for the radial-bias
term `CEN_GAIN·(pc−0.5)·(Rref−dc)`.

**park_fraction [0, 0.40]** — fraction of blocks made parks. 0 = none. The 0.40
ceiling is pragmatic: U.S. municipal park-land share is typically ~5–25 % (Trust for
Public Land ParkScore), so 0.40 is generous headroom while still leaving enough built
blocks to house the fixed population without forcing extreme heights. A separate hard
safety clamp `MAX_PARK_FRAC = 0.60` sits in the builder; the search stops short of it.

**roughness [0, 0.80]** — height heterogeneity ≈ σ(ln h) ≈ height coefficient of
variation. The log-normal construction makes σ(ln h) = roughness exactly, so 0 =
uniform canopy (skimming flow) and 0.80 ≈ a ~5× tall/short ratio (towers among
low-rise). Real urban height CoV runs ~0.3–0.6 (Xie, Coceal & Castro 2008, *BLM*
129:1; Nakayama, Takemi & Nagai 2011, *JAMC* 50:1692), so 0.80 is high-but-plausible
headroom. Log-normal keeps every height positive.

**street_width [8, 40] m** — canyon width → aspect ratio H/W (Oke 1988). The
physically-grounded replacement for the retired coverage knob. 8 m ≈ a narrow lane
(deep canyon), 40 m ≈ a wide avenue. With heights of 9–168 m, H/W sweeps from ~0.2
(isolated roughness) through Oke's wake-interference band (~0.3–0.65) well into the
skimming regime (>0.65) — i.e. it spans all three canyon-flow regimes. Lower 8 m is
also ~2 cells (resolution/drivability floor); above 40 m the city gets too sparse.

**wind_direction [0, π/4] — STAGED, INACTIVE.** Bounded by square-grid symmetry
(angles beyond 45° are redundant). Held out of the search because the lateral (±y)
boundaries are a symmetry plane rather than a fresh ABL inflow, so oblique wind has no
upstream fetch in y. Needs an oblique-inflow validation pass first (see `HANDOFF.md §8`).

**Fixed or retired** (so the list is complete): `base_height` fixed at 9 m (3 floors);
`population_total` fixed (compare designs at equal headcount; replaces `target_density`);
`coverage → street_width`; `biz_inner_frac → patchiness`; `mixed_frac` retired;
`cbd_aspect/cbd_angle/biz_aspect` removed (strictly circular model). The simulation
domain is derived from city size + COST-732 clearances, not searched.
