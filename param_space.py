"""param_space.py — THE single place that defines the optimization problem.

Changing the city builder's parameter set, the fixed run conditions, the exposure
coefficients, or the resolution schedule happens HERE and nowhere else; the
optimizer core (bayesopt.py) and the evaluator (evaluate.py) are untouched.

Each tunable parameter: (name, low, high, scale) with scale in {'lin','log'}.
'name' must match a key the C++ driver (evaluate.cpp) understands.
"""

# ── Tunable layout parameters (the optimizer's search space) ──
# Edit freely — add/remove rows; the optimizer adapts to the dimensionality.
PARAM_SPACE = [
    ("block_w",         24.0,   72.0,  'lin'),
    ("block_d",         16.0,   40.0,  'lin'),
    ("cbd_peak",        20.0,  120.0,  'lin'),
    ("cbd_decay",       1e-6,   3e-5,  'log'),
    ("patchiness",       0.0,   1.00,  'lin'),   # business SPREAD (v8.3): 0=concentric core,
                                                 #   1=business follows coherent noise patches
    ("park_centrality",  0.0,    1.0,  'lin'),   # WHERE parks go (placement bias)
    ("park_fraction",    0.0,   0.40,  'lin'),   # HOW MANY blocks are parks (direct;
                                                 # no open-space requirement, no merging)
    ("roughness",        0.0,   0.80,  'lin'),   # height heterogeneity ≈ σ(ln h) ≈ height CoV
                                                 #   (v8.2: log-normal scatter, range widened
                                                 #   0.40→0.80; patchiness removed, S_i≈1%)
    # v8: coverage REPLACED by street_width (m). Street width is a physical variable
    # (canyon W → aspect H/W, Oke 1988), unlike the abstract lot-fill ratio; plan
    # density λp = block_area/(block+street)² is now a derived diagnostic, logged
    # not steered. For an isotropic grid the driver sets road_w_x = road_w_y =
    # street_width; split into two dims for directional (anisotropic) canyons.
    ("street_width",     8.0,   40.0,  'lin'),
    # ── STAGED, NOT YET ACTIVE — wind angle relative to the grid-aligned streets ──
    # The clean way to study orientation: keep streets grid-aligned, rotate the
    # inflow. Range is [0, π/4] (square-grid symmetry). The solver supports oblique
    # wind via single-face skewed inflow (lbm_solver.cpp wind_setup), BUT the lateral
    # (±y) boundaries are zero-gradient (clamped) rather than a fresh ABL inflow, so
    # there is no proper upstream fetch in y. This path has only ever run at angle 0.
    # DO NOT add it to the live search until an oblique-inflow validation run
    # (mass closure, plume stays clear of the +y boundary, sane mean flow at
    # 15/30/45°) passes — and likely enlarge buf_yp for the +y plume drift.
    # ("wind_direction",  0.0,   0.7853981634,  'lin'),
    # NOTE: target_density REMOVED from the search space (v8). Population is now a
    # FIXED total headcount (population_total) allocated by floor area; designs are
    # compared at equal population (total exposure Σ w·C). Changing it => a NEW run.
    # NOTE: mixed_frac REMOVED (v8.1). Mixed-use zoning relabelled residential blocks
    # without changing the built envelope (≈0 morphological variation; sensitivity
    # max S_i ≈ 1%), and its commercial share was absorbed by the employment-
    # balancing pass — so it has been retired entirely (no MIXED usage class).
    # NOTE: base_height FIXED (v8.2) and patchiness REPURPOSED (v8.3). base_height
    # (~4% of morphology variance) was largely redundant with the cbd_peak tail and
    # is now held at BASE_HEIGHT_M (9 m ≈ 3 floors) in the builder — just below the
    # RES_LOW 4-floor threshold, so the periphery resolves to low-density housing.
    # `patchiness` was retired as a zoning-noise knob then reintroduced as a
    # business-SPREAD control (0 = concentric core, 1 = coherent off-centre patches).
    # roughness widened to [0,0.8] and made a log-normal height-CoV knob. Space is 9-D.
]
# NOTE: cbd_aspect / cbd_angle / biz_aspect were removed — the city is now a
# strictly circular ring/central-core model, so those ellipticity knobs are
# no-ops in the builder.
#
# ── DOMAIN (derived, not searched) ──────────────────────────────────────────
# The simulation domain is sized to the COST-732 / Tominaga et al. (2008) bare
# minimum and held FIXED across a sweep: one grid, sized for the TALLEST design,
# applied to all (never compare different-sized domains). For the wind-direction
# sweep θ∈[0,π/4] the wind enters from the −x/−y corner and leaves through +x/+y,
# so buffers are 5H on the inflow faces (−x,−y) and 15H on the outflow faces
# (+x,+y), with 5H top headroom (domain height = 6H). Power-of-2 is NOT enforced.
# Set by city_builder7.minimal_domain(p, Hmax, 5, 15) + nz_cost732(Hmax) in the
# driver — the FIXED buf_* below are only a fallback for non-sweep/manual runs.
# Consequence: asymmetric buffers ⇒ the city is offset toward the inflow corner
# (extra room downstream), i.e. NOT centred — the intended CFD layout.

# ── Fixed per-BO-run conditions (ONE source location, ONE wind direction) ──
FIXED = dict(
    city_w          = 600.0,
    population_total = 20000.0,  # v8 FIXED total headcount (replaces target_density)
    city_h          = 600.0,
    wind_angle    = 0.0,        # radians; +x. FIXED for this study.
    U_inlet       = 5.0,        # m/s
    source_x      = 50.0,       # m (domain coords)
    # source_y omitted → driver defaults to city centreline
    source_z      = 4.0,
    Q_source      = 1.0,        # solver units/s; objective treats as arbitrary exposure units
    # POLYDISPERSE source size distribution (supersedes single particle_diam).
    # Mass-weighted lognormal size distribution (Hinds 1999). Defaults: typical
    # atmospheric release MMAD~1 um (fine plumes ~0.6 um, coarse ~6 um; GSD~2).
    # These are SCENARIO constants, not design variables.
    source_mmad      = 1.0e-6,  # mass median aerodynamic diameter (m)
    source_gsd       = 2.0,     # geometric standard deviation
    n_size_bins      = 6,       # sectional bins for transport
    particle_density = 1800.0,
    particle_diam    = 10.0e-6, # legacy single-size fallback (n_size_bins<=1)
    release_time  = 300.0,      # s of dispersion (Phase B)
    max_warmup    = 20000,      # cap on flow warm-up steps
    avg_threshold = 2e-3,       # flow-steadiness threshold to end warm-up
    check_interval= 500,
    # Fallback buffers for manual/non-sweep runs ONLY. Production sweeps OVERRIDE
    # these via minimal_domain(p, Hmax, 5, 15) (COST-732 directional, see DOMAIN
    # note above), so editing them here does not affect ranking/robustness runs.
    buf_xn=120.0, buf_xp=200.0, buf_yn=120.0, buf_yp=120.0,
)

# ── Multi-resolution schedule: coarse exploration → fine refinement ──
# Each stage uses a binary compiled at that cell size (-DCELL_SIZE_M).
# seed_topk best points from a stage seed the next stage's initial design.
STAGES = [
    dict(name='coarse', binary='./solver_eval_8m', cell=8.0,
         n_init=12, n_iter=30, seed_topk=4),
    dict(name='fine',   binary='./solver_eval_4m', cell=4.0,
         n_init=4,  n_iter=20, seed_topk=0),
]

# ── Early-termination knobs (per stage) ──
EARLY_STOP = dict(rel_tol=0.01, patience=8, ei_tol=1e-4)


# ── unit-cube <-> physical mapping (used by the evaluator) ──
import numpy as np
def to_physical(x_unit):
    """Map a [0,1]^d point to a dict of named physical parameter values."""
    out = {}
    for i,(name,lo,hi,scale) in enumerate(PARAM_SPACE):
        u = float(np.clip(x_unit[i], 0, 1))
        out[name] = lo*(hi/lo)**u if scale=='log' else lo + (hi-lo)*u
    return out

DIM = len(PARAM_SPACE)
