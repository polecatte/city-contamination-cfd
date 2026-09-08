"""param_space_v2.py — 13-D IDENTIFIABLE parameter space for the v2 builder.

This space is deliberately quotiented so that distinct parameter vectors give
distinct cities (up to measure-zero simplex boundaries). Every redundancy found
in the earlier 15-D space is removed either by dropping a parameter or by
re-bounding it. The objective is exposure under a FIXED wind direction (+x); the
fixes below are stated relative to that objective, not to the wind-free geometry.

Redundancies removed
--------------------
1. GLOBAL ORIENTATION (the square's D4 symmetry).  A directional wind breaks D4
   down to a single reflection across the wind axis (y → −y).  So:
     • wind_angle is FIXED (canonical +x), not searched — you never search over
       "which way the city points", only over shape relative to the wind.
     • corridor_angle (business) is FIXED at 0° and park_corridor_angle at 90°.
       Both of these lines are symmetric under y→−y, so neither can create a
       reflected duplicate.  (A free absolute angle would; a wind-relative angle
       would need bounding to a [0°,90°] wedge — not done here, angles are fixed.)
     • the gradient (sector) terms phi_grad_x/y and park_w_grad_x/y are the ONLY
       field terms odd under y→−y.  They are dropped (fixed 0), which removes the
       last way to build a wind-axis mirror image.
   block_w vs block_d is a 90° ROTATION, which is NOT in the residual symmetry
   group (rotation was broken by the wind), so the two are independent and are
   BOTH kept — elongating a block along vs across the wind changes the objective.

2. POTENTIAL-WEIGHT SCALE (projective redundancy).  Because block ranks are
   min–max normalized inside scatter_order(), multiplying a field's whole weight
   vector by any t>0 gives the identical city AT EVERY scatter value (verified).
   So only the DIRECTION of (radial, corridor, bipeak) matters.  Each field is
   therefore parameterized as a point on the 2-simplex via stick-breaking (2 free
   params instead of 3), pinning the scale.  This one change also removes:
     • the "near-flat field swamped by scatter" case (a near-zero field is no
       longer representable — the field always has unit L1 weight), and
     • the "park_centrality magnitude saturates at scatter=0" case (centrality is
       now the radial share/sign, not an independent magnitude).

Search dimensions (13)
----------------------
Morphology (7):
  block_w, block_d, cbd_peak, cbd_decay, park_fraction, roughness, street_width
Business field (3):
  biz_radial_share  — radial share of the field; remainder → corridor+bipeak
  biz_cor_bip       — split of the remainder between corridor and bipeak
  biz_scatter       — 0=contiguous cluster, 1=maximin dispersal
Park field (3):
  park_radial       — 0=edge greenbelt … 0.5=no radial … 1=central park
  park_cor_bip      — split of the non-radial remainder between corridor/bipeak
  park_scatter      — 0=one contiguous green mass, 1=dispersed pocket parks
"""
import numpy as np

# ── Search space ──────────────────────────────────────────────────────────
# (name, lo, hi, scale)  scale: 'lin' or 'log'
PARAM_SPACE_V2 = [
    # Morphology
    ("block_w",             15.0,    50.0,  "lin"),
    ("block_d",             15.0,    45.0,  "lin"),
    ("cbd_peak",            20.0,    80.0,  "lin"),
    ("cbd_decay",            1e-6,   5e-5,  "log"),
    ("park_fraction",        0.05,    0.50, "lin"),
    ("roughness",            0.00,    0.80, "lin"),
    ("street_width",         6.0,    30.0,  "lin"),
    # Business field (normalized shape + orientation + scatter)
    ("biz_radial_share",     0.00,    1.00, "lin"),
    ("biz_cor_bip",          0.00,    1.00, "lin"),
    ("corridor_angle_deg",   0.00,  180.00, "lin"),   # biz spine axis (line: period 180°)
    ("biz_scatter",          0.00,    1.00, "lin"),
    # Park field (signed-radial normalized shape + orientation + scatter)
    ("park_radial",          0.00,    1.00, "lin"),
    ("park_cor_bip",         0.00,    1.00, "lin"),
    ("park_corridor_angle_deg", 0.00, 180.00, "lin"), # green spine axis (line: period 180°)
    ("park_scatter",         0.00,    1.00, "lin"),
    # Environment — wind canonicalized to the square's D4 fundamental wedge.
    ("wind_angle_deg",       0.00,   45.00, "lin"),
]

NAMES_V2 = [n for (n, lo, hi, s) in PARAM_SPACE_V2]

# ── Fixed run conditions (deliberately NOT searched — see module docstring) ──
# Canonicalizing WIND to the [0°,45°] wedge uses up the square's D4 freedom, which
# pins the frame. With the frame pinned, the corridor ORIENTATIONS and wind angle
# are all free, non-redundant design axes (they were removed only because the frame
# was previously fixed by the layout instead). Gradient (sector) terms stay off for
# now — not for symmetry reasons any more, but because a monotone density ramp is
# better handled as one of the richer field generators under discussion.
FIXED_V2 = {
    "city_w":               600.0,
    "city_h":               600.0,
    "population_total":   20000.0,
    "patchiness":             0.0,
    "phi_grad_x":             0.0,
    "phi_grad_y":             0.0,
    "park_w_grad_x":          0.0,
    "park_w_grad_y":          0.0,
}


def _simplex2(a, b):
    """Stick-breaking of two [0,1] params onto the 3-vertex simplex (v0,v1,v2)."""
    v0 = a
    v1 = (1.0 - a) * b
    v2 = (1.0 - a) * (1.0 - b)
    return v0, v1, v2


def to_physical_v2(x_unit):
    """Map a [0,1]^16 unit-cube vector to a physical parameter dict.

    Field weights are emitted already NORMALIZED (L1 = 1), so the physical Φ has
    no residual scale freedom. Corridor orientations and wind angle come straight
    from the unit vector (wind is confined to the [0,45]° wedge, which is what makes
    the free orientations non-redundant).
    """
    x_unit = np.asarray(x_unit, dtype=float)
    if len(x_unit) != len(PARAM_SPACE_V2):
        raise ValueError(f"Expected {len(PARAM_SPACE_V2)} dims, got {len(x_unit)}")

    phys = {}
    for (name, lo, hi, scale), xi in zip(PARAM_SPACE_V2, x_unit):
        xi = float(np.clip(xi, 0.0, 1.0))
        phys[name] = (lo * (hi / lo) ** xi) if scale == "log" else (lo + xi * (hi - lo))

    # ── Business field: simplex over {radial, corridor, bipeak} ──────────────
    radial, corridor, bipeak = _simplex2(phys["biz_radial_share"], phys["biz_cor_bip"])
    phys["phi_radial"]   = radial
    phys["phi_corridor"] = corridor
    phys["phi_bipeak"]   = bipeak
    phys["phi_bipeak_x"] = bipeak     # equal x&y → four business nodes
    phys["phi_bipeak_y"] = bipeak

    # ── Park field: signed radial (central/edge) + corridor/bipeak on a simplex ─
    s = 2.0 * phys["park_radial"] - 1.0            # −1 edge belt … 0 none … +1 central
    rad_share = abs(s)
    p_cor, p_bip = (1.0 - rad_share) * phys["park_cor_bip"], \
                   (1.0 - rad_share) * (1.0 - phys["park_cor_bip"])
    phys["park_centrality"]  = 0.5 + 0.5 * s        # phi_park uses 2*(pc-0.5)=s as w_radial
    phys["park_w_corridor"]  = p_cor
    phys["park_w_bipeak_x"]  = p_bip                # equal x&y → four green nodes
    phys["park_w_bipeak_y"]  = p_bip
    phys["park_radial_signed"] = s                  # for reporting

    # Merge fixed defaults (setdefault so derived keys above are not overwritten)
    for k, v in FIXED_V2.items():
        phys.setdefault(k, v)
    phys.setdefault("wind_angle", phys.get("wind_angle_deg", 0.0))

    return phys


def print_table(rows, params=None):
    """Print a parameter table.  rows = [(name, phys_dict), ...]"""
    if params is None:
        params = NAMES_V2
    hdr = "param".ljust(24) + "".join(n[:12].rjust(14) for n, _ in rows)
    print(hdr)
    print("-" * (24 + 14 * len(rows)))
    for pn in params:
        line = pn.ljust(24)
        for _, phys in rows:
            v = phys.get(pn, 0.0)
            line += (f"{v:.2e}" if pn == "cbd_decay" else f"{v:.2f}").rjust(14)
        print(line)
