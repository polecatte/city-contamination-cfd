"""param_space_density.py — parameter space for the density-field city builder.

This REPLACES the potential-field / scatter / RD zoning parameters with a compact,
physically-grounded description that varies the POPULATION DISTRIBUTION directly —
the receptor half of the exposure metric J = Σ w(x)·C(x) — while keeping land use
and built form coupled (dense job centres are genuinely tall, so they shape C too).

Two groups:

Morphology (flow side — sets the concentration field C):
  block_w, block_d   — block footprint → plan-area density λp and canyon aspect
  street_width       — canyon width → aspect ratio H/W
  roughness          — per-block height variance (turbulence, downwash)
  wind_angle_deg     — wind vs grid, in the square's [0,45]° fundamental wedge
                       (enters the CFD solve, not the generator)

Population distribution (receptor side — sets w, and via heights also C):
  peak_height        — building height (m) at the densest employment peak
  emp_centers        — number of employment centres (1..4)  [discrete]
  emp_centrality     — 0 = centres on an outer ring (spread), 1 = all central
  emp_concentration  — 0 = broad centres, 1 = tight peaks
  res_gradient       — residential falloff exp(−b·r/R); 0 = uniform sprawl
  jh_mix             — 0 = segregated (housing avoids job cores), 1 = mixed-use
  park_fraction      — open-space share (ventilation openings + recreation receptor)
"""
import numpy as np

# (name, lo, hi, scale)  scale: 'lin', 'log', or 'int'
PARAM_SPACE_DENSITY = [
    # Morphology
    ("block_w",            15.0,  50.0, "lin"),
    ("block_d",            15.0,  45.0, "lin"),
    ("street_width",        6.0,  30.0, "lin"),
    ("roughness",           0.0,   0.8, "lin"),
    ("wind_angle_deg",      0.0,  45.0, "lin"),
    # Population distribution
    ("peak_height",        30.0,  90.0, "lin"),
    ("emp_centers",         1.0,   4.0, "int"),
    ("emp_centrality",      0.0,   1.0, "lin"),
    ("emp_concentration",   0.0,   1.0, "lin"),
    ("res_gradient",        0.0,   3.0, "lin"),
    ("jh_mix",              0.0,   1.0, "lin"),
    ("park_fraction",       0.05,  0.40, "lin"),
]

NAMES_DENSITY = [n for (n, lo, hi, s) in PARAM_SPACE_DENSITY]

FIXED_DENSITY = {
    "city_w":             600.0,
    "city_h":             600.0,
    "population_total": 20000.0,
}


def to_physical_density(x_unit):
    """Map a [0,1]^12 unit-cube vector to a physical parameter dict."""
    x_unit = np.asarray(x_unit, dtype=float)
    if len(x_unit) != len(PARAM_SPACE_DENSITY):
        raise ValueError(f"Expected {len(PARAM_SPACE_DENSITY)} dims, got {len(x_unit)}")
    phys = {}
    for (name, lo, hi, scale), xi in zip(PARAM_SPACE_DENSITY, x_unit):
        xi = float(np.clip(xi, 0.0, 1.0))
        if scale == "log":
            phys[name] = lo * (hi / lo) ** xi
        elif scale == "int":
            phys[name] = int(np.clip(int(lo + xi * (hi - lo + 1)), int(lo), int(hi)))
        else:
            phys[name] = lo + xi * (hi - lo)
    for k, v in FIXED_DENSITY.items():
        phys.setdefault(k, v)
    return phys


def cli_args(phys, outfile):
    """Ordered argument list for ./gen_density_city."""
    return [f"{phys['block_w']:.5g}", f"{phys['block_d']:.5g}",
            f"{phys['roughness']:.5g}", f"{phys['street_width']:.5g}",
            f"{phys['park_fraction']:.5g}", f"{phys['peak_height']:.5g}",
            f"{int(phys['emp_centers'])}", f"{phys['emp_centrality']:.5g}",
            f"{phys['emp_concentration']:.5g}", f"{phys['res_gradient']:.5g}",
            f"{phys['jh_mix']:.5g}", outfile]


def print_table():
    print(f"{'#':>2}  {'parameter':22}{'range':>18}  {'scale':6}  role")
    print("-" * 78)
    roles = {
        "block_w":"morph: plan density λp / aspect", "block_d":"morph: plan density λp / aspect",
        "street_width":"morph: canyon aspect H/W", "roughness":"morph: height variance",
        "wind_angle_deg":"env: wind vs grid (CFD solve)",
        "peak_height":"pop: max height at job peak", "emp_centers":"pop: # employment centres",
        "emp_centrality":"pop: centres central↔ring", "emp_concentration":"pop: peak sharpness",
        "res_gradient":"pop: housing falloff", "jh_mix":"pop: segregated↔mixed-use",
        "park_fraction":"open space / recreation receptor",
    }
    for i,(n,lo,hi,s) in enumerate(PARAM_SPACE_DENSITY,1):
        rng = f"[{lo:g}, {hi:g}]" + ("  (int)" if s=="int" else "")
        print(f"{i:>2}  {n:22}{rng:>18}  {s:6}  {roles.get(n,'')}")


if __name__ == "__main__":
    print(f"Density-field parameter space — {len(PARAM_SPACE_DENSITY)} dimensions\n")
    print_table()
