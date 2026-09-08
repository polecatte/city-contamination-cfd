"""gen_paramspace.py — generate a spread of example cities by sampling the
project's LIVE 9-D parameter space (param_space.py) and driving the real
city_builder7 via ./gen_paramspace, then render 2D-plan + 3D-massing panels
with render_city.py's draw functions.

Each archetype is a point in the [0,1]^9 unit cube (dims in PARAM_SPACE order),
mapped to physical units by param_space.to_physical — i.e. we exercise the exact
search space the optimizer sees. Corner-ish points are chosen to make the knobs
legible, not to be optimal designs.
"""
import os, subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from param_space import PARAM_SPACE, to_physical, FIXED
from render_city import load_city, draw_2d, draw_3d, COLORS, LABELS

NAMES = [n for (n, lo, hi, s) in PARAM_SPACE]
OUT = "/mnt/user-data/outputs"
TMP = "/tmp/paramspace"
os.makedirs(OUT, exist_ok=True)
os.makedirs(TMP, exist_ok=True)

# dim order: block_w, block_d, cbd_peak, cbd_decay, patchiness,
#            park_centrality, park_fraction, roughness, street_width
#   value 0 -> low bound, 1 -> high bound (log for cbd_decay).
ARCHETYPES = [
    # name, short caption, unit-cube vector
    ("compact_highrise",
     "Compact high-rise core\nsmall blocks · narrow streets · tall tight CBD",
     [0.10, 0.15, 0.95, 0.90, 0.00, 1.00, 0.05, 0.25, 0.10]),
    ("lowrise_sprawl",
     "Low-rise superblocks\nlarge blocks · wide avenues · low broad core",
     [0.95, 0.95, 0.10, 0.10, 0.00, 0.50, 0.08, 0.10, 0.95]),
    ("green_city",
     "Green city\nhigh park fraction · central park cluster",
     [0.40, 0.40, 0.45, 0.50, 0.00, 1.00, 0.95, 0.20, 0.35]),
    ("greenbelt_ring",
     "Greenbelt ring\nparks pushed to the edge · moderate core",
     [0.40, 0.40, 0.45, 0.55, 0.00, 0.00, 0.70, 0.20, 0.35]),
    ("polycentric_patchy",
     "Polycentric / patchy\nbusiness follows off-centre noise patches",
     [0.30, 0.30, 0.70, 0.85, 1.00, 0.50, 0.15, 0.35, 0.25]),
    ("rough_skyline",
     "Rough skyline\nhigh height heterogeneity · towers among low-rise",
     [0.35, 0.35, 0.60, 0.45, 0.20, 0.50, 0.10, 0.95, 0.25]),
]

BIN = "./gen_paramspace"

def build(name, xunit):
    phys = to_physical(np.array(xunit, float))
    args = [f"{phys[n]:.8g}" for n in NAMES]
    out = os.path.join(TMP, f"{name}.txt")
    res = subprocess.run([BIN, *args, out], capture_output=True, text=True, check=True)
    print(res.stdout.strip())
    return out, phys

def panel(items, outfile, suptitle):
    n = len(items)
    fig = plt.figure(figsize=(5.4 * n, 9.8))
    for c, (name, caption, xunit) in enumerate(items):
        f = os.path.join(TMP, f"{name}.txt")
        bl, m = load_city(f)
        cell = m["cell"]
        ax2 = fig.add_subplot(2, n, c + 1)
        draw_2d(ax2, bl, m, cell=cell)
        ax2.set_title(f"{caption}\n{m['N']} blk · max {m['maxH']:.0f} m · {m['pop']:.0f} pop",
                      fontsize=9, fontweight="bold")
        ax2.set_xticks([]); ax2.set_yticks([])
        ax3 = fig.add_subplot(2, n, n + c + 1, projection="3d")
        draw_3d(ax3, bl, m, cell=cell)
        ax3.set_title(f"max {m['maxH']:.0f} m ({int(m['maxH']/3)} floors)", fontsize=9)
    leg = [Patch(facecolor=COLORS[i], edgecolor="#444", label=LABELS[i]) for i in range(4)]
    fig.legend(handles=leg, loc="lower center", ncol=4, fontsize=10, frameon=True)
    fig.suptitle(suptitle, fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    plt.savefig(outfile, dpi=140, bbox_inches="tight")
    plt.close()
    print("saved", outfile)

def main():
    print(f"# population fixed at {FIXED['population_total']:.0f}; "
          f"city {FIXED['city_w']:.0f}x{FIXED['city_h']:.0f} m; wind angle "
          f"{FIXED['wind_angle']:.2f} rad\n")
    # Build all, print the physical parameter table.
    rows = []
    for name, caption, x in ARCHETYPES:
        _, phys = build(name, x)
        rows.append((name, phys))
    # Parameter table
    print("\n# Physical parameters per archetype (from to_physical):")
    hdr = "param".ljust(16) + "".join(n[:14].rjust(16) for n, _ in rows)
    print(hdr)
    for pn in NAMES:
        line = pn.ljust(16)
        for _, phys in rows:
            v = phys[pn]
            line += (f"{v:.2e}" if pn == "cbd_decay" else f"{v:.2f}").rjust(16)
        print(line)

    # Two comparison panels (3 each) so the panels stay readable.
    panel(ARCHETYPES[:3], os.path.join(OUT, "example_cities_1.png"),
          "Example cities from the 9-D parameter space — density & core  (pop fixed 20,000)")
    panel(ARCHETYPES[3:], os.path.join(OUT, "example_cities_2.png"),
          "Example cities from the 9-D parameter space — parks, zoning & skyline  (pop fixed 20,000)")

if __name__ == "__main__":
    main()
