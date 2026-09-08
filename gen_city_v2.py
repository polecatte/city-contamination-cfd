"""gen_city_v2.py — dynamic suite of v2 cities (identifiable 13-D space).

Uses gen_zoning_demo (city_zoning.h / rezone()) as the backend and
param_space_v2.to_physical_v2() to map archetype unit-cube vectors to physical
parameters.  Field weights are emitted normalized, so no two archetypes differ
only by a redundant scale.

Archetype unit-cube vector layout (16 dims):
  [block_w, block_d, cbd_peak, cbd_decay, park_fraction, roughness, street_width,
   biz_radial_share, biz_cor_bip, corridor_angle_deg, biz_scatter,
   park_radial, park_cor_bip, park_corridor_angle_deg, park_scatter,
   wind_angle_deg]
Angles are unit-scaled: corridor 0.0→0°, 0.5→90°; wind 0.0→0°, 1.0→45°.
Wind does not affect the generated 2-D plan (it enters only the CFD/exposure solve).
"""
import os, subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from param_space_v2 import PARAM_SPACE_V2, NAMES_V2, to_physical_v2, print_table
from render_city import load_city, draw_2d, draw_3d, COLORS, LABELS

OUT = "/mnt/user-data/outputs"
TMP = "/tmp/city_v2"
BIN = "./gen_zoning_demo"
os.makedirs(OUT, exist_ok=True)
os.makedirs(TMP, exist_ok=True)

# ── Archetypes ─────────────────────────────────────────────────────────────
# (name, caption, x_unit[13], corridor_angle_deg, park_corridor_angle_deg)
#
#  compact_core  — biz pure radial (share=1), tight; park pure central radial.
#  boulevard     — biz pure corridor (share=0,cor_bip=1); park pure corridor,
#                  crossing it (default biz 0° / park 90°).
#  polycentric   — biz pure bipeak (share=0,cor_bip=0 → 4 nodes); park central.
#  mixed_use     — biz_scatter=1 & park_scatter=1: field only seeds, both dispersed.
#  garden_city   — biz radial; park slight-edge corridor greenway forced E-W (0°).
#  linear_city   — biz corridor forced N-S (90°); park corridor E-W (0°).

ARCHETYPES = [
    ("compact_core",
     "Compact core\nfine blocks · tall CBD · central green mass",
     [0.15, 0.15, 0.90, 0.85, 0.10, 0.25, 0.15,
      1.00, 0.00, 0.0, 0.00,          # biz: pure radial, clustered
      1.00, 0.00, 0.5, 0.20,          # park: pure central radial, near-contiguous
      0.0]),

    ("boulevard",
     "Boulevard city\nE-W biz spine · N-S green corridor",
     [0.45, 0.45, 0.55, 0.40, 0.28, 0.20, 0.45,
      0.00, 1.00, 0.0, 0.30,          # biz: pure corridor E-W
      0.50, 1.00, 0.5, 0.25,          # park: pure corridor N-S (⟂)
      0.0]),

    ("polycentric",
     "Polycentric\nfour biz nodes · central green heart",
     [0.30, 0.30, 0.65, 0.50, 0.18, 0.35, 0.30,
      0.00, 0.00, 0.0, 0.50,          # biz: pure bipeak (4 nodes), half-scattered
      1.00, 0.00, 0.5, 0.20,          # park: central contiguous green
      0.0]),

    ("mixed_use",
     "Mixed-use fabric\nscattered shops · scattered pocket parks",
     [0.10, 0.10, 0.45, 0.35, 0.22, 0.45, 0.15,
      1.00, 0.00, 0.0, 1.00,          # biz: radial seed, max scatter
      0.50, 0.00, 0.5, 1.00,          # park: neutral, max scatter
      0.0]),

    ("garden_city",
     "Garden city\nhigh parks · monocentric biz · E-W greenway",
     [0.35, 0.35, 0.55, 0.45, 0.42, 0.15, 0.40,
      1.00, 0.00, 0.0, 0.10,          # biz: radial
      0.35, 1.00, 0.0, 0.30,          # park: slight-edge + E-W corridor greenway
      0.0]),

    ("linear_city",
     "Linear city\nN-S business spine · E-W green band",
     [0.30, 0.30, 0.55, 0.40, 0.25, 0.20, 0.30,
      0.00, 1.00, 0.5, 0.00,          # biz: pure corridor N-S
      0.50, 1.00, 0.0, 0.20,          # park: pure corridor E-W
      0.0]),
]


def build(name, x_unit):
    phys = to_physical_v2(x_unit)
    args = [
        f"{phys['block_w']:.5g}", f"{phys['block_d']:.5g}",
        f"{phys['cbd_peak']:.5g}", f"{phys['cbd_decay']:.5g}",
        "0", f"{phys['park_fraction']:.5g}",
        f"{phys['roughness']:.5g}", f"{phys['street_width']:.5g}",
        # business potential
        f"{phys['phi_radial']:.5g}", f"{phys['phi_grad_x']:.5g}", f"{phys['phi_grad_y']:.5g}",
        f"{phys['phi_bipeak_x']:.5g}", f"{phys['phi_bipeak_y']:.5g}",
        f"{phys['phi_corridor']:.5g}", f"{phys['corridor_angle_deg']:.5g}", f"{phys['biz_scatter']:.5g}",
        # park potential
        f"{phys['park_centrality']:.5g}", f"{phys['park_w_grad_x']:.5g}", f"{phys['park_w_grad_y']:.5g}",
        f"{phys['park_w_bipeak_x']:.5g}", f"{phys['park_w_bipeak_y']:.5g}",
        f"{phys['park_w_corridor']:.5g}", f"{phys['park_corridor_angle_deg']:.5g}", f"{phys['park_scatter']:.5g}",
    ]
    out = os.path.join(TMP, f"{name}.txt")
    res = subprocess.run([BIN, *args, out], capture_output=True, text=True, check=True)
    print(res.stdout.strip())
    return out, phys


def panel(items, outfile, suptitle):
    n = len(items)
    fig = plt.figure(figsize=(5.6 * n, 6.0))
    for col, (name, caption, x_unit, *_) in enumerate(items):
        bl, m = load_city(os.path.join(TMP, f"{name}.txt"))
        ax2 = fig.add_subplot(1, n, col + 1)
        draw_2d(ax2, bl, m, cell=m["cell"])
        ax2.set_title(f"{caption}\n{m['N']} blk · max {m['maxH']:.0f} m · {m['pop']:.0f} pop",
                      fontsize=9.5, fontweight="bold", pad=6)
        ax2.set_xticks([]); ax2.set_yticks([])
    leg = [Patch(facecolor=COLORS[i], edgecolor="#444", label=LABELS[i]) for i in range(4)]
    fig.legend(handles=leg, loc="lower center", ncol=4, fontsize=10, frameon=True)
    fig.suptitle(suptitle, fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0.06, 1, 0.95])
    plt.savefig(outfile, dpi=140, bbox_inches="tight")
    plt.close()
    print("saved", outfile)


def main():
    rows = []
    for name, caption, x_unit in ARCHETYPES:
        _, phys = build(name, x_unit)
        rows.append((name[:12], phys))

    print("\n# Physical parameters per archetype:")
    print_table(rows, params=[
        "block_w", "block_d", "cbd_peak", "cbd_decay",
        "park_fraction", "roughness", "street_width",
        "phi_radial", "phi_corridor", "phi_bipeak", "biz_scatter",
        "park_centrality", "park_w_corridor", "park_w_bipeak_x", "park_scatter",
        "corridor_angle_deg", "park_corridor_angle_deg", "wind_angle_deg",
    ])

    panel(ARCHETYPES[:3], os.path.join(OUT, "city_v2_A.png"),
          "V2 city suite — identifiable space  (compact · boulevard · polycentric)")
    panel(ARCHETYPES[3:], os.path.join(OUT, "city_v2_B.png"),
          "V2 city suite — identifiable space  (mixed-use · garden · linear)")


if __name__ == "__main__":
    main()
