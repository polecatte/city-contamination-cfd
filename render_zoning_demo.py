"""render_zoning_demo.py — render the mixed-potential zoning demo.

Produces six comparison panels (2D plan + 3D massing):

  biz_A.png    : Monocentric  |  Sector (gradient)  |  Twin-core (bipeak-x)
  biz_B.png    : E-W corridor |  N-S corridor        |  Offset core (blend)
  park_A.png   : Central mass |  Edge greenbelt      |  Dispersed pockets
  park_B.png   : E-W greenway |  N-S greenway         |  Four green bands
  scatter_A.png: Monocentric Φ, biz_scatter ∈ {0, 0.5, 1}
  scatter_B.png: Flat Φ (fine blocks), biz_scatter ∈ {0, 0.5, 1}

Parks now use the SAME designed potential field as business (radial via
park_centrality, gradient, twin-peak, corridor) plus a park_scatter knob — the
old periodic "stripe" field is gone.
"""
import os, subprocess
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from render_city import load_city, draw_2d, draw_3d, COLORS, LABELS

OUT    = "/mnt/user-data/outputs"
TMP    = "/tmp/zdemo"
BIN    = "./gen_zoning_demo"
os.makedirs(OUT, exist_ok=True)
os.makedirs(TMP, exist_ok=True)

# ── Base morphologies ──────────────────────────────────────────────────────
# block_w block_d cbd_peak cbd_decay patchiness park_frac roughness street_w
MORPH      = "40 32 60 8e-6 0 0.15 0.2 20"   # standard blocks (~110 blocks)
MORPH_FINE = "20 20 60 8e-6 0 0.15 0.2 10"   # fine blocks  (~400 blocks) — shows individual buildings

# Neutral default park field: park_centrality=0.5, no structure, scatter=1
# (even-dispersed pocket parks — the familiar baseline).
DEF_PARK = "0.5 0 0 0 0 0 0 1"

# ── Archetype table ──────────────────────────────────────────────────────
# Each entry: (filename_stem, short_title, detailed_caption,
#              params_string, morph_override_or_None)
#
# params_string format (16 values):
#   BUSINESS: w_rad w_gx w_gy w_bpx w_bpy w_cor cor_ang° biz_scatter
#   PARK:     park_cent park_gx park_gy park_bpx park_bpy park_cor park_cor_ang° park_scatter

ARCHETYPES = [
    # ── Business potential modes (biz_scatter=0; default parks) ─────────
    ("monocentric",
     "Monocentric\n(w_radial=1)",
     "baseline — radial Φ,\nidentical to original builder",
     f"1 0 0 0 0 0 0 0   {DEF_PARK}",  None),

    ("sector",
     "Sector\n(w_grad_x=1)",
     "Φ = −x gradient → business\npulled to right (east) half",
     f"0 1 0 0 0 0 0 0   {DEF_PARK}",  None),

    ("twin_core",
     "Twin-core\n(w_bipeak_x=1)",
     "Φ = −cos(2πx/W) → two districts\nat x = cx ± W/4",
     f"0 0 0 1 0 0 0 0   {DEF_PARK}",  None),

    ("corridor_ew",
     "E-W corridor\n(w_corridor=1, θ=0°)",
     "Φ = |y−cy|/R → linear spine\nalong city's east-west axis",
     f"0 0 0 0 0 1 0 0   {DEF_PARK}",  None),

    ("corridor_ns",
     "N-S corridor\n(w_corridor=1, θ=90°)",
     "Φ = |x−cx|/R → linear spine\nalong north-south axis",
     f"0 0 0 0 0 1 90 0   {DEF_PARK}",  None),

    ("offset_core",
     "Offset core\n(0.5·radial + 0.5·grad_x)",
     "blend → monocentric core\nshifted toward right half",
     f"0.5 0.5 0 0 0 0 0 0   {DEF_PARK}",  None),

    # ── Park modes (business fixed monocentric; vary the PARK field) ─────
    ("park_central",
     "Central park mass\n(park_cent=1, scatter=0)",
     "Φ_park radial → one contiguous\ngreen mass at the city center",
     "1 0 0 0 0 0 0 0   1 0 0 0 0 0 0 0",  None),

    ("park_edge",
     "Edge greenbelt\n(park_cent=0, scatter=1)",
     "Φ_park radial inverted → parks\npushed outward, dispersed as a belt",
     "1 0 0 0 0 0 0 0   0 0 0 0 0 0 0 1",  None),

    ("park_dispersed",
     "Dispersed pockets\n(neutral, scatter=1)",
     "flat Φ_park, pure maximin →\neven pocket parks, no bias",
     "1 0 0 0 0 0 0 0   0.5 0 0 0 0 0 0 1",  None),

    ("park_greenway_ew",
     "E-W greenway\n(park_corridor, θ=0°)",
     "Φ_park corridor → a single\nE-W green spine (near-contiguous)",
     "1 0 0 0 0 0 0 0   0.5 0 0 0 0 1 0 0.2",  None),

    ("park_greenway_ns",
     "N-S greenway\n(park_corridor, θ=90°)",
     "Φ_park corridor → a single\nN-S green spine (cross-wind)",
     "1 0 0 0 0 0 0 0   0.5 0 0 0 0 1 90 0.2",  None),

    ("park_twin",
     "Four green bands\n(park_bipeak_x&y)",
     "Φ_park twin-peak on both axes →\nfour quadrant green nodes",
     "1 0 0 0 0 0 0 0   0.5 0 0 1 1 0 0 0.2",  None),

    # ── Scatter axis — monocentric Φ, standard blocks (default parks) ────
    ("scatter_mono_0",
     "Cluster\n(biz_scatter=0)",
     "greedy rank-fill → all business\nblocks adjacent (tight CBD)",
     f"1 0 0 0 0 0 0 0   {DEF_PARK}",  None),

    ("scatter_mono_05",
     "Mixed\n(biz_scatter=0.5)",
     "blend → CBD core kept,\nremaining commercials spread out",
     f"1 0 0 0 0 0 0 0.5   {DEF_PARK}",  None),

    ("scatter_mono_1",
     "Scatter\n(biz_scatter=1.0)",
     "maximin dispersal → center-biased\nbut individually distributed",
     f"1 0 0 0 0 0 0 1   {DEF_PARK}",  None),

    # ── Scatter axis — flat Φ, fine blocks (individual building grain) ────
    ("scatter_flat_0",
     "Flat Φ · Cluster\n(biz_scatter=0)",
     "weak radial field, no scatter\n→ compact central block",
     f"0.05 0 0 0 0 0 0 0   {DEF_PARK}",  MORPH_FINE),

    ("scatter_flat_05",
     "Flat Φ · Mixed\n(biz_scatter=0.5)",
     "slight bias + moderate dispersal\n→ loose cluster, some satellites",
     f"0.05 0 0 0 0 0 0 0.5   {DEF_PARK}",  MORPH_FINE),

    ("scatter_flat_1",
     "Flat Φ · Scatter\n(biz_scatter=1.0)",
     "pure maximin → each commercial\nbuilding maximally isolated",
     f"0.05 0 0 0 0 0 0 1   {DEF_PARK}",  MORPH_FINE),
]


def build_all():
    for stem, title, caption, params, morph in ARCHETYPES:
        m    = morph if morph is not None else MORPH
        outf = os.path.join(TMP, f"{stem}.txt")
        cmd  = f"{BIN} {m} {params} {outf}"
        res  = subprocess.run(cmd.split(), capture_output=True, text=True)
        if res.returncode != 0:
            print(f"FAILED {stem}:", res.stderr)
        else:
            print(res.stdout.strip())


def panel(stems_titles, outfile, suptitle):
    """Render a 3-column panel: 2D plan (row 0) + 3D massing (row 1)."""
    n = len(stems_titles)
    fig = plt.figure(figsize=(5.6*n, 10.2))
    for col, (stem, title, caption, *_) in enumerate(stems_titles):
        f = os.path.join(TMP, f"{stem}.txt")
        bl, m = load_city(f)
        cell = m["cell"]

        ax2 = fig.add_subplot(2, n, col+1)
        draw_2d(ax2, bl, m, cell=cell)
        ax2.set_title(f"{title}\n{m['N']} blk · max {m['maxH']:.0f} m",
                      fontsize=10, fontweight="bold", pad=4)
        ax2.set_xlabel(caption, fontsize=8.5, labelpad=5, style="italic")
        ax2.set_xticks([]); ax2.set_yticks([])

        ax3 = fig.add_subplot(2, n, n+col+1, projection="3d")
        draw_3d(ax3, bl, m, cell=cell)
        ax3.set_title(f"max {m['maxH']:.0f} m  ({int(m['maxH']/3)} floors)",
                      fontsize=8.5)

    leg = [Patch(facecolor=COLORS[i], edgecolor="#444", label=LABELS[i]) for i in range(4)]
    fig.legend(handles=leg, loc="lower center", ncol=4, fontsize=10, frameon=True)
    fig.suptitle(suptitle, fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    plt.savefig(outfile, dpi=140, bbox_inches="tight")
    plt.close()
    print("saved", outfile)


def main():
    build_all()

    panel(ARCHETYPES[0:3],
          os.path.join(OUT, "zoning_biz_A.png"),
          "Business potential Φ — basis terms I: radial · sector · twin-core")

    panel(ARCHETYPES[3:6],
          os.path.join(OUT, "zoning_biz_B.png"),
          "Business potential Φ — basis terms II: E-W corridor · N-S corridor · offset blend")

    panel(ARCHETYPES[6:9],
          os.path.join(OUT, "zoning_park_A.png"),
          "Park potential Φ_park — centrality & scatter: central mass · edge greenbelt · dispersed pockets")

    panel(ARCHETYPES[9:12],
          os.path.join(OUT, "zoning_park_B.png"),
          "Park potential Φ_park — designed structure: E-W greenway · N-S greenway · four green bands")

    panel(ARCHETYPES[12:15],
          os.path.join(OUT, "scatter_A.png"),
          "Business scatter axis — monocentric Φ: cluster · mixed · scatter  (40 m blocks)")

    panel(ARCHETYPES[15:18],
          os.path.join(OUT, "scatter_B.png"),
          "Business scatter axis — flat Φ: cluster · mixed · scatter  (20 m fine blocks)")


if __name__ == "__main__":
    main()
