"""gen_city_gallery.py — a 2D-only gallery of v2 city examples.

Twelve designs spanning the business AND park potential fields plus morphology.
Plan views only (no 3D massing). Backend: gen_zoning_demo (city_zoning.h).

Each city is (name, title, morph[8], biz[8], park[8]):
  morph: block_w block_d cbd_peak cbd_decay patchiness park_frac roughness street_w
  biz:   w_rad w_gx w_gy w_bpx w_bpy w_cor cor_ang° biz_scatter
  park:  park_cent park_gx park_gy park_bpx park_bpy park_cor park_cor_ang° park_scatter
"""
import os, subprocess
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from render_city import load_city, draw_2d, COLORS, LABELS

OUT = "/mnt/user-data/outputs"
TMP = "/tmp/city_gallery"
BIN = "./gen_zoning_demo"
os.makedirs(OUT, exist_ok=True)
os.makedirs(TMP, exist_ok=True)

STD  = "40 32 55 6e-6 0 0.15 0.20 20"   # standard blocks (~110)
FINE = "22 20 55 6e-6 0 0.15 0.25 12"   # fine grain (~300)
COAR = "50 45 45 5e-6 0 0.15 0.15 26"   # superblocks (~70)

def morph(base, park_frac=None, cbd_peak=None, cbd_decay=None):
    v = base.split()
    if park_frac is not None: v[5] = str(park_frac)
    if cbd_peak  is not None: v[2] = str(cbd_peak)
    if cbd_decay is not None: v[3] = str(cbd_decay)
    return " ".join(v)

CITIES = [
    ("monocentric", "Monocentric core\nradial biz · central green ring",
     STD, "1 0 0 0 0 0 0 0", "0.7 0 0 0 0 0 0 0.2"),

    ("twin_cores", "Twin cores\nbiz twin-peak(x) · green between",
     STD, "0 0 0 1 0 0 0 0", "0.6 0 0 0 0 0 0 0.3"),

    ("sector_belt", "Sector + greenbelt\nbiz pulled east · west green belt",
     STD, "0.3 1 0 0 0 0 0 0.2", "0 0 0 0 0 0 0 1"),

    ("garden_grid", "Garden grid\nscattered shops · 40% dispersed parks",
     morph(FINE, park_frac=0.40), "0.05 0 0 0 0 0 0 1", "0.5 0 0 0 0 0 0 1"),

    ("boulevard", "Boulevard\nE-W biz spine · N-S green corridor",
     STD, "0 0 0 0 0 1 0 0.3", "0.5 0 0 0 0 1 90 0.2"),

    ("linear_spine", "Linear spine\nN-S biz spine · E-W green band",
     STD, "0 0 0 0 0 1 90 0", "0.5 0 0 0 0 1 0 0.2"),

    ("four_quadrant", "Four quadrant\nbiz nodes ×4 · central green heart",
     STD, "0 0 0 1 1 0 0 0.3", "0.7 0 0 0 0 0 0 0.2"),

    ("downtown_belt", "Downtown + belt\ntall tight CBD · outer greenbelt",
     morph(STD, cbd_peak=80, cbd_decay="2.5e-5"), "1 0 0 0 0 0 0 0", "0 0 0 0 0 0 0 1"),

    ("mixed_use", "Dispersed mixed-use\nfine grain · shops & parks scattered",
     FINE, "0.05 0 0 0 0 0 0 1", "0.5 0 0 0 0 0 0 1"),

    ("diagonal_green", "Diagonal greenway\nradial biz · 45° green corridor",
     STD, "1 0 0 0 0 0 0 0", "0.5 0 0 0 0 1 45 0.15"),

    ("polynuclear", "Polynuclear\ntwin-peak biz half-scattered · scattered parks",
     STD, "0 0 0 1 0 0 0 0.5", "0.4 0 0 0 0 0 0 1"),

    ("superblock", "Superblock corridors\ncoarse grid · crossed biz/green corridors",
     COAR, "0 0 0 0 0 1 30 0.1", "0.5 0 0 0 0 1 120 0.2"),
]


def build_all():
    for name, title, m, biz, park in CITIES:
        outf = os.path.join(TMP, f"{name}.txt")
        cmd  = f"{BIN} {m} {biz} {park} {outf}"
        res  = subprocess.run(cmd.split(), capture_output=True, text=True)
        if res.returncode != 0:
            print(f"FAILED {name}:", res.stderr.strip())
        else:
            print(res.stdout.strip())


def gallery(outfile, ncol=4):
    n = len(CITIES)
    nrow = (n + ncol - 1) // ncol
    fig = plt.figure(figsize=(4.6 * ncol, 5.6 * nrow))
    for i, (name, title, *_ ) in enumerate(CITIES):
        bl, m = load_city(os.path.join(TMP, f"{name}.txt"))
        ax = fig.add_subplot(nrow, ncol, i + 1)
        draw_2d(ax, bl, m, cell=m["cell"])
        ax.set_title(f"{title}\n{m['N']} blk · max {m['maxH']:.0f} m",
                     fontsize=9.5, fontweight="bold", pad=6)
        ax.set_xticks([]); ax.set_yticks([])

    leg = [Patch(facecolor=COLORS[i], edgecolor="#444", label=LABELS[i]) for i in range(4)]
    fig.legend(handles=leg, loc="lower center", ncol=4, fontsize=11, frameon=True)
    fig.suptitle("V2 city gallery — designed business & parks  (plan view · pop fixed 20,000)",
                 fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0.03, 1, 0.965], h_pad=4.5, w_pad=1.5)
    plt.savefig(outfile, dpi=140, bbox_inches="tight")
    plt.close()
    print("saved", outfile)


if __name__ == "__main__":
    build_all()
    gallery(os.path.join(OUT, "city_gallery.png"))
