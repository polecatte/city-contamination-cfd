"""Render the generated example cities into 2D-plan + 3D-massing comparison
panels, reusing render_city's draw functions (with the correct cell size)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from render_city import load_city, draw_2d, draw_3d, COLORS, LABELS

def panel(files, labels, out, suptitle):
    n = len(files)
    fig = plt.figure(figsize=(5.2*n, 9.6))
    for c,(f,lab) in enumerate(zip(files, labels)):
        bl, m = load_city(f)
        cell = m['cell']                      # real cell size (4 m), not the 2.0 default
        ax2 = fig.add_subplot(2, n, c+1)
        draw_2d(ax2, bl, m, cell=cell)
        ax2.set_title(f"{lab}\n{m['N']} blk · max {m['maxH']:.0f}m · {m['pop']:.0f} pop",
                      fontsize=10, fontweight='bold')
        ax2.set_xticks([]); ax2.set_yticks([])
        ax3 = fig.add_subplot(2, n, n+c+1, projection='3d')
        draw_3d(ax3, bl, m, cell=cell)
        ax3.set_title(f"max {m['maxH']:.0f} m ({int(m['maxH']/3)} floors)", fontsize=9)
    leg = [Patch(facecolor=COLORS[i], edgecolor='#444', label=LABELS[i]) for i in range(4)]
    fig.legend(handles=leg, loc='lower center', ncol=5, fontsize=10, frameon=True)
    fig.suptitle(suptitle, fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    plt.savefig(out, dpi=140, bbox_inches='tight'); plt.close()
    print("saved", out)

# Group 1 — street width & grid anisotropy
panel(["/tmp/ex_A.txt","/tmp/ex_B.txt","/tmp/ex_C.txt","/tmp/ex_D.txt"],
      ["A baseline (20 m)","B narrow (8 m)","C wide (40 m)","D anisotropic (40/8 m)"],
      "/mnt/user-data/outputs/examples_streets.png",
      "Street-width & grid parameters  (population fixed at 20,000)")

# Group 2 — parks & morphology height field
panel(["/tmp/ex_A.txt","/tmp/ex_E.txt","/tmp/ex_F.txt"],
      ["A baseline (park 0.10)","E high parks (0.30)","F tall CBD (peak 120 m)"],
      "/mnt/user-data/outputs/examples_parks_height.png",
      "Park-fraction & morphology-height parameters  (population fixed at 20,000)")
