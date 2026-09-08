"""Render the roughness series: top row 2D plans, bottom row 3D massing, across
roughness = 0.0 / 0.3 / 0.6 / 0.8. The 3D row makes the exaggerated height
heterogeneity (≈ CoV σ_H/H̄) obvious. Population is fixed at 20000 in every panel."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from render_city import load_city, draw_2d, draw_3d

CELL = 4.0
cases = [("/tmp/rough_0.txt","roughness 0.0  (CoV≈0.09, ~uniform)"),
         ("/tmp/rough_3.txt","roughness 0.3  (CoV≈0.32)"),
         ("/tmp/rough_6.txt","roughness 0.6  (CoV≈0.62)"),
         ("/tmp/rough_8.txt","roughness 0.8  (CoV≈0.85, towers among low-rise)")]

fig = plt.figure(figsize=(20, 9))
for i,(f,title) in enumerate(cases):
    bl,m = load_city(f)
    ax2 = fig.add_subplot(2,4,i+1)
    draw_2d(ax2, bl, m, cell=CELL); ax2.set_title(title, fontsize=11, fontweight="bold")
    ax3 = fig.add_subplot(2,4,i+5, projection="3d")
    draw_3d(ax3, bl, m, cell=CELL, z_exag=1.2)
    ax3.set_title("massing", fontsize=10)

fig.suptitle("Height-heterogeneity (roughness) series — log-normal scatter, "
             "population fixed at 20,000  ·  patchiness removed",
             fontsize=14, fontweight="bold")
plt.tight_layout(rect=[0,0,1,0.96])
out="/mnt/user-data/outputs/examples_roughness.png"
plt.savefig(out, dpi=140, bbox_inches="tight"); plt.close()
print("saved", out)
