"""Larger example cities across patchiness (business spread) with the RES_LOW ring.
Top row: 2D usage plan (blue=business, dark gold=hi-dens res, light gold=lo-dens res
= houses at 25% lot coverage, green=park). Bottom: 3D massing."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from render_city import load_city, draw_2d, draw_3d
CELL=4.0
cases=[("/tmp/big_p0.txt","patchiness 0.0  (concentric business core)"),
       ("/tmp/big_p5.txt","patchiness 0.5  (business spreading)"),
       ("/tmp/big_p10.txt","patchiness 1.0  (business in coherent off-centre patches)")]
fig=plt.figure(figsize=(19,11))
for i,(f,title) in enumerate(cases):
    bl,m=load_city(f)
    ax2=fig.add_subplot(2,3,i+1); draw_2d(ax2,bl,m,cell=CELL)
    ax2.set_title(title,fontsize=11,fontweight="bold")
    ax3=fig.add_subplot(2,3,i+4,projection="3d"); draw_3d(ax3,bl,m,cell=CELL,z_exag=1.2)
    ax3.set_title("massing",fontsize=10)
fig.suptitle("Larger cities (720 m, pop 40k) — patchiness spreads the business district; "
             "low-density residential ring (light gold, 25% lot coverage) at the periphery",
             fontsize=13,fontweight="bold")
plt.tight_layout(rect=[0,0,1,0.96])
out="/mnt/user-data/outputs/examples_larger.png"
plt.savefig(out,dpi=140,bbox_inches="tight"); plt.close()
print("saved",out)
