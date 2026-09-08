import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from render_city import load_city, draw_2d
CELL=4.0
cases=[("/tmp/park_e.txt","park_centrality 0.1  (edge greenbelt)"),
       ("/tmp/park_d.txt","park_centrality 0.5  (evenly distributed)"),
       ("/tmp/park_c.txt","park_centrality 0.9  (central park)")]
fig,axes=plt.subplots(1,3,figsize=(16,5.6))
for ax,(f,t) in zip(axes,cases):
    bl,m=load_city(f); draw_2d(ax,bl,m,cell=CELL); ax.set_title(t,fontsize=12,fontweight="bold")
fig.suptitle("park_centrality — parks now disperse symmetrically (no more one-sided walls); "
             "0=edge ring · 0.5=distributed · 1=central",fontsize=13,fontweight="bold")
plt.tight_layout(rect=[0,0,1,0.95])
plt.savefig("/mnt/user-data/outputs/examples_park_centrality.png",dpi=140,bbox_inches="tight")
print("saved examples_park_centrality.png")
