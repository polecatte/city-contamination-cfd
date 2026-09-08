import numpy as np, matplotlib.pyplot as plt, struct
# (a) NHAPS split + pathway mapping
labels=["residence\n(indoor)","other indoor\n(work/commercial)","in-vehicle","outdoors"]
vals=[69,18,5.5,7.6]
pathway=["indoor (infiltration)","indoor (infiltration)","STREET (full outdoor)","park + STREET"]
colors=["#2a5a9a","#3a7 aa0".replace(" ",""),"#d45050","#4a8a3a"]
colors=["#2a5a9a","#3a7aa0","#d45050","#4a8a3a"]
fig,ax=plt.subplots(1,2,figsize=(13,5.2))
b=ax[0].barh(labels,vals,color=colors)
for i,(v,pw) in enumerate(zip(vals,pathway)):
    ax[0].text(v+0.5,i,f"{v}%  ->  {pw}",va="center",fontsize=9)
ax[0].set_xlim(0,100); ax[0].set_xlabel("% of 24-h day (population average)")
ax[0].set_title("(a) Microenvironment time budget\nNHAPS — Klepeis et al. 2001 (n=9386, EPA)")
ax[0].invert_yaxis()
ax[0].annotate("24-h average — NO time-of-release assumption\n(event time unknown -> weight all hours equally)",
               (50,3.4),fontsize=8,color="#555",ha="center")

# (b) street pedestrian-density field
with open("street_field.bin","rb") as f:
    nx,ny=struct.unpack("2i",f.read(8))
    w=np.frombuffer(f.read(nx*ny*4),dtype=np.float32).reshape(ny,nx)
m=np.ma.masked_less_equal(w,0)
im=ax[1].imshow(m,origin="lower",cmap="magma")
plt.colorbar(im,ax=ax[1],shrink=0.8,label="street occupants / cell")
ax[1].set_title("(b) Street inhabitance on the road network\n(pedestrians weighted by adjacent building activity)")
ax[1].set_xlabel("x (cells)"); ax[1].set_ylabel("y (cells)")
fig.suptitle("Effective street inhabitance + sourced occupancy split",fontsize=13,fontweight="bold")
plt.tight_layout(rect=[0,0,1,0.95])
plt.savefig("/mnt/user-data/outputs/occupancy_street.png",dpi=150,bbox_inches="tight")
print("saved occupancy_street.png")
