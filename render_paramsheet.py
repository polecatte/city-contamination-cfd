"""Per-parameter contact sheet. Each panel varies ONE knob against a common
baseline. Zoning/geometry knobs are usage-coloured (blue=business, dark gold=
hi-dens res, light gold=lo-dens res houses at 25% lot coverage, green=park);
height knobs (cbd_peak, cbd_decay, roughness) are height-coloured (viridis).
Gray = the lot/parcel; the coloured building sits inside the 4 m setback."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mc
from matplotlib.patches import Rectangle, Patch
from matplotlib.cm import ScalarMappable
from render_city import load_city, COLORS, LABELS
CELL=4.0

def draw(ax, bl, m, mode):
    Sx,Sy,mxH = m['Sx'], m['Sy'], max(m['maxH'],1)
    ax.set_xlim(0,Sx); ax.set_ylim(0,Sy); ax.set_aspect('equal'); ax.set_facecolor('#555')
    ax.set_xticks([]); ax.set_yticks([])
    for b in bl:  # lot/parcel (gray)
        ax.add_patch(Rectangle((b['bx0']*CELL,b['by0']*CELL),(b['bx1']-b['bx0'])*CELL,
                     (b['by1']-b['by0'])*CELL, facecolor='#b5b0a5', zorder=2, linewidth=0))
    for b in bl:  # building — USAGE hue, brightness ∝ height (so housing always visible AND height reads)
        x0,y0=b['x0']*CELL,b['y0']*CELL; w=(b['x1']-b['x0'])*CELL; h=(b['y1']-b['y0'])*CELL
        if w<=0 or h<=0: continue
        u,hc=b['usage'],b['hc']
        if u==0:
            fc='#4a8a3a'
        else:
            rgb=mc.to_rgb(COLORS[u]); br=0.38+0.62*(hc*CELL/mxH); fc=tuple(min(1,c*br) for c in rgb)
        ax.add_patch(Rectangle((x0,y0),w,h,facecolor=fc,edgecolor='#444',linewidth=0.1,zorder=3))

cases=[l.strip().split('|') for l in open("/tmp/ps_manifest.txt") if l.strip()]
n=len(cases); cols=5; rows=(n+cols-1)//cols
fig=plt.figure(figsize=(4*cols, 4.4*rows))
for i,(f,name,mode) in enumerate(cases):
    bl,m=load_city(f)
    ax=fig.add_subplot(rows,cols,i+1); draw(ax,bl,m,mode)
    ax.set_title(name, fontsize=10, fontweight="bold")

# legends
usage_leg=[Patch(facecolor=COLORS[i],edgecolor='#444',label=LABELS[i]) for i in range(4)]
usage_leg.append(Patch(facecolor='#b5b0a5',edgecolor='#444',label='Lot (setback margin)'))
fig.legend(handles=usage_leg, loc='lower center', ncol=5, fontsize=10, frameon=False,
           bbox_to_anchor=(0.5,-0.02))
fig.suptitle("Effect of each parameter (baseline = 560 m, pop 30k) — usage-coloured, "
             "brightness ∝ height; population 30000 & workers 14100 verified in every panel",
             fontsize=13, fontweight="bold")
plt.tight_layout(rect=[0,0.03,1,0.97])
out="/mnt/user-data/outputs/examples_parameters.png"
plt.savefig(out,dpi=135,bbox_inches="tight"); plt.close()
print("saved",out)
