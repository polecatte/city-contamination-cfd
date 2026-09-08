"""Draw the ACTUAL simulation-domain borders around each city: plan (x-y) and
elevation (x-z), with COST-732 clearances annotated and the *required* minima
(5H up/lateral/top, 15H downstream) shown dashed so the margin is visible."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import matplotlib.colors as mc
from render_city import load_city, COLORS

def city_bbox(bl, cell):
    xs0=min(b['bx0'] for b in bl); ys0=min(b['by0'] for b in bl)
    xs1=max(b['bx1'] for b in bl); ys1=max(b['by1'] for b in bl)
    return xs0*cell, ys0*cell, xs1*cell, ys1*cell

def skyline_x(bl, nx, cell):
    h=[0.0]*nx
    for b in bl:
        if b['usage']==0: continue
        hh=b['hc']*cell
        for x in range(max(0,b['x0']),min(nx,b['x1'])): h[x]=max(h[x],hh)
    return h

def dim(ax, x0, x1, y, txt, color='#222'):
    ax.annotate('', (x1,y),(x0,y), arrowprops=dict(arrowstyle='<->',color=color,lw=1.3))
    ax.text((x0+x1)/2, y, txt, ha='center', va='bottom', fontsize=8.5, color=color,
            bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.8))

def draw(fileA, fileF):
    fig, axes = plt.subplots(2, 2, figsize=(15, 13))
    for col,(f,name) in enumerate([(fileA,"A baseline"),(fileF,"F tall CBD")]):
        bl,m = load_city(f); cell=m['cell']; Sx,Sy=m['Sx'],m['Sy']
        nx=int(round(Sx/cell))
        H=max((b['hc']*cell for b in bl if b['usage']!=0), default=1)
        Sz=6*H
        cx0,cy0,cx1,cy1 = city_bbox(bl,cell)
        up,down,latn,latp = cx0, Sx-cx1, cy0, Sy-cy1

        # ---------- Plan (x-y) ----------
        ax=axes[0,col]; ax.set_facecolor('#eef0f2')
        ax.add_patch(Rectangle((0,0),Sx,Sy,fill=False,ec='#111',lw=2))           # domain border
        for b in bl:                                                              # actual buildings
            w=(b['x1']-b['x0'])*cell; h=(b['y1']-b['y0'])*cell
            if w<=0 or h<=0: continue
            ax.add_patch(Rectangle((b['x0']*cell,b['y0']*cell),w,h,
                         facecolor=COLORS[b['usage']],ec='none'))
        ax.add_patch(Rectangle((cx0,cy0),cx1-cx0,cy1-cy0,fill=False,ec='#d4500a',
                     lw=1.6,ls='--'))                                             # city footprint
        # required minima (dashed): 5H up/lateral, 15H downstream
        ax.add_patch(Rectangle((5*H,5*H),Sx-5*H-15*H,Sy-2*5*H,fill=False,
                     ec='#1565c0',lw=1.1,ls=':'))                                 # COST-732 min box
        # wind
        ax.add_patch(FancyArrowPatch((up*0.15,Sy*0.5),(up*0.7,Sy*0.5),
                     arrowstyle='-|>',mutation_scale=22,color='#1565c0',lw=2.5))
        ax.text(up*0.42,Sy*0.5+Sy*0.03,'wind +x',color='#1565c0',fontsize=10,ha='center',fontweight='bold')
        dim(ax,0,cx0,Sy*0.5,f"{up:.0f} m = {up/H:.1f}H\n(≥5H={5*H:.0f}) ✓")
        dim(ax,cx1,Sx,Sy*0.5,f"{down:.0f} m = {down/H:.1f}H\n(≥15H={15*H:.0f}) ✓")
        dim(ax,cx0+ (cx1-cx0)*0.5-1,cx0+(cx1-cx0)*0.5+1,0, "")  # noop spacer
        ax.annotate('', (cx0+(cx1-cx0)*0.5,0),(cx0+(cx1-cx0)*0.5,cy0),
                    arrowprops=dict(arrowstyle='<->',color='#222',lw=1.3))
        ax.text(cx0+(cx1-cx0)*0.5, cy0*0.5, f"lateral {latn:.0f} m = {latn/H:.1f}H\n(≥5H={5*H:.0f}) ✓",
                ha='center',va='center',fontsize=8.5,
                bbox=dict(boxstyle='round,pad=0.15',fc='white',ec='none',alpha=0.8))
        ax.set_xlim(-Sx*0.02,Sx*1.02); ax.set_ylim(-Sy*0.02,Sy*1.02); ax.set_aspect('equal')
        ax.set_title(f"{name} — plan (x–y)\ncity {cx1-cx0:.0f} m in {Sx:.0f}×{Sy:.0f} m domain",
                     fontsize=11,fontweight='bold')
        ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')

        # ---------- Elevation (x-z) ----------
        ax=axes[1,col]; ax.set_facecolor('#eef0f2')
        ax.add_patch(Rectangle((0,0),Sx,Sz,fill=False,ec='#111',lw=2))           # domain border
        ax.axhline(0,color='#7a6a55',lw=2)                                       # ground
        sky=skyline_x(bl,nx,cell)
        ax.fill_between([i*cell for i in range(nx)], 0, sky, color='#8a8170', step='mid')
        ax.axhline(H,color='#444',ls='--',lw=0.8)
        # top clearance
        ax.annotate('', (Sx*0.5,H),(Sx*0.5,Sz),arrowprops=dict(arrowstyle='<->',color='#222',lw=1.3))
        ax.text(Sx*0.5,(H+Sz)/2,f"top {Sz-H:.0f} m\n= {(Sz-H)/H:.1f}H (≥5H) ✓",ha='center',va='center',fontsize=8.5,
                bbox=dict(boxstyle='round,pad=0.15',fc='white',ec='none',alpha=0.85))
        dim(ax,0,cx0,H*0.5 if H*0.5<Sz else Sz*0.3,f"{up:.0f} m={up/H:.1f}H\n(≥5H) ✓")
        dim(ax,cx1,Sx,H*0.5 if H*0.5<Sz else Sz*0.3,f"{down:.0f} m={down/H:.1f}H\n(≥15H) ✓")
        ax.text(cx0+(cx1-cx0)*0.5,H+Sz*0.01,f"tallest H={H:.0f} m",ha='center',va='bottom',fontsize=8.5)
        ax.set_xlim(-Sx*0.02,Sx*1.02); ax.set_ylim(-Sz*0.04,Sz*1.08)
        ax.set_title(f"{name} — elevation (x–z), domain height {Sz:.0f} m = 6H",
                     fontsize=11,fontweight='bold')
        ax.set_xlabel('x (m), wind →'); ax.set_ylabel('z (m)')

    fig.suptitle("Actual COST-732 simulation borders  (solid=domain, dashed orange=city, dotted blue=required 5H/15H minima)",
                 fontsize=12.5,fontweight='bold')
    plt.tight_layout(rect=[0,0,1,0.97])
    out="/mnt/user-data/outputs/domain_borders.png"
    plt.savefig(out,dpi=140,bbox_inches='tight'); plt.close(); print("saved",out)

draw("/tmp/cost_A.txt","/tmp/cost_F.txt")
