"""viz_airflow.py — paper-quality figures from the airflow validation outputs.
Reads the mean-flow slices + CSVs written by airflow_validation (run AFTER the
averaging fix, so these are true time-means). Produces av_fig_*.png.

  python3 viz_airflow.py

Untested against real data here (no GPU in the authoring env); structurally it
mirrors the binary layout dumped by dump_xz/dump_xy in airflow_validation.cpp.
"""
import numpy as np, csv, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def read_slice(path, nch):
    with open(path,"rb") as f:
        nx,ni = np.fromfile(f,dtype=np.int32,count=2)
        arrs=[np.fromfile(f,dtype=np.float32,count=nx*ni).reshape(ni,nx) for _ in range(nch)]
    return nx,ni,arrs

def meta(path):
    r=list(csv.reader(open(path))); k=r[0]; v=r[1]
    return {k[i]:(float(v[i]) if '.' in v[i] else int(v[i])) for i in range(len(k))}

# ── Fig 1: cube wake, vertical centreline slice (x–z) ───────────────────────
def fig_cube_xz():
    if not (os.path.exists("av_cube_xz.bin") and os.path.exists("av_cube_meta.csv")): return
    m=meta("av_cube_meta.csv"); c=m["cell"]; H=m["H"]; x0=m["x0"]; xlee=m["x_lee"]; Xr=m["Xr_over_H"]
    nx,nz,(ux,uz,sp,nut)=read_slice("av_cube_xz.bin",4)
    Uref=np.percentile(sp[int(0.8*nz):,:5],90) or 1.0          # freestream (upper-upstream)
    X=np.arange(nx)*c; Z=np.arange(nz)*c
    fig,ax=plt.subplots(figsize=(12,4.2))
    pc=ax.pcolormesh(X,Z,sp/Uref,cmap="viridis",shading="auto",vmin=0,vmax=1.3)
    # streamlines of the mean flow
    ax.streamplot(X,Z,ux,uz,color="white",density=1.3,linewidth=0.6,arrowsize=0.7)
    ax.add_patch(Rectangle((x0*c,0),H*c,H*c,facecolor="#333",edgecolor="k",zorder=5))
    xr_abs=(xlee+Xr*H)*c
    ax.axvline(xr_abs,color="#e8503a",ls="--",lw=1.5,zorder=6)
    ax.text(xr_abs+4,H*c*1.05,f"reattachment\n$X_r/H$={Xr:.2f}",color="#e8503a",fontsize=9)
    ax.set_xlim(X[0],X[-1]); ax.set_ylim(0,min(Z[-1],6*H*c))
    ax.set_xlabel("x (m), wind →"); ax.set_ylabel("z (m)")
    ax.set_title("Surface-mounted cube — mean-flow wake (vertical centreline slice)",fontweight="bold")
    fig.colorbar(pc,ax=ax,label="$|\\overline{u}|/U_{ref}$",shrink=0.8)
    plt.tight_layout(); plt.savefig("av_fig_cube_xz.png",dpi=150); plt.close()
    print("saved av_fig_cube_xz.png")

# ── Fig 2: cube wake, horizontal slice (x–y) at mid-height ──────────────────
def fig_cube_xy():
    if not (os.path.exists("av_cube_xy.bin") and os.path.exists("av_cube_meta.csv")): return
    m=meta("av_cube_meta.csv"); c=m["cell"]; H=m["H"]; x0=m["x0"]; y0=m["y0"]
    nx,ny,(ux,uy,sp)=read_slice("av_cube_xy.bin",3)
    Uref=np.percentile(sp[:,:5],90) or 1.0
    X=np.arange(nx)*c; Y=np.arange(ny)*c
    fig,ax=plt.subplots(figsize=(12,5))
    pc=ax.pcolormesh(X,Y,sp/Uref,cmap="viridis",shading="auto",vmin=0,vmax=1.3)
    ax.streamplot(X,Y,ux,uy,color="white",density=1.4,linewidth=0.6,arrowsize=0.7)
    ax.add_patch(Rectangle((x0*c,y0*c),H*c,H*c,facecolor="#333",edgecolor="k",zorder=5))
    ax.set_xlabel("x (m), wind →"); ax.set_ylabel("y (m)"); ax.set_aspect("equal")
    ax.set_title("Surface-mounted cube — mean-flow (horizontal slice, z=H/2)",fontweight="bold")
    fig.colorbar(pc,ax=ax,label="$|\\overline{u}|/U_{ref}$",shrink=0.8)
    plt.tight_layout(); plt.savefig("av_fig_cube_xy.png",dpi=150); plt.close()
    print("saved av_fig_cube_xy.png")

# ── Fig 3: ABL homogeneity — U(z) at 3 stations + log law ───────────────────
def fig_abl():
    if not os.path.exists("av_abl.csv"): return
    r=np.array([[float(v) for v in row] for row in list(csv.reader(open("av_abl.csv")))[1:]])
    z=r[:,0]*4.0; Ui,Um,Uo=r[:,1],r[:,2],r[:,3]                # z in cells→m (cell=4)
    kappa,ustar,z0=0.41,0.505,0.70
    zl=np.linspace(max(z0,z[1]),z.max(),200); Ulog=(ustar/kappa)*np.log((zl+z0)/z0)
    # scale log law into LU using the inlet profile's own u* mapping (match at mid-height)
    imid=len(z)//2; scale=(Ui[imid])/max(1e-9,(ustar/kappa)*np.log((z[imid]+z0)/z0))
    fig,ax=plt.subplots(figsize=(5.5,6))
    ax.plot(Ui,z,"o-",label="inlet",ms=3); ax.plot(Um,z,"s-",label="mid",ms=3)
    ax.plot(Uo,z,"^-",label="outlet",ms=3)
    ax.plot(Ulog*scale,zl,"k--",lw=1.5,label="log law (R&H 1993)")
    ax.set_xlabel("$\\overline{u}_x$ (LU)"); ax.set_ylabel("z (m)")
    ax.set_title("ABL horizontal homogeneity\n(overlapping curves = homogeneous)",fontweight="bold")
    ax.legend(); plt.tight_layout(); plt.savefig("av_fig_abl.png",dpi=150); plt.close()
    print("saved av_fig_abl.png")

# ── Fig 4: cube Cp vs Silsoe reference bands ────────────────────────────────
def fig_cp():
    if not os.path.exists("av_cp.csv"): return
    d={row[0]:float(row[1]) for row in list(csv.reader(open("av_cp.csv")))[1:] if len(row)>=2}
    faces=["windward","leeward","roof","side"]; vals=[d.get(f,np.nan) for f in faces]
    bands={"windward":(0.4,1.0),"leeward":(-0.6,0.0),"roof":(-1.0,-0.2),"side":(-1.0,-0.2)}
    fig,ax=plt.subplots(figsize=(7,4.5))
    for i,f in enumerate(faces):
        lo,hi=bands[f]; ax.add_patch(Rectangle((i-0.4,lo),0.8,hi-lo,facecolor="#bfe3bf",edgecolor="none",zorder=0))
    ax.bar(range(4),vals,width=0.5,color="#2a5a9a",zorder=3)
    ax.axhline(0,color="k",lw=0.8); ax.set_xticks(range(4)); ax.set_xticklabels(faces)
    ax.set_ylabel("$C_p$"); ax.set_title("Cube pressure coefficients vs Silsoe bands (green)",fontweight="bold")
    plt.tight_layout(); plt.savefig("av_fig_cp.png",dpi=150); plt.close()
    print("saved av_fig_cp.png")

# ── Fig 5: Blasius δ99(x) growth ────────────────────────────────────────────
def fig_blasius():
    if not os.path.exists("av_blasius.csv"): return
    r=np.array([[float(v) for v in row] for row in list(csv.reader(open("av_blasius.csv")))[1:]])
    x,d99=r[:,0]*4.0,r[:,1]*4.0
    fig,ax=plt.subplots(1,2,figsize=(11,4.2))
    ax[0].plot(x,d99,"o",ms=4)
    if len(x)>=2:
        A=np.polyfit(np.sqrt(x),d99,1); xf=np.linspace(x.min(),x.max(),100)
        ax[0].plot(xf,A[0]*np.sqrt(xf)+A[1],"k--",label="$\\delta_{99}\\propto\\sqrt{x}$ fit")
    ax[0].set_xlabel("x (m)"); ax[0].set_ylabel("$\\delta_{99}$ (m)"); ax[0].legend()
    ax[0].set_title("Blasius BL growth")
    ax[1].plot(x,r[:,4],"s-",ms=4); ax[1].axhline(2.59,color="#e8503a",ls="--",label="Blasius 2.59")
    ax[1].set_xlabel("x (m)"); ax[1].set_ylabel("shape factor $H=\\delta^*/\\theta$"); ax[1].legend()
    ax[1].set_title("Shape factor")
    plt.tight_layout(); plt.savefig("av_fig_blasius.png",dpi=150); plt.close()
    print("saved av_fig_blasius.png")

def fig_resolution():
    if not os.path.exists("av_resolution.csv"): return
    r=np.array([[float(v) for v in row] for row in list(csv.reader(open("av_resolution.csv")))[1:]])
    H,Xr=r[:,0],r[:,2]; inv=1.0/H
    fig,ax=plt.subplots(figsize=(6,4.5))
    ax.plot(inv,Xr,"o-",ms=6)
    if len(H)>=2:                       # Richardson-style extrapolation to zero cell size
        A=np.polyfit(inv,Xr,1); ax.plot([0,inv.max()],[A[1],A[0]*inv.max()+A[1]],"k--",
            label=f"extrap. $X_r/H\\to${A[1]:.2f} as $1/H\\to0$")
        ax.plot(0,A[1],"k*",ms=12)
    for xi,yi,hh in zip(inv,Xr,H): ax.annotate(f"H={int(hh)}",(xi,yi),textcoords="offset points",xytext=(6,6),fontsize=8)
    ax.set_xlabel("1 / H  (cell size)"); ax.set_ylabel("$X_r/H$")
    ax.set_title("Grid convergence — cube reattachment length",fontweight="bold")
    ax.legend(); ax.set_xlim(left=-0.005)
    plt.tight_layout(); plt.savefig("av_fig_resolution.png",dpi=150); plt.close()
    print("saved av_fig_resolution.png")

for fn in (fig_cube_xz,fig_cube_xy,fig_abl,fig_cp,fig_blasius,fig_resolution):
    try: fn()
    except Exception as e: print(f"  {fn.__name__} skipped: {e}")
print("done — av_fig_*.png")
