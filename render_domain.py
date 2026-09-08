#!/usr/bin/env python3
# render_domain.py — plan + elevation view of the forward_city simulation domain
# (city AND relaxed COST 732 buffers). Reads the geometry datasets written by
# forward_city (DUMP_GEOM_ONLY=1): plan_code.u8, plan_height.f32, meta.txt.
#
# Usage:  python3 render_domain.py [OUT_DIR] [out.png]
import sys, os, struct
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow
from matplotlib.colors import ListedColormap, BoundaryNorm

OUT = sys.argv[1] if len(sys.argv) > 1 else "forward_city_out"
PNG = sys.argv[2] if len(sys.argv) > 2 else os.path.join(OUT, "domain.png")

def read_field(path):
    with open(path, "rb") as f:
        nx, ny, nz, dxm, nc = struct.unpack("<5i", f.read(20))
        buf = f.read()
    return nx, ny, nz, dxm/1000.0, nc, buf

def meta(path):
    d = {}
    for ln in open(path):
        ln = ln.strip()
        if not ln or ln.startswith("#"): continue
        parts = ln.split()
        if len(parts) >= 2:
            d[parts[0]] = parts[1:] if len(parts) > 2 else parts[1]
    return d

M = meta(os.path.join(OUT, "meta.txt"))
nx, ny, _pnz, dx, _, cbuf = read_field(os.path.join(OUT, "plan_code.u8"))
# plan rasters are 2-D (header nz=1); take the true 3-D nz/dx from meta.txt
nz = int(M.get("grid_nz", _pnz)); dx = float(M.get("dx_m", dx))
code = np.frombuffer(cbuf, dtype=np.uint8, count=nx*ny).reshape(ny, nx)
_, _, _, _, _, hbuf = read_field(os.path.join(OUT, "plan_height.f32"))
H = np.frombuffer(hbuf, dtype=np.float32, count=nx*ny).reshape(ny, nx)

Sx, Sy = nx*dx, ny*dx
bup   = float(M.get("buf_upwind_m", 0));  bdn = float(M.get("buf_downwind_m", 0))
blat  = float(M.get("buf_lateral_m", 0))
cxn, cyn = (int(v) for v in M["city_origin_cells"])
cxp, cyp = (int(v) for v in M["city_extent_cells"])
maxH  = float(M.get("maxH_m", H.max() if H.size else 0))
nz_relax = int(M.get("nz_relaxed", nz)); nz_cost = int(M.get("nz_cost732", nz))
wind_deg = float(M.get("wind_deg", 0)); U = float(M.get("U_inlet_ms", 0))
Ztop = nz*dx; Zcost = nz_cost*dx

# discrete land-use colormap: 0 buffer,1 street,2 park,3 business,4 res_high,5 res_low
cmap = ListedColormap(["#e9edf2", "#ffffff", "#2e8b57", "#1f3a93", "#e67e22", "#f2c14e"])
norm = BoundaryNorm(range(0, 7), cmap.N)
labels = ["buffer (open)", "street", "park", "business", "res-high", "res-low"]

fig = plt.figure(figsize=(15, 6.4))
gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.22)

# ── Plan (x–y) ───────────────────────────────────────────────────────────────
axp = fig.add_subplot(gs[0, 0])
axp.imshow(code, origin="lower", extent=[0, Sx, 0, Sy], cmap=cmap, norm=norm,
           interpolation="nearest", aspect="equal")
# city boundary
axp.add_patch(Rectangle((cxn*dx, cyn*dx), (cxp-cxn)*dx, (cyp-cyn)*dx,
                        fill=False, ec="k", lw=2.0, ls="--"))
# buffer boundary (full domain)
axp.add_patch(Rectangle((0, 0), Sx, Sy, fill=False, ec="#555", lw=1.4))
# wind arrow (0 deg = +x)
th = np.radians(wind_deg)
axp.add_patch(FancyArrow(0.06*Sx, 1.03*Sy, 0.14*Sx*np.cos(th), 0.14*Sx*np.sin(th),
              width=0.006*Sy, head_width=0.03*Sy, length_includes_head=True,
              color="#c0392b", clip_on=False))
axp.text(0.22*Sx, 1.03*Sy, f"wind {wind_deg:.0f}°, U={U:g} m/s", color="#c0392b",
         va="center", fontsize=10)
# source line (upwind release plane used for the single-source fallback)
sx = float(bup) - 10.0
axp.plot([sx, sx], [cyn*dx, cyp*dx], color="#8e44ad", lw=1.3, ls=":")
axp.text(sx, cyp*dx, " src plane", color="#8e44ad", fontsize=8, va="bottom")
# buffer annotations
axp.annotate("", xy=(bup, -0.045*Sy), xytext=(0, -0.045*Sy),
             arrowprops=dict(arrowstyle="<->", color="#333"), annotation_clip=False)
axp.text(bup/2, -0.075*Sy, f"upwind\n{bup:.0f} m", ha="center", va="top", fontsize=8)
axp.annotate("", xy=(Sx, -0.045*Sy), xytext=(Sx-bdn, -0.045*Sy),
             arrowprops=dict(arrowstyle="<->", color="#333"), annotation_clip=False)
axp.text(Sx-bdn/2, -0.075*Sy, f"downwind\n{bdn:.0f} m", ha="center", va="top", fontsize=8)
axp.annotate("", xy=(-0.05*Sx, blat), xytext=(-0.05*Sx, 0),
             arrowprops=dict(arrowstyle="<->", color="#333"), annotation_clip=False)
axp.text(-0.065*Sx, blat/2, f"lat\n{blat:.0f} m", ha="right", va="center", fontsize=8)
axp.set_xlim(-0.1*Sx, 1.02*Sx); axp.set_ylim(-0.12*Sy, 1.09*Sy)
axp.set_xlabel("x  [m]  (streamwise)"); axp.set_ylabel("y  [m]  (lateral)")
axp.set_title(f"Plan — city {M.get('city_w_m','?')}×{M.get('city_h_m','?')} m  +  relaxed buffers")
handles = [plt.Rectangle((0, 0), 1, 1, color=cmap(i)) for i in range(6)]
axp.legend(handles, labels, loc="upper right", fontsize=8, framealpha=0.9, ncol=2)

# ── Elevation (x–z at the tallest column) ────────────────────────────────────
axe = fig.add_subplot(gs[0, 1])
skyline = H.max(axis=0)            # tallest building vs x
xm = (np.arange(nx)+0.5)*dx
axe.fill_between(xm, 0, skyline, step="mid", color="#34495e", label="buildings (silhouette)")
axe.axhspan(0, 0, color="k")
# domain box
axe.add_patch(Rectangle((0, 0), Sx, Ztop, fill=False, ec="#555", lw=1.4))
# ground
axe.axhline(0, color="k", lw=1.5)
# headroom band (tallest -> domain top)
axe.axhspan(maxH, Ztop, xmin=0, xmax=1, color="#aed6f1", alpha=0.35)
axe.text(0.02*Sx, (maxH+Ztop)/2, f"headroom {Ztop-maxH:.0f} m  (= {float(M.get('headroom_xH',0)):g}×H)",
         fontsize=9, va="center")
# COST 732 top that was trimmed
axe.axhline(Zcost, color="#c0392b", lw=1.4, ls="--")
axe.text(0.98*Sx, Zcost, f" COST 732 top {Zcost:.0f} m (trimmed)", color="#c0392b",
         ha="right", va="bottom", fontsize=8)
axe.axhline(Ztop, color="#2471a3", lw=1.4)
axe.text(0.98*Sx, Ztop, f" relaxed top {Ztop:.0f} m", color="#2471a3", ha="right", va="bottom", fontsize=8)
# upwind / downwind shading
axe.axvspan(0, bup, color="#f2f4f6"); axe.axvspan(Sx-bdn, Sx, color="#f2f4f6")
axe.set_xlim(0, Sx); axe.set_ylim(0, Zcost*1.08)
axe.set_xlabel("x  [m]  (streamwise)"); axe.set_ylabel("z  [m]")
axe.set_title(f"Elevation — grid {nx}×{ny}×{nz} @ {dx:g} m  ({nx*ny*nz/1e6:.1f}M cells)")
axe.legend(loc="upper left", fontsize=8)

blk = M.get("blocks", "?"); pop = M.get("population", "?"); om = M.get("omega_source_cells", "?")
fig.suptitle(f"forward_city simulation domain — {blk} blocks, pop {pop}, |Ω|={om} source cells "
             f"| nz relaxed {nz_relax} vs COST 732 {nz_cost}", fontsize=12, y=1.02)
fig.savefig(PNG, dpi=130, bbox_inches="tight")
print("wrote", PNG)
