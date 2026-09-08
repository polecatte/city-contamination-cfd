#!/usr/bin/env python3
# render_material_map.py — visual verification of the OpenLB geometry bridge.
# Reads geom_out/material_map.dat (+ source_mask.u8) and renders a plan slice and a
# vertical section so the material classification can be eyeballed against the city.
import sys, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

OUT = sys.argv[1] if len(sys.argv) > 1 else "geom_out"

def read5(path, dtype):
    with open(path, "rb") as f:
        nx, ny, nz, dxm, nc = np.fromfile(f, np.int32, 5)
        data = np.fromfile(f, dtype)
    return nx, ny, nz, dxm/1000.0, nc, data

nx, ny, nz, dx, nc, mat = read5(f"{OUT}/material_map.dat", np.int32)
mat = mat.reshape(nz, ny, nx)                       # z-major
_,_,_,_,_, src = read5(f"{OUT}/source_mask.u8", np.uint8)
src = src.reshape(nz, ny, nx)

names  = ["donothing","fluid","wall","inlet","outlet","slip","porous","ground"]
colors = ["#ffffff","#eaf2fb","#3a3a3a","#2f9e44","#e03131","#a5d8ff","#94d82d","#8b6b4a"]
cmap = ListedColormap(colors); norm = BoundaryNorm(np.arange(-0.5,8.5,1), cmap.N)

fig, ax = plt.subplots(1, 3, figsize=(17, 6))

# (a) plan view at ground-release level z=1
z1 = 1
ax[0].imshow(mat[z1], origin="lower", cmap=cmap, norm=norm, interpolation="nearest")
ax[0].set_title(f"plan @ z={z1} (ground release level)")
ax[0].set_xlabel("x (cells)"); ax[0].set_ylabel("y (cells)")

# (b) plan view mid-height — shows building footprints as walls, parks as porous
zmid = max(2, nz//6)
ax[1].imshow(mat[zmid], origin="lower", cmap=cmap, norm=norm, interpolation="nearest")
ax[1].set_title(f"plan @ z={zmid} ({zmid*dx:.0f} m — mid building height)")
ax[1].set_xlabel("x (cells)"); ax[1].set_ylabel("y (cells)")

# (c) vertical section through the middle (x–z), y = ny//2
ysec = ny//2
ax[2].imshow(mat[:, ysec, :], origin="lower", cmap=cmap, norm=norm,
             interpolation="nearest", aspect="auto")
ax[2].set_title(f"section x–z @ y={ysec}")
ax[2].set_xlabel("x (cells)"); ax[2].set_ylabel("z (cells)")

handles = [Patch(facecolor=colors[i], edgecolor="#888", label=f"{i} {names[i]}") for i in range(8)]
fig.legend(handles=handles, loc="lower center", ncol=8, frameon=False, fontsize=9)
fig.suptitle(f"OpenLB material map — {nx}×{ny}×{nz} @ dx={dx:.1f} m   "
             f"(Ω source cells: {int(src.sum())})", fontsize=12)
fig.tight_layout(rect=[0,0.05,1,0.96])
fig.savefig(f"{OUT}/material_map.png", dpi=110)
print(f"wrote {OUT}/material_map.png")

# quick sanity assertions printed for the record
inlet_face = (mat[:,:,0] == 3).sum()
outlet_face = (mat[:,:,-1] == 4).sum()
top_slip = (mat[-1] == 5).sum()
print(f"inlet cells on x=0 face      : {inlet_face}")
print(f"outlet cells on x=nx-1 face  : {outlet_face}")
print(f"slip cells on top z=nz-1     : {top_slip}")
print(f"source Ω cells at z=1        : {int(src[1].sum())} (of {int(src.sum())} total)")
