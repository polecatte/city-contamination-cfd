#!/usr/bin/env python3
# visualize_forward.py — full visualization suite for a forward_city run.
# Reads ONLY the stored datasets in OUT_DIR (self-describing binaries + frames +
# exposure_timeseries.csv), so it can be re-run any time from stored values.
#
# Produces (into OUT_DIR/figs/):
#   conc_ground_maps.png   2D ground concentration at release / mid / end (buildings masked)
#   dep_ground_maps.png    2D ground deposited-material at release / mid / end (buildings masked)
#   exposure_over_time.png total population exposure vs time (cumulative + rate)
#   airflow_ground.png     2D ground wind speed (m/s) + streamlines (mean flow)
#   airflow_3d.png         3D final (mean) airflow with the city buildings
#   concentration_3d.png   3D city with the concentration cloud (open air only)
#   airflow.gif            2D ground airflow — speed + streamlines every frame
#   concentration.gif      2D ground concentration progression (buildings masked)
#
# Buildings are HARD boundaries: every concentration/airflow field is masked to open
# air (fluid streets + porous park ground), so nothing is drawn "inside" a building.
# Each figure is wrapped in try/except: a failure prints a warning and continues,
# and the stored data is untouched so you can retry. Usage:
#   python3 visualize_forward.py [OUT_DIR]
import sys, os, struct, glob
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize

OUT  = sys.argv[1] if len(sys.argv) > 1 else "forward_city_out"
FIGS = os.path.join(OUT, "figs"); os.makedirs(FIGS, exist_ok=True)

U_LB = 0.1 / np.sqrt(3.0)          # lattice inlet speed (Ma=0.1, c_s=1/sqrt3) <-> U_inlet m/s
GIF_STRIDE = max(1, int(os.environ.get("VIZ_GIF_STRIDE", "1")))   # use every Nth frame in the GIFs

# ── readers for the self-describing binaries ──────────────────────────────────
def read5(p):
    with open(p, "rb") as f:
        nx, ny, nz, dxm, nc = struct.unpack("<5i", f.read(20)); buf = f.read()
    return nx, ny, nz, dxm/1000.0, nc, np.frombuffer(buf, dtype=np.float32)
def read2(p):
    with open(p, "rb") as f:
        nx, ny = struct.unpack("<2i", f.read(8)); buf = f.read()
    return nx, ny, np.frombuffer(buf, dtype=np.float32)
def read_meta(p):
    d = {}
    if os.path.exists(p):
        for ln in open(p):
            s = ln.strip()
            if s and not s.startswith("#") and len(s.split()) >= 2:
                k, *v = s.split(); d[k] = v
    return d
def read_u8_grid(p):
    with open(p, "rb") as f:
        nx, ny, nz, dxm, nc = struct.unpack("<5i", f.read(20)); buf = f.read()
    return nx, ny, nz, dxm/1000.0, np.frombuffer(buf, dtype=np.uint8, count=nx*ny*nz).reshape(nz, ny, nx)
def parse_city():
    CELL = 2.0; Sx = Sy = 0.0; B = []
    p = os.path.join(OUT, "city_density.txt")
    if not os.path.exists(p): return CELL, Sx, Sy, B
    for ln in open(p):
        tk = ln.split()
        if not tk: continue
        if tk[0] == "CELL": CELL = float(tk[1])
        elif tk[0] == "SIZE": Sx, Sy = float(tk[1]), float(tk[2])
        elif tk[0] == "B":
            v = list(map(float, tk[1:]))
            # x0..x1 = BUILDING footprint (setback applied, matches what the solver
            # voxelizes); bx0..bx1 = the full lot. Draw the building, not the lot.
            B.append((v[0]*CELL, v[1]*CELL, (v[2]-v[0])*CELL, (v[3]-v[1])*CELL, v[8]*CELL, int(v[9])))
    return CELL, Sx, Sy, B

M = read_meta(os.path.join(OUT, "meta.txt"))
NX, NY, NZ, DX, TYPE = read_u8_grid(os.path.join(OUT, "geom_type.u8"))
U_INLET = float(M.get("U_inlet_ms", [4.0])[0]); V2MS = U_INLET / U_LB      # lattice velocity -> m/s
zped = max(1, min(NZ-1, int(round(2.0/DX))))
EXT = [0, NX*DX, 0, NY*DX]

# source set Ω (streets + porous park ground) so parks stay visible while solid
# buildings are hidden. FLUID (0) is always open air.
SRC = None
sp = os.path.join(OUT, "source_mask.u8")
if os.path.exists(sp):
    _, _, _, _, SRC = read_u8_grid(sp)
tp0 = TYPE[zped]                                   # cell types at pedestrian height
src0 = (SRC[zped] != 0) if SRC is not None else np.zeros_like(tp0, bool)
OPEN = (tp0 == 0) | src0                           # cells where a scalar field is physical
SOLID = ~OPEN                                       # buildings / indoor / solid ground -> mask out
BUILDING = (tp0 == 2) & ~src0                       # solid building footprints (for outline overlay)

TS = None
tsp = os.path.join(OUT, "exposure_timeseries.csv")
if os.path.exists(tsp):
    TS = np.genfromtxt(tsp, delimiter=",", names=True)

def frame_files(prefix):
    pat = "vel_*.bin" if prefix == "vel" else prefix + "_*.f32"
    return sorted(glob.glob(os.path.join(OUT, "frames", pat)))
def frame_time(idx):
    if TS is None: return None
    try: return float(np.atleast_1d(TS["t_s"])[idx])
    except Exception: return None
def mask_solid(field2d):
    return np.ma.masked_where(SOLID, field2d)
def draw_buildings(ax):
    # translucent footprints of solid buildings, drawn on top of the field
    ax.imshow(np.where(BUILDING, 1.0, np.nan), origin="lower", extent=EXT,
              cmap="Greys", vmin=0, vmax=2.2, interpolation="nearest", alpha=0.30, zorder=3)

# ── 2D ground concentration / deposition at release / mid / end ───────────────
def three_time_maps(prefix, title, cmap_name, out):
    fs = frame_files(prefix)
    if not fs: print(f"[viz] no {prefix} frames — skipping {out}"); return
    pick = [0, len(fs)//2, len(fs)-1]; names = ["at release", "after release", "at end"]
    fields = []; vmax = 0.0
    for i in pick:
        nx, ny, a = read2(fs[i]); f = a.reshape(ny, nx); fields.append(f); vmax = max(vmax, np.nanmax(f))
    if vmax <= 0: vmax = 1.0
    floor = vmax*1e-4; norm = LogNorm(floor, vmax)
    cmap = plt.get_cmap(cmap_name).copy(); cmap.set_bad("#c9ccd1")   # masked buildings -> grey
    fig, axs = plt.subplots(1, 3, figsize=(16, 5.2), constrained_layout=True)
    for ax, i, f, nm in zip(axs, pick, fields, names):
        fm = mask_solid(np.maximum(f, floor))
        im = ax.imshow(fm, origin="lower", extent=EXT, cmap=cmap, norm=norm, interpolation="nearest", aspect="equal")
        draw_buildings(ax)
        t = frame_time(i); ax.set_title(f"{nm}" + (f"  (t={t:.0f}s)" if t is not None else f"  (frame {i})"))
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    fig.colorbar(im, ax=axs, shrink=0.8, label=title + " (log)")
    fig.suptitle(title + " — ground level (z≈2 m), buildings masked", fontsize=14)
    fig.savefig(os.path.join(FIGS, out), dpi=120, bbox_inches="tight"); plt.close(fig)
    print("[viz] wrote", out)

def exposure_graph():
    if TS is None: print("[viz] no exposure_timeseries.csv — skipping exposure graph"); return
    t = np.atleast_1d(TS["t_s"]); cum = np.atleast_1d(TS["cumulative_dose"]); rate = np.atleast_1d(TS["exposure_rate"])
    fig, ax1 = plt.subplots(figsize=(9, 5.2))
    ax1.plot(t, cum, "-", color="#1f3a93", lw=2.2, label="cumulative dose ⟨w,∫C dt⟩")
    ax1.set_xlabel("time since release [s]"); ax1.set_ylabel("cumulative population dose", color="#1f3a93")
    ax1.tick_params(axis="y", labelcolor="#1f3a93")
    ax2 = ax1.twinx()
    ax2.plot(t, rate, "--", color="#c0392b", lw=1.6, label="exposure rate ⟨w,C(t)⟩")
    ax2.set_ylabel("instantaneous exposure rate", color="#c0392b"); ax2.tick_params(axis="y", labelcolor="#c0392b")
    for a in (0, len(t)//2, len(t)-1): ax1.axvline(t[a], color="#999", ls=":", lw=1)
    ax1.set_title("Total population exposure over time")
    l1, la1 = ax1.get_legend_handles_labels(); l2, la2 = ax2.get_legend_handles_labels()
    ax1.legend(l1+l2, la1+la2, loc="center right", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "exposure_over_time.png"), dpi=120); plt.close(fig)
    print("[viz] wrote exposure_over_time.png")

# ── airflow: speed heatmap (m/s) + streamlines, buildings masked ──────────────
def _uv(ux2d, uy2d):
    # convert to m/s and zero out solids so streamlines route around them
    ux = np.nan_to_num(ux2d) * V2MS; uy = np.nan_to_num(uy2d) * V2MS
    ux = np.where(SOLID, 0.0, ux); uy = np.where(SOLID, 0.0, uy)
    return ux, uy
def _speed_vmax(frames):
    vm = 0.0
    for p in frames[:: max(1, len(frames)//8)]:
        nx, ny, a = read2(p); ux = a[:nx*ny].reshape(ny, nx); uy = a[nx*ny:2*nx*ny].reshape(ny, nx)
        vm = max(vm, np.nanpercentile(np.hypot(ux, uy)*V2MS, 99))
    return max(1.0, float(np.ceil(vm)))
def _draw_airflow(ax, ux, uy, vmax):
    x = (np.arange(NX)+0.5)*DX; y = (np.arange(NY)+0.5)*DX
    spd = np.hypot(ux, uy)
    im = ax.imshow(np.ma.masked_where(SOLID, spd), origin="lower", extent=EXT, cmap="coolwarm",
                   vmin=0, vmax=vmax, interpolation="bilinear", aspect="equal", zorder=1)
    lw = 0.6 + 1.0*np.clip(spd/max(vmax, 1e-9), 0, 1)
    ax.streamplot(x, y, ux, uy, color="k", density=1.3, linewidth=lw, arrowsize=0.8, zorder=2)
    draw_buildings(ax)
    ax.set_xlim(EXT[0], EXT[1]); ax.set_ylim(EXT[2], EXT[3])
    ax.set_xlabel("x [m] (wind →)"); ax.set_ylabel("y [m]")
    return im

def airflow_ground():
    p = os.path.join(OUT, "umean_full.f32")
    if not os.path.exists(p): print("[viz] no umean_full.f32 — skipping airflow_ground"); return
    nx, ny, nz, dx, nc, a = read5(p); N = nx*ny*nz
    ux = a[0:N].reshape(nz, ny, nx)[zped]; uy = a[N:2*N].reshape(nz, ny, nx)[zped]
    ux, uy = _uv(ux, uy); vmax = max(1.0, float(np.ceil(np.nanpercentile(np.hypot(ux, uy), 99))))
    fig, ax = plt.subplots(figsize=(9.5, 8.6))
    im = _draw_airflow(ax, ux, uy, vmax)
    fig.colorbar(im, ax=ax, shrink=0.85, label="wind speed (m/s)")
    ax.set_title(f"Ground airflow (z≈2 m) — mean speed + streamlines")
    fig.savefig(os.path.join(FIGS, "airflow_ground.png"), dpi=130, bbox_inches="tight"); plt.close(fig)
    print("[viz] wrote airflow_ground.png")

def airflow_gif():
    from matplotlib.animation import FuncAnimation, PillowWriter
    fs = frame_files("vel")[::GIF_STRIDE]
    if len(fs) < 2: print(f"[viz] <2 vel frames — skipping airflow.gif"); return
    vmax = _speed_vmax(fs)
    print(f"[viz] rendering airflow.gif ({len(fs)} frames, streamlines each — the slow step; "
          f"set VIZ_GIF_STRIDE=N to use every Nth frame)...", flush=True)
    fig, ax = plt.subplots(figsize=(8.4, 7.6))
    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=Normalize(0, vmax)); sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, shrink=0.85, label="wind speed (m/s)")
    def frame(i):
        ax.clear()
        nx, ny, a = read2(fs[i]); ux = a[:nx*ny].reshape(ny, nx); uy = a[nx*ny:2*nx*ny].reshape(ny, nx)
        ux, uy = _uv(ux, uy); _draw_airflow(ax, ux, uy, vmax)
        t = frame_time(i); ax.set_title("Ground airflow — speed + streamlines  " +
                                        (f"(t={t:.0f}s)" if t is not None else f"(frame {i})"))
    an = FuncAnimation(fig, frame, frames=len(fs), blit=False)
    an.save(os.path.join(FIGS, "airflow.gif"), writer=PillowWriter(fps=6)); plt.close(fig)
    print("[viz] wrote airflow.gif", f"({len(fs)} frames)")

def concentration_gif():
    from matplotlib.animation import FuncAnimation, PillowWriter
    fs = frame_files("conc")[::GIF_STRIDE]
    if len(fs) < 2: print("[viz] <2 conc frames — skipping concentration.gif"); return
    print(f"[viz] rendering concentration.gif ({len(fs)} frames)...", flush=True)
    vmax = max(np.nanmax(read2(p)[2]) for p in fs) or 1.0
    floor = vmax*1e-4; norm = LogNorm(floor, vmax)
    cmap = plt.get_cmap("inferno").copy(); cmap.set_bad("#c9ccd1")
    fig, ax = plt.subplots(figsize=(7.6, 6.8))
    im = ax.imshow(mask_solid(np.full((NY, NX), floor)), origin="lower", extent=EXT,
                   cmap=cmap, norm=norm, interpolation="nearest", aspect="equal")
    fig.colorbar(im, ax=ax, shrink=0.85, label="concentration C (log)")
    def frame(i):
        nx, ny, a = read2(fs[i]); f = mask_solid(np.maximum(a.reshape(ny, nx), floor))
        im.set_data(f); t = frame_time(i)
        ax.set_title("Ground concentration — " + (f"t={t:.0f}s" if t is not None else f"frame {i}"))
    an = FuncAnimation(fig, frame, frames=len(fs), blit=False)
    # buildings drawn once (static geometry)
    draw_buildings(ax); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    an.save(os.path.join(FIGS, "concentration.gif"), writer=PillowWriter(fps=6)); plt.close(fig)
    print("[viz] wrote concentration.gif", f"({len(fs)} frames)")

# ── 3D figures ────────────────────────────────────────────────────────────────
def buildings3d(ax, B):
    col = {0:"#2e8b57",1:"#1f3a93",2:"#e67e22",3:"#f2c14e"}
    if not B: return
    ax.bar3d([b[0] for b in B],[b[1] for b in B],[0]*len(B),
             [b[2] for b in B],[b[3] for b in B],[b[4] for b in B],
             color=[col.get(b[5],"#888") for b in B], edgecolor="k", linewidth=0.15, shade=True, alpha=0.55)

def airflow_3d():
    p = os.path.join(OUT, "umean_full.f32")
    if not os.path.exists(p): print("[viz] no umean_full.f32 — skipping airflow_3d"); return
    nx, ny, nz, dx, nc, a = read5(p); N = nx*ny*nz
    ux = a[0:N].reshape(nz,ny,nx); uy = a[N:2*N].reshape(nz,ny,nx); uz = a[2*N:3*N].reshape(nz,ny,nx)
    _, Sx, Sy, B = parse_city(); s = max(1, nx//26); sz = max(1, nz//10)
    X=[];Y=[];Z=[];U=[];V=[];W=[]
    for z in range(1, min(nz, int(nz*0.6)), sz):
        for y in range(1, ny-1, s):
            for x in range(1, nx-1, s):
                if TYPE[z,y,x] != 0: continue
                X.append(x*dx); Y.append(y*dx); Z.append(z*dx)
                U.append(ux[z,y,x]); V.append(uy[z,y,x]); W.append(uz[z,y,x])
    if not X: print("[viz] no fluid vectors — skipping airflow_3d"); return
    spd = np.sqrt(np.array(U)**2+np.array(V)**2+np.array(W)**2)*V2MS
    fig = plt.figure(figsize=(12, 8.5)); ax = fig.add_subplot(111, projection="3d")
    buildings3d(ax, B)
    ax.quiver(X, Y, Z, U, V, W, length=0.9*s*dx/(np.sqrt((np.array(U)**2+np.array(V)**2+np.array(W)**2)).max() or 1),
              normalize=False, linewidth=0.6, color=plt.cm.coolwarm(spd/(spd.max() or 1)))
    ax.set_xlim(0,nx*dx); ax.set_ylim(0,ny*dx)
    ztop = min(nz*dx, (max(b[4] for b in B) if B else nz*dx)*2.2); ax.set_zlim(0,ztop)
    ax.set_box_aspect((nx*dx, ny*dx, ztop)); ax.view_init(elev=32, azim=-60)
    ax.set_xlabel("x [m] (wind→)"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
    m = plt.cm.ScalarMappable(cmap="coolwarm", norm=Normalize(0, spd.max() or 1)); m.set_array(spd)
    fig.colorbar(m, ax=ax, shrink=0.6, label="mean wind speed (m/s)")
    ax.set_title("Final (mean) airflow with city environment")
    fig.savefig(os.path.join(FIGS, "airflow_3d.png"), dpi=120, bbox_inches="tight"); plt.close(fig)
    print("[viz] wrote airflow_3d.png")

def concentration_3d():
    p = os.path.join(OUT, "conc3d_mid.f32")
    if not os.path.exists(p): print("[viz] no conc3d_mid.f32 — skipping concentration_3d"); return
    nx, ny, nz, dx, nc, a = read5(p); C = a.reshape(nz, ny, nx).copy()
    C[TYPE != 0] = 0.0                              # open air only — never inside solids
    _, Sx, Sy, B = parse_city(); cmax = np.nanmax(C)
    if cmax <= 0: print("[viz] concentration field empty — skipping concentration_3d"); return
    thr = cmax*0.02; zz, yy, xx = np.where(C > thr)
    if len(xx) > 60000:
        sel = np.random.default_rng(0).choice(len(xx), 60000, replace=False); zz, yy, xx = zz[sel], yy[sel], xx[sel]
    cvals = C[zz, yy, xx]
    fig = plt.figure(figsize=(12, 8.5)); ax = fig.add_subplot(111, projection="3d")
    buildings3d(ax, B)
    ax.scatter(xx*dx, yy*dx, zz*dx, c=cvals, cmap="inferno", norm=LogNorm(thr, cmax), s=6, alpha=0.35, edgecolors="none")
    ax.set_xlim(0,nx*dx); ax.set_ylim(0,ny*dx)
    ztop = min(nz*dx, (max(b[4] for b in B) if B else nz*dx)*2.5); ax.set_zlim(0,ztop)
    ax.set_box_aspect((nx*dx, ny*dx, ztop)); ax.view_init(elev=28, azim=-60)
    ax.set_xlabel("x [m] (wind→)"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
    m = plt.cm.ScalarMappable(cmap="inferno", norm=LogNorm(thr, cmax)); m.set_array(cvals)
    fig.colorbar(m, ax=ax, shrink=0.6, label="concentration C")
    ax.set_title("City with 3D concentration cloud (open air only)")
    fig.savefig(os.path.join(FIGS, "concentration_3d.png"), dpi=120, bbox_inches="tight"); plt.close(fig)
    print("[viz] wrote concentration_3d.png")

TASKS = [
    ("conc ground maps",  lambda: three_time_maps("conc", "Concentration C", "inferno", "conc_ground_maps.png")),
    ("deposition maps",   lambda: three_time_maps("dep",  "Deposited material", "viridis", "dep_ground_maps.png")),
    ("exposure graph",    exposure_graph),
    ("airflow ground 2D", airflow_ground),
    ("airflow 3D",        airflow_3d),
    ("concentration 3D",  concentration_3d),
    ("airflow gif",       airflow_gif),
    ("concentration gif", concentration_gif),
]
ok = 0
for name, fn in TASKS:
    try: fn(); ok += 1
    except Exception as e:
        import traceback; print(f"[viz] FAILED: {name}: {e}"); traceback.print_exc()
print(f"[viz] done — {ok}/{len(TASKS)} figures into {FIGS}/  (stored data untouched; safe to re-run)")
