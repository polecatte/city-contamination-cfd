"""lab_visualize.py — visualize the lab_test outputs.
Produces: (1) city 2D + 3D, (2) velocity at z=2, (3) particulate dispersion
after 1 h, (4) exposure map + total exposure. Run after ./lab_test."""
import os, struct
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from render_city import load_city, draw_2d, draw_3d, COLORS, LABELS
from matplotlib.patches import Patch

OUT = os.environ.get("FIG_DIR", "figures")   # local folder; override with FIG_DIR=…
os.makedirs(OUT, exist_ok=True)

def load_field(fn):
    with open(fn, "rb") as f:
        nx, ny = struct.unpack("2i", f.read(8))
        a = np.frombuffer(f.read(nx*ny*4), dtype=np.float32).reshape(ny, nx).copy()
    return a, nx, ny

def load_vel(fn):
    with open(fn, "rb") as f:
        nx, ny = struct.unpack("2i", f.read(8))
        ux = np.frombuffer(f.read(nx*ny*4), dtype=np.float32).reshape(ny, nx).copy()
        uy = np.frombuffer(f.read(nx*ny*4), dtype=np.float32).reshape(ny, nx).copy()
    return ux, uy, nx, ny

bl, m = load_city("city_export.txt")
cell = m.get("cell", 4.0)
Sx, Sy = m["Sx"], m["Sy"]
legend = [Patch(facecolor=COLORS[i], edgecolor="#444", label=LABELS[i]) for i in range(5)]

# ── (1) City: 2D plan + 3D massing ──────────────────────────────────────────
fig = plt.figure(figsize=(20, 9))
ax2d = fig.add_subplot(1, 2, 1)
draw_2d(ax2d, bl, m, cell=cell)
ax2d.set_title(f"City plan — {m['N']} blocks, max {m['maxH']:.0f} m, pop {m['pop']:.0f}",
               fontsize=13, fontweight="bold")
ax2d.set_xlabel("x (m)"); ax2d.set_ylabel("y (m)")
ax3d = fig.add_subplot(1, 2, 2, projection="3d")
draw_3d(ax3d, bl, m, cell=cell)
ax3d.set_title("City massing (3D)", fontsize=13, fontweight="bold")
fig.legend(handles=legend, loc="lower center", ncol=5, fontsize=11)
fig.suptitle("(1) City: streets, usages, massing", fontsize=15, fontweight="bold")
plt.tight_layout(rect=[0, 0.04, 1, 0.97])
plt.savefig(f"{OUT}/lab_1_city.png", dpi=140, bbox_inches="tight"); plt.close()
print("saved lab_1_city.png")

# ── (2) Velocity field at z=2 ───────────────────────────────────────────────
if os.path.exists("vel_z2.bin"):
    ux, uy, nx, ny = load_vel("vel_z2.bin")
    spd = np.sqrt(ux**2 + uy**2)
    fig, ax = plt.subplots(figsize=(12, 11))
    draw_2d(ax, bl, m, cell=cell)
    xs = np.linspace(0, Sx, nx); ys = np.linspace(0, Sy, ny)
    im = ax.pcolormesh(xs, ys, spd, cmap="viridis", alpha=0.6, shading="auto", zorder=5)
    plt.colorbar(im, ax=ax, shrink=0.7, label="speed (lattice units)")
    st = max(1, nx // 48)
    ax.quiver(xs[::st], ys[::st], ux[::st, ::st], uy[::st, ::st],
              color="white", scale=2.0, width=0.0018, zorder=6, alpha=0.8)
    ax.set_title("(2) Velocity field at z=2 (≈8 m AGL)", fontsize=14, fontweight="bold")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    plt.tight_layout(); plt.savefig(f"{OUT}/lab_2_velocity_z2.png", dpi=140, bbox_inches="tight"); plt.close()
    print("saved lab_2_velocity_z2.png")

# ── (3) Particulate dispersion after 1 h ────────────────────────────────────
if os.path.exists("dispersion_z1.bin"):
    disp, nx, ny = load_field("dispersion_z1.bin")
    fig, ax = plt.subplots(figsize=(12, 11))
    draw_2d(ax, bl, m, cell=cell)
    pos = disp[disp > 0]
    vmax = np.percentile(pos, 99.5) if pos.size else 1.0
    vmin = max(vmax * 1e-3, pos.min() if pos.size else 1e-6)   # log floor
    masked = np.ma.masked_less_equal(disp, vmin)
    xs = np.linspace(0, Sx, nx); ys = np.linspace(0, Sy, ny)
    im = ax.pcolormesh(xs, ys, masked, cmap="inferno", alpha=0.85, shading="auto",
                       norm=LogNorm(vmin=vmin, vmax=vmax), zorder=6)
    plt.colorbar(im, ax=ax, shrink=0.7, label="time-integrated air conc. (mass-weighted, log)")
    ax.set_title("(3) Particulate dispersion after 1 h\n(mass-weighted TIAC, z=1, log scale)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    plt.tight_layout(); plt.savefig(f"{OUT}/lab_3_dispersion.png", dpi=140, bbox_inches="tight"); plt.close()
    print("saved lab_3_dispersion.png")

# ── (4) Exposure map + total exposure ───────────────────────────────────────────────
if os.path.exists("exposure_z1.bin"):
    expo, nx, ny = load_field("exposure_z1.bin")
    total = {}
    if os.path.exists("total_exposure.txt"):
        for ln in open("total_exposure.txt"):
            k, v = ln.split(); total[k] = float(v)
    fig, ax = plt.subplots(figsize=(12, 11))
    draw_2d(ax, bl, m, cell=cell)
    pos = expo[expo > 0]
    vmax = np.percentile(pos, 99.5) if pos.size else 1.0
    vmin = max(vmax * 1e-3, pos.min() if pos.size else 1e-6)
    masked = np.ma.masked_less_equal(expo, vmin)
    xs = np.linspace(0, Sx, nx); ys = np.linspace(0, Sy, ny)
    im = ax.pcolormesh(xs, ys, masked, cmap="magma", alpha=0.85, shading="auto",
                       norm=LogNorm(vmin=vmin, vmax=vmax), zorder=6)
    plt.colorbar(im, ax=ax, shrink=0.7, label="exposure (airborne + deposited-surface, arb. units)")
    td = total.get("objective", float("nan"))
    sub = (f"TOTAL collective exposure = {td:.3e} a.u.   "
           f"(indoor {total.get('buildings',0):.2e} | street {total.get('street',0):.2e} | "
           f"park {total.get('park',0):.2e}); pop {total.get('population',0):.0f}")
    ax.set_title("(4) Exposure map after 1 h\n" + sub, fontsize=12, fontweight="bold")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    plt.tight_layout(); plt.savefig(f"{OUT}/lab_4_exposure.png", dpi=140, bbox_inches="tight"); plt.close()
    print("saved lab_4_exposure.png")
    print(f"TOTAL EXPOSURE = {td:.4e} (arbitrary units)")
