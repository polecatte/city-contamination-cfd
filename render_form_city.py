#!/usr/bin/env python3
"""render_form_city.py GEOM_DIR [GEOM_DIR ...] OUT.png — built-form cities from gen_form_city.

One column per city: a 3-D view (height colour, parks green) over a plan of the whole domain
(city, environs ring with its release zones, inlet on the left, wake buffer on the right).
"""
import os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def read5(fn, dt):
    b = open(fn, "rb").read(); h = np.frombuffer(b[:20], np.int32)
    return h, np.frombuffer(b[20:], dt)


def meta(g):
    d = {}
    for l in open(os.path.join(g, "meta_geom.txt")):
        p = l.split()
        if len(p) == 2 and not p[0].startswith("#"):
            try: d[p[0]] = float(p[1])
            except ValueError: d[p[0]] = p[1]
    return d


def main():
    dirs, png = sys.argv[1:-1], sys.argv[-1]
    fig = plt.figure(figsize=(6.2 * len(dirs), 10))
    cm = plt.cm.plasma
    for c, g in enumerate(dirs):
        m = meta(g)
        dx = m["dx_m"]; ox, oy, L, ring = m["city_origin_x_m"], m["city_origin_y_m"], m["city_m"], m["environ_ring_m"]
        rows = np.genfromtxt(os.path.join(g, "buildings.csv"), delimiter=",", names=True)
        h, mat = read5(os.path.join(g, "material_map.dat"), np.int32)
        nx, ny, nz = h[:3]; mat = mat.reshape(nz, ny, nx)
        park = (mat == 6).any(0)
        hmax = 100.0
        ax = fig.add_subplot(2, len(dirs), c + 1, projection="3d")
        for r in rows:
            x0, y0 = r["x0"] * dx - ox, r["y0"] * dx - oy
            ax.bar3d(x0, y0, 0, (r["x1"] - r["x0"]) * dx, (r["y1"] - r["y0"]) * dx, r["height_m"],
                     color=cm(r["height_m"] / hmax), edgecolor="k", linewidth=0.15, shade=True)
        pys, pxs = np.where(park)
        if len(pxs):
            ax.scatter(pxs * dx - ox, pys * dx - oy, np.full(len(pxs), 5.0), s=0.3, c="#51cf66", alpha=0.4)
        ax.set_xlim(0, L); ax.set_ylim(0, L); ax.set_zlim(0, hmax); ax.set_box_aspect((1, 1, 0.35))
        ax.view_init(elev=32, azim=-60); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
        t = (f"block {m['block_size']:.0f} m, aspect {m['block_aspect']:.2f}, street {m['street_w']:.0f} m, "
             f"corridors {int(m['corridors'])}\nheight conc {m['height_conc']:.2f}, cores {int(m['height_cores'])}, "
             f"scatter {m['height_scatter']:.2f}; parks {m['park_frac']:.2f} (layout {m['park_layout']:.1f})\n"
             f"{int(m['buildings'])} buildings, {int(m['parks'])} parks, 2–{int(m['max_floors'])} floors, "
             f"GFA {m['gfa_m2']/1e3:.0f}k m², {m['population']:.0f} people")
        ax.set_title(t, fontsize=9)
        ax2 = fig.add_subplot(2, len(dirs), len(dirs) + c + 1)
        hz, zone = read5(os.path.join(g, "release_zone.u8"), np.uint8)
        zone = zone.reshape(nz, ny, nx).max(0).astype(float)
        zone[zone == 0] = np.nan
        ext = [0, nx * dx, 0, ny * dx]
        zz = np.where(np.isnan(zone), 0, zone).astype(int) - 1; zpar = np.where(np.isnan(zone), np.nan, (zz // int(np.sqrt(np.nanmax(zone))) + zz % int(np.sqrt(np.nanmax(zone)))) % 2)
        ax2.imshow(zpar.astype(float), origin="lower", extent=ext,
                   cmap="Greys", vmin=-3, vmax=4, alpha=0.6, interpolation="nearest")
        hmap = np.zeros((ny, nx));
        for r in rows: hmap[int(r["y0"]):int(r["y1"]), int(r["x0"]):int(r["x1"])] = r["height_m"]
        ax2.imshow(np.ma.masked_where(hmap == 0, hmap), origin="lower", extent=ext, cmap=cm, vmin=0, vmax=hmax,
                   interpolation="nearest")
        ax2.imshow(np.where(park, 1.0, np.nan), origin="lower", extent=ext, cmap="Greens", vmin=0, vmax=1.3,
                   interpolation="nearest")
        ax2.add_patch(Rectangle((ox, oy), L, L, fill=False, ec="k", lw=1))
        ax2.add_patch(Rectangle((ox - ring, oy - ring), L + 2 * ring, L + 2 * ring, fill=False, ec="#e8590c", lw=1, ls="--"))
        ax2.annotate("inlet", xy=(0, ny * dx / 2), xytext=(4, ny * dx / 2 + 40), fontsize=8)
        ax2.set_title(f"domain {nx}×{ny}×{nz} at {dx:.0f} m; dashed = environs (release zones shaded)", fontsize=9)
        ax2.set_xlabel("x (m)"); ax2.set_ylabel("y (m)")
    fig.tight_layout(); fig.savefig(png, dpi=100)


if __name__ == "__main__":
    main()
