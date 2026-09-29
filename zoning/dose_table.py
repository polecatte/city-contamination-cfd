#!/usr/bin/env python3
"""dose_table.py — per-building dose for every scenario: the zoning programme's only input.

A dose table (.npz) holds, for S scenarios (wind direction x release zone) and B buildings:
    env[S, B]     mean dose Theta over the building's envelope air cells (all heights): what its
                  occupants' indoor air derives from (x infiltration factor in the programme)
    street[S, B]  mean dose over its envelope cells at street level (z = 1): outdoor time
    prob[S]       scenario probability (wind-rose weight x release-zone area share), sums to 1
    wind[S], zone[S]
All per unit released mass.

Two producers:
  from_runs   urban_flow outputs (theta.f32 per scenario) + the geometry's envelope.i32.
              One labelled-tracer run per wind direction will give one theta per release zone
              (TWO_TIER_DESIGN.md step 3); until then a single uniform-release theta per wind
              is one scenario with zone 0.
  proxy       a 2-D steady advection-diffusion-removal stand-in on the city plan, for building
              and testing the programme before the CFD dose exists. NOT CFD, NOT CALIBRATED:
              uniform wind slowed by the local built fraction, eddy diffusion, removal to the
              air above the canopy, and a vertical decay exp(-z/LZ) of a ground release.

    python3 zoning/dose_table.py proxy GEOM_DIR OUT.npz [--winds 16]
    python3 zoning/dose_table.py from_runs GEOM_DIR OUT.npz RUN_DIR:WIND_DEG:ZONE:PROB ...
"""
import os, sys, csv
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu


def read5(fn, dt):
    b = open(fn, "rb").read()
    h = np.frombuffer(b[:20], np.int32)
    nx, ny, nz = (int(v) for v in h[:3])
    return (nx, ny, nz, h[3] / 1000.0), np.frombuffer(b[20:], dt)[: nx * ny * nz].reshape(nz, ny, nx)


def buildings(geom):
    with open(os.path.join(geom, "buildings.csv")) as f:
        return list(csv.DictReader(f))


def envelope_index(geom):
    (nx, ny, nz, dx), env = read5(os.path.join(geom, "envelope.i32"), np.int32)
    return env, dx


def from_runs(geom, specs):
    """specs: list of (run_dir, wind_deg, zone, prob)."""
    env, _ = envelope_index(geom)
    B = len(buildings(geom))
    flat = env.ravel(); street = env[1].ravel()
    cnt = np.bincount(flat, minlength=B + 1)[1:]
    cnt_s = np.bincount(street, minlength=B + 1)[1:]
    E, S, P, W, Z = [], [], [], [], []
    for run, wind, zone, prob in specs:
        _, th = read5(os.path.join(run, "theta.f32"), np.float32)
        meta = {l.split()[0]: l.split()[1] for l in open(os.path.join(run, "meta_flow.txt")) if len(l.split()) == 2}
        emit = float(meta.get("mass_emitted_total", meta["mass_emitted"]))
        th = th.astype(np.float64) / emit
        E.append(np.bincount(flat, weights=th.ravel(), minlength=B + 1)[1:] / np.maximum(cnt, 1))
        S.append(np.bincount(street, weights=th[1].ravel(), minlength=B + 1)[1:] / np.maximum(cnt_s, 1))
        P.append(prob); W.append(wind); Z.append(zone)
    P = np.array(P, float)
    return dict(env=np.array(E), street=np.array(S), prob=P / P.sum(), wind=np.array(W, float), zone=np.array(Z, int))


def proxy(geom, n_winds=16, U=3.0, K=4.0, LAM0=0.02, LZ=20.0):
    """2-D stand-in dose on the city plan (see module docstring)."""
    (nx, ny, nz, dx), mat = read5(os.path.join(geom, "material_map.dat"), np.int32)
    _, zone3 = read5(os.path.join(geom, "release_zone.u8"), np.uint8)
    rows = buildings(geom)
    solid = mat[1] == 2                                      # buildings at street level
    zone = zone3.max(axis=0).astype(int)                     # release tile per plan cell
    hgt = np.zeros((ny, nx))
    for r in rows:
        hgt[int(r["y0"]):int(r["y1"]), int(r["x0"]):int(r["x1"])] = float(r["height_m"])
    # local built fraction and height -> wind slow-down and canopy ventilation (5 x 5 cells)
    from numpy.lib.stride_tricks import sliding_window_view as sw
    pad = np.pad(solid.astype(float), 2, mode="edge"); lam = sw(pad, (5, 5)).mean(axis=(-1, -2))
    padh = np.pad(hgt, 2, mode="edge"); hloc = sw(padh, (5, 5)).max(axis=(-1, -2))
    speed = U * (1.0 - 0.8 * lam)
    vent = LAM0 * np.clip(20.0 / np.maximum(hloc, 20.0), 0.1, 1.0)   # taller surroundings ventilate less
    open_ = ~solid
    idx = -np.ones((ny, nx), int); cells = np.argwhere(open_); idx[open_] = np.arange(len(cells))
    n = len(cells); inv = 1.0 / dx
    zones = np.arange(1, zone.max() + 1)
    src = np.zeros((n, len(zones)))
    for k, zz in enumerate(zones):
        m = (zone == zz) & open_
        src[idx[m], k] = 1.0 / max(1, ((zone == zz)).sum())   # unit mass per zone, spread per unit area
        # (release on roofs of solid plan cells is lumped onto the zone's open cells in 2-D)
    env_rows = [np.zeros((ny, nx), bool) for _ in rows]
    for b, r in enumerate(rows):
        m = env_rows[b]; y0, y1, x0, x1 = int(r["y0"]), int(r["y1"]), int(r["x0"]), int(r["x1"])
        m[max(0, y0 - 1):y1 + 1, max(0, x0 - 1):x1 + 1] = True; m &= open_
    hb = np.array([float(r["height_m"]) for r in rows])
    zbar = np.array([LZ * (1 - np.exp(-h / LZ)) / max(h, 1e-9) for h in hb])    # mean of exp(-z/LZ) over the height
    E, S, P, W, Z = [], [], [], [], []
    for w in np.arange(n_winds) * 360.0 / n_winds:
        th = np.deg2rad(270.0 - w); ux = speed * np.cos(th); uy = speed * np.sin(th)
        r_, c_, v_ = [], [], []; diag = np.zeros(n)
        for k, (y, x) in enumerate(cells):
            d = vent[y, x]
            for dy, dxn, uc in ((0, 1, ux[y, x]), (0, -1, -ux[y, x]), (1, 0, uy[y, x]), (-1, 0, -uy[y, x])):
                yy, xx = y + dy, x + dxn
                inside = 0 <= yy < ny and 0 <= xx < nx
                if inside and open_[yy, xx]:
                    d += K * inv * inv; r_.append(k); c_.append(idx[yy, xx]); v_.append(-K * inv * inv)
                elif not inside:
                    d += K * inv * inv
                if uc < 0:
                    a = -uc * inv; d += a
                    if inside and open_[yy, xx]:
                        r_.append(k); c_.append(idx[yy, xx]); v_.append(-a)
            diag[k] = d
        A = (sp.csr_matrix((v_, (r_, c_)), shape=(n, n)) + sp.diags(diag)).tocsc()
        C = splu(A).solve(src)                                # all release zones at once
        for k, zz in enumerate(zones):
            field = np.zeros((ny, nx)); field[open_] = C[:, k]
            ground = np.array([field[m].mean() for m in env_rows])
            E.append(ground * zbar); S.append(ground)
            P.append((zone == zz).sum()); W.append(w); Z.append(zz)
    P = np.array(P, float) / n_winds
    return dict(env=np.array(E), street=np.array(S), prob=P / P.sum(), wind=np.array(W), zone=np.array(Z))


if __name__ == "__main__":
    mode, geom, out = sys.argv[1:4]
    if mode == "proxy":
        nw = int(sys.argv[sys.argv.index("--winds") + 1]) if "--winds" in sys.argv else 16
        T = proxy(geom, nw)
    else:
        specs = []
        for s in sys.argv[4:]:
            run, w, z, p = s.split(":"); specs.append((run, float(w), int(z), float(p)))
        T = from_runs(geom, specs)
    np.savez(out, **T)
    print(f"dose table {out}: {T['env'].shape[0]} scenarios x {T['env'].shape[1]} buildings")
