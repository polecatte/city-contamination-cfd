#!/usr/bin/env python3
"""gate6_analyze.py — Phase-5 airflow gates from urban_flow's time-mean field.

    python3 tests/gate6_analyze.py abl  GEOM_DIR OUT_DIR   # Gate 6a: ABL drift
    python3 tests/gate6_analyze.py cube GEOM_DIR OUT_DIR   # Gate 6b: cube Xr/H

Reads GEOM_DIR/material_map.dat and OUT_DIR/uavg.f32 (written by urban_flow with
AVG_FT>0; 5-int header [nx,ny,nz,dx*1000,ncomp] + ncomp float32 planes, z-major).
Exit status 0 = gate PASS, 1 = FAIL, 2 = input problem. Writes a PNG next to uavg.f32.

Gate 6a (OPENLB_PORT_STATUS_AND_VERIFICATION.md Phase 5a). Drift is measured the way the
old solver's T_abl measured it (tests/airflow_validation.cpp): max over z of
|U_station(z) - U_inlet(z)| / U_ref, U_ref = inlet U at z = 40 m, U laterally averaged
over the interior. The gate is the city-face station (BUF_UP = 40 m downstream of the
inlet); the 1/4, 1/2, 3/4 L stations are reported as the G1 evidence, since the city
itself spans them. Also reported: the inlet's time mean against the target log law,
which separates inlet error from solver drift.

Gate 6b (Phase 5b). Xr = distance from the leeward face (half-way between the last
wall cell and the first fluid cell) to where the near-ground (first fluid cell)
centreline mean u_x turns positive. Band 1.4-1.8 H (LES; Martinuzzi & Tropea 1993,
Tominaga et al. 2008); the old validation roster's sanity band 1.0-2.5 H is also shown.
"""
import sys, os
import numpy as np

MAT_FLUID, MAT_WALL, MAT_INLET, MAT_OUTLET, MAT_SLIP, MAT_GROUND = 1, 2, 3, 4, 5, 7


def read5(fn, dtype):
    with open(fn, 'rb') as f:
        h = np.frombuffer(f.read(20), dtype=np.int32)
        nx, ny, nz, dxmm, nc = (int(v) for v in h)
        a = np.frombuffer(f.read(), dtype=dtype)
    if a.size != nx * ny * nz * nc:
        raise SystemExit(f"{fn}: payload {a.size} != {nx}*{ny}*{nz}*{nc}")
    # file order: component, z, y, x  ->  array [c, z, y, x]
    return a.reshape(nc, nz, ny, nx), dxmm / 1000.0


def log_law(z, U_ref=4.0, z_ref=4.0, z0=0.045, kappa=0.41):
    ustar = U_ref * kappa / np.log((z_ref + z0) / z0)
    zz = np.maximum(z, 0.1)
    return ustar / kappa * np.log((zz + z0) / z0)


def gate_abl(mat, u, dx, out):
    _, nz, ny, nx = u.shape
    ux = u[0]
    ys = slice(2, ny - 2)                        # interior: clear of the slip faces
    prof = lambda x: ux[:, ys, x].mean(axis=1)  # U(z), laterally averaged
    z = np.arange(nz) * dx
    zs = np.arange(1, nz - 1)                    # first fluid cell .. below the top slip
    iref = int(round(40.0 / dx))
    Uin = prof(0)
    Uref = Uin[iref]
    stations = {'city face (40 m)': int(round(40.0 / dx)),
                '1/4 L': nx // 4, '1/2 L': nx // 2, '3/4 L': (3 * nx) // 4}
    print(f"[6a] grid {nx}x{ny}x{nz} dx={dx} m; U_ref = U_inlet(z=40 m) = {Uref:.3f} m/s")
    tgt = log_law(z)
    inl_err = np.max(np.abs(Uin[zs] - tgt[zs])) / Uref
    print(f"[6a] inlet time-mean vs target log law: max |dU|/U_ref = {100*inl_err:.1f}%"
          "  (the inlet's own fidelity, not solver drift)")
    res = {}
    lowz = zs[z[zs] <= 100.0]
    for name, xs in stations.items():
        U = prof(xs)
        d_all = np.max(np.abs(U[zs] - Uin[zs])) / Uref
        d_low = np.max(np.abs(U[lowz] - Uin[lowz])) / Uref
        kmax = zs[np.argmax(np.abs(U[zs] - Uin[zs]))]
        res[name] = d_all
        print(f"[6a] {name:17s} x={xs*dx:6.0f} m  drift max_z = {100*d_all:5.1f}%"
              f" (worst at z={kmax*dx:.0f} m)   z<=100 m: {100*d_low:5.1f}%"
              f"   U(4 m) {Uin[1]:.2f}->{U[1]:.2f}  U(40 m) {Uin[iref]:.2f}->{U[iref]:.2f} m/s")
    gate = res['city face (40 m)']
    worst = max(res.values())
    ok = gate < 0.10
    print(f"[6a] GATE 6a (inlet -> city face < 10%): {100*gate:.1f}%  {'PASS' if ok else 'FAIL'}")
    print(f"[6a] over the city fetch (to 3/4 L) the worst drift is {100*worst:.1f}%"
          f"  -> G1 evidence: {'floor holds the profile' if worst < 0.10 else 'floor does NOT hold the profile'}")
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 6))
        ax.plot(tgt[zs], z[zs], 'k--', lw=1, label='target log law')
        ax.plot(Uin[zs], z[zs], lw=2, label='inlet (x=0)')
        for name, xs in stations.items():
            ax.plot(prof(xs)[zs], z[zs], lw=1.2, label=f'{name}')
        ax.set_xlabel('time-mean U (m/s)'); ax.set_ylabel('z (m)'); ax.set_yscale('log')
        ax.set_title('Gate 6a: ABL profile along the fetch'); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(out, 'gate6a_profiles.png'), dpi=120)
    except Exception as e:  # plotting is a convenience, never a gate
        print(f"[6a] (plot skipped: {e})")
    return ok


def gate_cube(mat, u, dx, out):
    _, nz, ny, nx = u.shape
    ux = u[0]
    wall = (mat == MAT_WALL)
    zc, yc, xc = np.nonzero(wall)
    x0, x1, y0, y1, H = xc.min(), xc.max(), yc.min(), yc.max(), zc.max()
    print(f"[6b] cube x {x0}..{x1}, y {y0}..{y1}, height {H} cells ({H*dx:.0f} m); grid {nx}x{ny}x{nz}")
    ycs = [(y0 + y1) // 2, (y0 + y1 + 1) // 2]   # the centreline (two cells for even H)
    line = ux[1, ycs, :].mean(axis=0)            # first fluid cell above the ground
    xs = np.arange(nx)
    lee = x1 + 0.5                               # leeward wall face (half-way BB)
    down = xs > x1
    rev = np.nonzero(down & (line < 0))[0]
    if rev.size == 0:
        print("[6b] no reversed near-ground flow behind the cube -- no recirculation: FAIL")
        return False
    # first sign change from - to + after the reversed region begins
    xr = None
    for i in range(rev[0], nx - 1):
        if line[i] < 0 <= line[i + 1]:
            xr = i + (-line[i]) / (line[i + 1] - line[i])
            break
    if xr is None:
        print("[6b] reversed flow never reattaches before the outlet: FAIL")
        return False
    XrH = (xr - lee) / H
    up = line[:x0]
    upstream_rev = bool(np.any(up[max(0, x0 - H):] < 0))
    roof = ux[H + 1, ycs, x0:x1 + 1].mean(axis=0)
    print(f"[6b] near-ground centreline u_x: min {line[down].min():.3f} m/s behind the cube")
    print(f"[6b] reattachment at x = {xr*dx:.1f} m; leeward face at {lee*dx:.1f} m"
          f"  ->  Xr/H = {XrH:.2f}")
    print(f"[6b] upstream horseshoe reversal (within 1H of the windward face): {'yes' if upstream_rev else 'no'};"
          f" roof reversal: {'yes' if np.any(roof < 0) else 'no'}")
    ok = 1.4 <= XrH <= 1.8
    wide = 1.0 <= XrH <= 2.5
    print(f"[6b] GATE 6b (Xr/H in 1.4-1.8): {'PASS' if ok else 'FAIL'}"
          f"   (validation-roster sanity band 1.0-2.5: {'inside' if wide else 'outside'})")
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.plot((xs - lee) / H, line, lw=1.5)
        ax.axhline(0, color='k', lw=0.6); ax.axvspan((x0 - 0.5 - lee) / H, 0, color='0.8')
        ax.axvline(XrH, color='C3', ls='--', lw=1, label=f'Xr/H = {XrH:.2f}')
        ax.axvspan(1.4, 1.8, color='C2', alpha=0.15, label='gate band 1.4-1.8')
        ax.set_xlim(-3, 6); ax.set_xlabel('(x - leeward face) / H'); ax.set_ylabel('mean u_x at z=1 cell (m/s)')
        ax.set_title('Gate 6b: near-ground centreline'); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(out, 'gate6b_centreline.png'), dpi=120)
    except Exception as e:
        print(f"[6b] (plot skipped: {e})")
    return ok


def main():
    if len(sys.argv) != 4 or sys.argv[1] not in ('abl', 'cube'):
        print(__doc__); return 2
    case, geom, out = sys.argv[1:]
    m, dxm = read5(os.path.join(geom, 'material_map.dat'), np.int32)
    u, dx = read5(os.path.join(out, 'uavg.f32'), np.float32)
    if m.shape[1:] != u.shape[1:] or abs(dx - dxm) > 1e-9:
        print("material map and uavg.f32 disagree on grid"); return 2
    mat = m[0]
    return 0 if (gate_abl if case == 'abl' else gate_cube)(mat, u, dx, out) else 1


if __name__ == '__main__':
    sys.exit(main())
