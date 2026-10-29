#!/usr/bin/env python3
"""plume_check.py — the scalar transport against the exact point-source solution.

    python3 tests/plume_check.py GEOM_DIR OUT_DIR --src X,Y,Z --U 4 [--band strict|loose]

The case (lab_openlb.sh plume): an empty box (gen_gate6_geom CASE=box, no deposition), a frozen
uniform wind U along +x (STEP4_UNIFORM_U), a short puff from the single cell SRC (SRC_CELLS), run
to 99.9 % clearance. Theta = integral of C dt from a puff of mass M is exactly the STEADY field of
a continuous source of rate M, which for constant D in a uniform wind is (no plume approximation)

    theta(r) = M / (4 pi D |r|) * exp(-U (|r| - dx_s) / (2 D)),   dx_s = x - x_s

the Gaussian plume being its far-field limit (sigma_y = sigma_z = sqrt(2 D dx_s / U)). The
zero-flux walls (ground, lateral slip faces, lid; bounce-back half-way between the wall cell
and the first fluid cell) are added as image sources. Lattice units: Theta_lattice =
dx^3 * theta, M = mass_emitted_total, D = D_eff_m2s (both from meta_flow.txt).

Checks, at every cross-section from MIN_DX cells downstream of the source to 6 cells before the
outlet:
  mass     sum over the cross-section of Theta * U / dx = M  (all of the puff passes every plane;
           exact for any D, so this tests advection and conservation, not diffusion)
  width    sigma_y and sigma_z of Theta in the cross-section against the same moments of the
           exact field on the fluid cells (sigma ~ sqrt(D): tests the diffusivity the lattice
           actually applies; a 10 % error in D, or a D_eff conversion error like defect S3,
           fails the strict band)
  centre   Theta on the source line over the exact value (worst next to the source, where a
           one-cell puff is not a point source)
Bands: strict (resolved plume, cell Peclet ~4): mass 2 %, width 5 %, centre 15 %; loose (the
production D, cell Peclet ~13, under-resolved near the source): mass 3 %, width 10 %, centre 40 %.
Calibration: a numpy D3Q7 BGK with OpenLB's equilibrium, full-way bounce-back walls and this set-up
gives mass 0.0 / 0.6 %, width 0.4 / 0.5 %, centre 10.7 / 25.8 % (strict / production D).
Exit 0 = PASS, 1 = FAIL, 2 = input problem. Writes plume_check.png next to theta.f32.
"""
import sys, os, math, argparse
import numpy as np

MAT_FLUID, MAT_GROUND, MAT_SLIP = 1, 7, 5
BANDS = {"strict": dict(mass=0.02, width=0.05, centre=0.15, min_dx=8),
         "loose":  dict(mass=0.03, width=0.10, centre=0.40, min_dx=8)}


def read5(fn, dtype):
    with open(fn, "rb") as f:
        h = np.frombuffer(f.read(20), dtype=np.int32)
        a = np.frombuffer(f.read(), dtype=dtype)
    nx, ny, nz, dxmm, nc = (int(v) for v in h)
    return a.reshape(nc, nz, ny, nx)[0].transpose(2, 1, 0), dxmm / 1000.0     # [x, y, z]


def meta(path):
    d = {}
    for line in open(path):
        p = line.split()
        if len(p) == 2 and not p[0].startswith("#"):
            try: d[p[0]] = float(p[1])
            except ValueError: d[p[0]] = p[1]
    return d


def images(s, lo, hi, n=3):
    """positions of the source s and its images between zero-flux planes lo < hi (index units)"""
    L = hi - lo
    return [s + 2 * k * L for k in range(-n, n + 1)] + [2 * lo - s + 2 * k * L for k in range(-n, n + 1)]


def exact(shape, src, planes, U, D, dx, M):
    """Theta_lattice on the cell centres: dx^3 * sum over images of M/(4 pi D r) exp(-U(r - dX)/2D)"""
    nx, ny, nz = shape
    X = (np.arange(nx) - src[0]) * dx
    th = np.zeros(shape)
    for ys in images(src[1], *planes["y"]):
        for zs in images(src[2], *planes["z"]):
            Y = (np.arange(ny) - ys) * dx; Z = (np.arange(nz) - zs) * dx
            r = np.sqrt(X[:, None, None] ** 2 + Y[None, :, None] ** 2 + Z[None, None, :] ** 2)
            r = np.maximum(r, 0.5 * dx)                       # the source cell itself is not compared
            th += M / (4 * math.pi * D * r) * np.exp(-U * (r - X[:, None, None]) / (2 * D))
    return th * dx ** 3


def moments(a, axis_len, axis):
    """centroid and sigma (cells) of a cross-section's Theta along one axis"""
    p = a.sum(axis=1 - axis) if a.ndim == 2 else a
    i = np.arange(axis_len); s = p.sum()
    m = (p * i).sum() / s
    return m, math.sqrt(max((p * (i - m) ** 2).sum() / s, 0.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("geom"); ap.add_argument("out")
    ap.add_argument("--src", required=True, help="source cell x,y,z (as SRC_CELLS)")
    ap.add_argument("--U", type=float, required=True, help="the uniform wind, m/s (STEP4_UNIFORM_U)")
    ap.add_argument("--band", choices=list(BANDS), default="strict")
    a = ap.parse_args()
    src = tuple(int(v) for v in a.src.split(","))
    th, dx = read5(os.path.join(a.out, "theta.f32"), np.float32)
    mat, dxm = read5(os.path.join(a.geom, "material_map.dat"), np.int32)
    if th.shape != mat.shape or abs(dx - dxm) > 1e-9:
        print("theta.f32 and material_map.dat disagree on the grid"); return 2
    m = meta(os.path.join(a.out, "meta_flow.txt"))
    M, D = m.get("mass_emitted_total", m.get("mass_emitted")), m.get("D_eff_m2s")
    if not (M and D):
        print("meta_flow.txt lacks mass_emitted_total or D_eff_m2s"); return 2
    if m.get("mass_deposited", 0.0) > 1e-6 * M:
        print(f"[plume] WARNING: {m['mass_deposited']:.3g} of {M:.3g} deposited: run with VD_SCALE=0")
    nx, ny, nz = th.shape
    # zero-flux planes half-way between the wall cells and the first fluid cells
    gz = [z for z in range(nz) if np.all(mat[nx // 2, ny // 2, z] == MAT_GROUND)]
    top = [z for z in range(nz) if mat[nx // 2, ny // 2, z] == MAT_SLIP]
    side = [y for y in range(ny) if mat[nx // 2, y, nz // 2] == MAT_SLIP]
    planes = {"z": (max(gz) + 0.5 if gz else -0.5, min(top) - 0.5 if top else nz - 0.5),
              "y": (min(side) + 0.5 if side and min(side) < ny // 2 else -0.5,
                    max(side) - 0.5 if side and max(side) > ny // 2 else ny - 0.5)}
    U = a.U
    ex = exact(th.shape, src, planes, U, D, dx, M)
    ex[mat != MAT_FLUID] = 0.0          # Theta accumulates on fluid cells only (urban_flow Step 4)
    B = BANDS[a.band]
    Pe = U * dx / D
    print(f"[plume] grid {nx}x{ny}x{nz} dx={dx} m, source cell {src}, U={U} m/s, D={D:.4g} m^2/s "
          f"(cell Peclet {Pe:.1f}), M={M:.6g}, walls z {planes['z']}, y {planes['y']} (cells)")
    xs = list(range(src[0] + B["min_dx"], nx - 6))
    rows = []
    for x in xs:
        num, ref = th[x], ex[x]
        mass = num.sum() * U / dx / M
        sy, sz = moments(num, ny, 0)[1], moments(num.T, nz, 0)[1]
        ry, rz = moments(ref, ny, 0)[1], moments(ref.T, nz, 0)[1]
        cen = num[src[1], src[2]] / ref[src[1], src[2]]
        rows.append((x, mass, sy / ry, sz / rz, cen, ry))
    print(f"[plume] {'x-xs':>5s} {'sigma_y exact':>13s} {'mass':>7s} {'sig_y':>7s} {'sig_z':>7s} {'centre':>7s}   (numerical / exact)")
    for r in rows[::max(1, len(rows) // 12)]:
        print(f"[plume] {r[0] - src[0]:5d} {r[5] * dx:11.1f} m {r[1]:7.3f} {r[2]:7.3f} {r[3]:7.3f} {r[4]:7.3f}")
    err = lambda k: max(abs(r[k] - 1) for r in rows)
    em, ey, ez, ec = err(1), err(2), err(3), err(4)
    checks = [("mass through each cross-section", em, B["mass"]), ("plume width sigma_y", ey, B["width"]),
              ("plume width sigma_z", ez, B["width"]), ("centre-line Theta", ec, B["centre"])]
    ok = True
    for name, e, lim in checks:
        good = e <= lim; ok &= good
        print(f"[plume] {name:32s} worst {100 * e:5.1f} %  (band {100 * lim:.0f} %)  {'PASS' if good else 'FAIL'}")
    print(f"[plume] GATE plume ({a.band}): {'PASS' if ok else 'FAIL'}")
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
        xd = (np.arange(nx) - src[0]) * dx
        ax[0].semilogy(xd, th[:, src[1], src[2]], label="lattice"); ax[0].semilogy(xd, ex[:, src[1], src[2]], "k--", label="exact")
        ax[0].set_xlabel("x - x_s (m)"); ax[0].set_ylabel("Theta on the source line"); ax[0].legend()
        xm = src[0] + (nx - 6 - src[0]) // 2
        yd = (np.arange(ny) - src[1]) * dx
        ax[1].plot(yd, th[xm, :, src[2]], label="lattice"); ax[1].plot(yd, ex[xm, :, src[2]], "k--", label="exact")
        ax[1].set_xlabel(f"y - y_s (m) at x - x_s = {(xm - src[0]) * dx:.0f} m"); ax[1].legend()
        fig.suptitle(f"plume: D = {D:.3g} m^2/s, U = {U} m/s, dx = {dx} m (cell Peclet {Pe:.1f})")
        fig.tight_layout(); fig.savefig(os.path.join(a.out, "plume_check.png"), dpi=110)
    except Exception as e:  # a figure is a convenience, never the gate
        print(f"[plume] (plot skipped: {e})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
