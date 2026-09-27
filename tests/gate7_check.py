#!/usr/bin/env python3
"""gate7_check.py — Phase-6 scalar gates 7a and 7c from urban_flow's meta_flow.txt.

    python3 tests/gate7_check.py budget OUT_DIR            # 7a: mass budget closes < 1 %
    python3 tests/gate7_check.py nodep  OUT_DIR            # 7c: v_d = w_s = 0 -> nothing deposits,
                                                           #     everything leaves downstream
    python3 tests/gate7_check.py dep    OUT_DIR GEOM_DIR   # 7c: deposited fraction vs analytic

(Gate 7b, linearity, is tests/linearity_guard.cpp.)

The budget is MEASURED by the solver (exact lattice flux across the control volume's faces,
see urban_flow.cpp STEP 4), so closure tests mass conservation of the AD lattice and its
boundaries; it is not an identity.

The `dep` reference: a ground-level release in a uniform wind U over a floor with deposition
velocity v_d, diffusivity D. Following a parcel (x = U t), deposition is a 1-D problem in z:
diffusion above a partially absorbing wall, -D dC/dz = -v_d C at z = 0, for the residence time
t_res = (x_cv_end - x_source)/U. The reference is that problem solved on a fine grid (dz = 5 cm)
with the release spread over the lattice's first cell (0..dx above the half-way wall, which is
where the lattice puts it). For a release AT the wall it reduces to the closed form
    F = 1 - exp(k^2 t) erfc(k sqrt t),   k = v_d / sqrt(D),
printed as a check on the solver. (Using that closed form as the reference was wrong by ~25 %:
it is a surface release, and the lattice's release sits 0..4 m up.) What is left between the
lattice and the reference is first-cell resolution at dx = 4 m, so the band is 25 %; a units
error in v_d (defect S2, 80x) cannot pass it.
"""
import sys, os, math
import numpy as np


def meta(out):
    d = {}
    with open(os.path.join(out, 'meta_flow.txt')) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            k, *v = line.split()
            try:
                d[k] = float(v[0]) if len(v) == 1 else [float(x) for x in v]
            except ValueError:
                d[k] = ' '.join(v)
    return d


def budget(out):
    m = meta(out)
    e, dep, air, dr = m['mass_emitted'], m['mass_deposited'], m['mass_airborne'], m['mass_drained']
    resid = abs(e - (dep + air + dr)) / e
    print(f"[7a] emitted {e:.6g} = deposited {dep:.6g} + airborne {air:.6g} + drained {dr:.6g}"
          f" (down {m['mass_out_downstream']:.6g}, up {m['mass_out_upstream']:.6g})")
    print(f"[7a] closure |emit-(dep+air+out)|/emit = {resid:.3e}  -> {'PASS' if resid < 0.01 else 'FAIL'} (< 1%)")
    return resid < 0.01


def nodep(out):
    m = meta(out)
    e = m['mass_emitted']
    dep = m['mass_deposited'] + m['mass_deposited_beyond_cv']
    down = m['mass_out_downstream'] / e
    ok_dep = dep == 0.0
    ok_out = down > 0.99
    print(f"[7c] v_d=0, w_s=0: deposited = {dep!r} (must be exactly 0) -> {'PASS' if ok_dep else 'FAIL'}")
    print(f"[7c] left through the outlet side: {100*down:.3f}% of emitted, airborne {100*m['mass_airborne']/e:.3f}%,"
          f" upstream {100*m['mass_out_upstream']/e:.3f}% -> {'PASS' if ok_out else 'FAIL'} (> 99%)")
    return ok_dep and ok_out


def dep(out, geom):
    m = meta(out)
    e = m['mass_emitted']
    F = m['mass_deposited'] / e
    D = m['D_eff_m2s']
    dx = m['dx_m']
    U = float(os.environ.get('STEP4_UNIFORM_U', '4'))
    vdf = os.path.join(geom, 'dep_vel.f32')
    with open(vdf, 'rb') as f:
        h = np.frombuffer(f.read(20), np.int32)
        vd = np.frombuffer(f.read(), np.float32)
    vd_ground = float(vd[vd > 0].max()) * float(os.environ.get('VD_SCALE', '1'))
    with open(os.path.join(geom, 'source_mask.u8'), 'rb') as f:
        f.read(20); src = np.frombuffer(f.read(), np.uint8).reshape(h[2], h[1], h[0])
    xs = np.nonzero(src)[2].mean()
    xcv = h[0] - 3 + 0.5                             # downstream CV face
    t = (xcv - xs) * dx / U
    k = vd_ground / math.sqrt(D)
    a = k * math.sqrt(t)
    Fs = 1.0 - math.exp(a * a) * math.erfc(a)
    Fr = deposit_1d(vd_ground, D, t, 0.0, dx)
    Fw = deposit_1d(vd_ground, D, t, 0.0, 0.05)
    rel = (F - Fr) / Fr
    ok = abs(rel) < 0.25
    print(f"[7c] v_d={vd_ground:g} m/s, D={D:.4g} m^2/s, U={U:g} m/s, residence {t:.1f} s")
    print(f"[7c] solver check: 1-D release at the wall {100*Fw:.3f}% vs closed form {100*Fs:.3f}%")
    print(f"[7c] deposited fraction in CV: lattice {100*F:.3f}%  reference (release in first cell) {100*Fr:.3f}%"
          f"  ({100*rel:+.1f}%) -> {'PASS' if ok else 'FAIL'} (within 25%)")
    return ok


def deposit_1d(vd, D, t_end, z_lo, z_hi, dz=0.05):
    """Fraction deposited by t_end: unit mass spread over [z_lo, z_hi] above a wall with
    deposition velocity vd, diffusivity D, explicit finite volumes on a fine grid."""
    H = max(8.0 * math.sqrt(D * t_end), 4.0 * z_hi)
    n = int(H / dz)
    z = (np.arange(n) + 0.5) * dz
    m = (z >= z_lo) & (z <= max(z_hi, z_lo + dz))
    C = np.where(m, 1.0 / (m.sum() * dz), 0.0)
    dt = 0.2 * dz * dz / D
    nt = int(t_end / dt)
    dep = 0.0
    for _ in range(nt):
        fl = D * np.diff(C) / dz
        dC = np.zeros(n); dC[:-1] += fl; dC[1:] -= fl
        w = vd * C[0]; dC[0] -= w; dep += w * dt
        C += dt * dC / dz
    return dep


if __name__ == '__main__':
    if len(sys.argv) < 3 or sys.argv[1] not in ('budget', 'nodep', 'dep'):
        print(__doc__); sys.exit(2)
    mode, out = sys.argv[1], sys.argv[2]
    ok = budget(out) if mode == 'budget' else nodep(out) if mode == 'nodep' else dep(out, sys.argv[3])
    sys.exit(0 if ok else 1)
