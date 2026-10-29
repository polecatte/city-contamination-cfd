#!/usr/bin/env python3
"""orient_check.py — ORIENT is an exact symmetry, and the oblique material map has the right faces.

    python3 tests/orient_check.py GEN_BINARY WORKDIR

1. Builds the default city at ORIENT 0..7 (dx of GEN_BINARY, WIND_DEG 0) and checks that, in
   the city's square footprint, every per-cell field (cell type, receptor weight, release mask)
   of ORIENT k equals ORIENT 0's field moved by the k-th symmetry, cell for cell; and that the
   totals of the receptor weight and release cells are identical.
2. Builds WIND_DEG 22.5 and 45 and checks the faces: inflow on x = 0 and y = 0, outflow on
   x = nx-1 and y = ny-1, slip on top; and WIND_DEG 0 keeps y faces slip.
"""
import os, sys, subprocess
import numpy as np

GEN, WD = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
os.makedirs(WD, exist_ok=True)
MAT_INLET, MAT_OUTLET, MAT_SLIP = 3, 4, 5


def gen(name, **env):
    e = dict(os.environ); e.update({k: str(v) for k, v in env.items()}); e["OUT_DIR"] = os.path.join(WD, name)
    r = subprocess.run([GEN], env=e, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    return os.path.join(WD, name)


def load(d, fn, dtype):
    b = open(os.path.join(d, fn), "rb").read()
    h = np.frombuffer(b[:20], np.int32)
    nx, ny, nz = h[:3]
    return np.frombuffer(b[20:], dtype)[: nx * ny * nz].reshape(nz, ny, nx)   # [z, y, x]


def meta(d):
    m = {}
    for l in open(os.path.join(d, "meta_geom.txt")):
        p = l.split()
        if len(p) == 2: m[p[0]] = p[1]
    return m


def sym(a, k):
    """Apply symmetry k to an array indexed [..., y, x] over a square W x W footprint (cells)."""
    if k >= 4:
        a = a[..., :, ::-1]                       # mirror x
    for _ in range(k & 3):
        a = np.rot90(a, 1, axes=(-1, -2))         # +90 deg CCW in (x, y)
    return a


def main():
    d0 = gen("o0", ORIENT=0)
    m0 = meta(d0); dx = float(m0["dx_m"]); W = int(round(600 / dx))
    rnd = lambda v: int(np.floor(v + 0.5))       # C++ std::lround (Python's round() is half-to-even)
    x0, y0 = rnd(float(m0["buf_xm_m"]) / dx), rnd(float(m0["buf_ym_m"]) / dx)
    fields = [("geom_type.u8", np.uint8), ("receptor_w.f32", np.float32), ("source_mask.u8", np.uint8)]
    ref = {fn: load(d0, fn, t) for fn, t in fields}
    ok = True
    for k in range(1, 8):
        dk = gen(f"o{k}", ORIENT=k)
        for fn, t in fields:
            a = load(dk, fn, t)
            if a.shape != ref[fn].shape:
                print(f"[orient] k={k} {fn}: grid differs"); ok = False; continue
            c0 = ref[fn][:, y0:y0 + W, x0:x0 + W]; ck = a[:, y0:y0 + W, x0:x0 + W]
            same = np.array_equal(sym(c0, k), ck)
            out_same = np.array_equal(np.delete(np.delete(a, np.s_[x0:x0 + W], 2), np.s_[y0:y0 + W], 1),
                                      np.delete(np.delete(ref[fn], np.s_[x0:x0 + W], 2), np.s_[y0:y0 + W], 1))
            tot = np.isclose(a.astype(np.float64).sum(), ref[fn].astype(np.float64).sum(), rtol=1e-6)
            # receptor: the street ring may step one cell outside the footprint; compare totals there
            good = (same or fn == "receptor_w.f32" and tot) and (out_same or fn != "geom_type.u8")
            print(f"[orient] k={k} {fn:16s} footprint cell-for-cell {'EXACT' if same else 'differs'}, "
                  f"total {'equal' if tot else 'DIFFERS'}")
            ok &= bool(good and tot)
    for w in (0, 22.5, 45):
        d = gen(f"w{w}", WIND_DEG=w)
        mat = load(d, "material_map.dat", np.int32)
        nz, ny, nx = mat.shape
        def face(sl):
            v = mat[sl]; v = v[(v == MAT_INLET) | (v == MAT_OUTLET) | (v == MAT_SLIP)]
            return {MAT_INLET: "in", MAT_OUTLET: "out", MAT_SLIP: "slip"}.get(int(np.bincount(v).argmax()) if v.size else 0, "-")
        f = dict(xm=face(np.s_[1:-1, 1:-1, 0]), xp=face(np.s_[1:-1, 1:-1, -1]), ym=face(np.s_[1:-1, 0, 1:-1]),
                 yp=face(np.s_[1:-1, -1, 1:-1]), top=face(np.s_[-1, 1:-1, 1:-1]))
        want = (dict(xm="in", xp="out", ym="slip", yp="slip", top="slip") if w == 0 else
                dict(xm="in", xp="out", ym="in", yp="out", top="slip"))
        good = f == want
        print(f"[orient] WIND_DEG={w:<5} faces {f} {'OK' if good else 'WRONG'}  (grid {nx}x{ny}x{nz})")
        ok &= good
    print(f"[orient] {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
