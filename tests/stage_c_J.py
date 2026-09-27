#!/usr/bin/env python3
"""stage_c_J.py — Stage-C contraction J and the Gate-8 seed-stability check.

    python3 tests/stage_c_J.py GEOM_DIR OUT_DIR [OUT_DIR2 ...]

J = <w, Theta> / M_emitted, with w = GEOM_DIR/receptor_w.f32 (gen_openlb_geom.cpp) and
Theta = OUT_DIR/theta.f32 (urban_flow STEP4=1). Every Omega cell releases the same mass, so
this is forward_city.cpp's <w,Theta>/|Omega| for a unit release per source cell: population
dose per unit released mass. Lattice concentration units; only ratios between runs matter.

Gate 8 (OPENLB_PORT_STATUS_AND_VERIFICATION.md Phase 7): J finite and positive for every run,
and, given two or more runs of the same city with different inlet seeds (ABL_SEED), the
spread max|J_i - J_mean| / J_mean within a few percent (5 % here). Exit 0 = PASS.
"""
import sys, os
import numpy as np


def read5(fn, dtype):
    with open(fn, 'rb') as f:
        h = np.frombuffer(f.read(20), dtype=np.int32)
        a = np.frombuffer(f.read(), dtype=dtype)
    return h, a


def meta(out):
    d = {}
    for line in open(os.path.join(out, 'meta_flow.txt')):
        p = line.split()
        if len(p) == 2 and not p[0].startswith('#'):
            try: d[p[0]] = float(p[1])
            except ValueError: pass
    return d


def main():
    if len(sys.argv) < 3:
        print(__doc__); return 2
    geom, outs = sys.argv[1], sys.argv[2:]
    hw, w = read5(os.path.join(geom, 'receptor_w.f32'), np.float32)
    Js = []
    for out in outs:
        ht, th = read5(os.path.join(out, 'theta.f32'), np.float32)
        if tuple(ht[:3]) != tuple(hw[:3]):
            print(f"{out}: theta grid {ht[:3]} != receptor grid {hw[:3]}"); return 2
        m = meta(out)
        # all released mass: runs after the CV/Omega split report it as mass_emitted_total
        emit = m.get('mass_emitted_total', m['mass_emitted'])
        J = float(np.dot(w.astype(np.float64), th.astype(np.float64)) / emit)
        neg = float(-th[th < 0].sum() / max(th[th > 0].sum(), 1e-30))
        print(f"[J] {out}: J = {J:.6e}   emitted {emit:.4g}, deposited {100*m['mass_deposited']/emit:.2f}%,"
              f" budget closure {m.get('budget_closure', float('nan')):.2e}, burst steps {int(m['burst_steps'])},"
              f" negative-Theta mass {100*neg:.2f}%")
        Js.append(J)
    ok = all(np.isfinite(J) and J > 0 for J in Js)
    print(f"[8] J finite and positive: {'PASS' if ok else 'FAIL'}")
    if len(Js) > 1:
        Jm = float(np.mean(Js)); spread = max(abs(J - Jm) for J in Js) / Jm
        sok = spread < 0.05
        print(f"[8] seed spread max|J-mean|/mean = {100*spread:.2f}%  -> {'PASS' if sok else 'FAIL'} (< 5%)")
        ok = ok and sok
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
