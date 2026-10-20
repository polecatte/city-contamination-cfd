#!/usr/bin/env python3
"""pathways.py — who breathes the dose: J split by where people are (indoor / park / street).

    python3 tests/pathways.py RX_GEOM RUN_DIR [RUN_DIR ...] [--legacy GEOM_DIR]

RX_GEOM : a geometry written with RECEPTOR_PARTS=1 (receptor_w_indoor/_park/_street.f32 and
          meta_geom.txt), the same grid as the runs. The runs' Theta does not depend on the
          receptor weights, so finished runs are re-scored without new CFD.
--legacy: the geometry the runs were made with; its receptor_w.f32 gives J under the old
          (pre-fix) weights, for comparison.

For each run: the intake fraction (ppm, iF = B J / dx^3) in total and per pathway, each
pathway's share of person-time and of dose, and the dose per person-hour relative to indoors
(how much more a street hour costs than an hour at home or at work).
"""
import os, sys
import numpy as np

B = 1.6e-4   # m^3/s


def read5(fn):
    b = open(fn, "rb").read()
    h = np.frombuffer(b[:20], np.int32)
    return h, np.frombuffer(b[20:], np.float32)[: h[0] * h[1] * h[2]].astype(np.float64)


def meta(fn):
    m = {}
    for l in open(fn):
        p = l.split()
        if len(p) == 2:
            try: m[p[0]] = float(p[1])
            except ValueError: m[p[0]] = p[1]
    return m


def main(argv):
    if len(argv) < 2: print(__doc__); return 2
    legacy = None
    if "--legacy" in argv:
        i = argv.index("--legacy"); legacy = argv[i + 1]; argv = argv[:i] + argv[i + 2:]
    rx, runs = argv[0], argv[1:]
    mg = meta(os.path.join(rx, "meta_geom.txt"))
    finf = float(mg.get("f_inf", 0.62))
    parts = {k: read5(os.path.join(rx, f"receptor_w_{k}.f32")) for k in ("indoor", "park", "street")}
    hw = parts["indoor"][0]
    pt = {"indoor": parts["indoor"][1].sum() / finf, "park": parts["park"][1].sum(), "street": parts["street"][1].sum()}
    ptot = sum(pt.values())
    wl = read5(os.path.join(legacy, "receptor_w.f32"))[1] if legacy else None
    rows = []
    for run in runs:
        ht, th = read5(os.path.join(run, "theta.f32"))
        if tuple(ht[:3]) != tuple(hw[:3]):
            print(f"[path] {run}: grid {ht[:3]} != receptor grid {hw[:3]}"); return 2
        mf = meta(os.path.join(run, "meta_flow.txt"))
        M = mf.get("mass_emitted_total", mf["mass_emitted"]); dx = float(mf.get("dx_m", hw[3] / 1000))
        iF = {k: B * float(np.dot(w, th)) / M / dx ** 3 * 1e6 for k, (h, w) in parts.items()}
        tot = sum(iF.values())
        line = f"[path] {os.path.basename(run.rstrip('/'))}: intake fraction {tot:.2f} ppm with the corrected weights"
        if wl is not None:
            line += f" (old weights: {B * float(np.dot(wl, th)) / M / dx ** 3 * 1e6:.2f} ppm)"
        print(line)
        print(f"[path]   {'pathway':8s} {'person-time':>12s} {'dose':>8s} {'dose per person-hour':>22s}")
        per_in = iF["indoor"] / pt["indoor"]
        for k in ("indoor", "park", "street"):
            print(f"[path]   {k:8s} {100 * pt[k] / ptot:11.1f}% {100 * iF[k] / tot:7.1f}% "
                  f"{(iF[k] / pt[k]) / per_in:17.2f} x indoor")
        rows.append((tot, iF))
    if len(rows) > 1:
        t = np.array([r[0] for r in rows])
        print(f"[path] across the {len(rows)} runs: intake fraction {t.mean():.2f} ppm, spread {100 * (t.max() - t.min()) / t.mean():.2f} %")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
