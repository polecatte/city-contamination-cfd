#!/usr/bin/env python3
"""showcase_report.py — one-page health check of a complete urban_flow city run.

    python3 tests/showcase_report.py GEOM_DIR OUT_DIR RUN_LOG

For a STEP4=1 run (flow spin-up + accidental release to clearance) it checks, end to end,
that every stage of the pipeline did its job, and writes OUT_DIR/SHOWCASE.txt:

  flow      the run finished, no divergence, peak lattice |u| < 0.1 (Mach < 0.17)
  wake      the city's wake closes before the outlet: < 2 % reversed near-ground flow over
            the last 15 % of the domain (the 15 H buffer doing its job)
  budget    released = deposited + airborne + drained, closure < 1 %
  clearance the burst ran until < 1 % of the release was still airborne (else J is truncated)
  J         population exposure J = <w, Theta> / M_released finite and positive
  Theta     negative-Theta mass (a numerical-diffusion artefact) < 1 % of the positive mass
Exit 0 when all pass.
"""
import os
import re
import sys

import numpy as np


def read5(fn, dtype):
    with open(fn, "rb") as f:
        h = np.frombuffer(f.read(20), np.int32)
        a = np.frombuffer(f.read(), dtype)
    return h, a


def meta(path):
    d = {}
    for line in open(path):
        p = line.split()
        if len(p) == 2 and not p[0].startswith("#"):
            try:
                d[p[0]] = float(p[1])
            except ValueError:
                pass
    return d


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        return 2
    geom, out, logf = sys.argv[1:4]
    log = open(logf, errors="replace").read()
    rows, ok_all = [], True

    def row(name, ok, text):
        nonlocal ok_all
        ok_all = ok_all and ok
        rows.append(f"  {'PASS' if ok else 'FAIL'}  {name:<10s} {text}")

    # flow
    peak = re.findall(r"peak lattice \|u\| over run = ([0-9.eE+-]+)", log)
    diverged = "DIVERGED" in log
    pk = float(peak[-1]) if peak else float("nan")
    row("flow", bool(peak) and not diverged and pk < 0.1,
        f"peak lattice |u| {pk:.3f} (Mach {pk*3**0.5:.3f})" + ("; DIVERGED" if diverged else ""))

    # wake
    h, u = read5(os.path.join(out, "umean_full.f32"), np.float32)
    nx, ny, nz = (int(v) for v in h[:3])
    ux = u[: nx * ny * nz].reshape(nz, ny, nx)
    tail = ux[1, 5:-5, int(0.85 * nx):-2]
    rev = float((tail < 0).mean())
    row("wake", rev < 0.02, f"reversed near-ground flow over the last 15 % of x: {100*rev:.1f} %")

    # budget, clearance, J
    md = meta(os.path.join(out, "meta_flow.txt"))
    emit = md.get("mass_emitted_total", md["mass_emitted"])
    clo = md.get("budget_closure", float("nan"))
    row("budget", clo < 0.01, f"closure {clo:.2e}  (emitted {emit:.4g}: deposited "
        f"{100*(md['mass_deposited']+md.get('mass_deposited_beyond_cv',0))/emit:.2f} %, "
        f"drained {100*md['mass_drained']/emit:.2f} %)")
    air = md["mass_airborne"] / emit
    row("clearance", air < 0.01, f"{100*air:.2f} % of the release still airborne at the end "
        f"({int(md['burst_steps'])} burst steps = {md['burst_steps']*md['dt_s']:.0f} s)")
    hw, w = read5(os.path.join(geom, "receptor_w.f32"), np.float32)
    ht, th = read5(os.path.join(out, "theta.f32"), np.float32)
    J = float(np.dot(w.astype(np.float64), th.astype(np.float64)) / emit)
    row("J", np.isfinite(J) and J > 0, f"J = {J:.6e} (population dose per unit released mass, lattice units)")
    neg = float(-th[th < 0].sum() / max(th[th > 0].sum(), 1e-30))
    row("Theta", neg < 0.01, f"negative-Theta mass {100*neg:.3f} % of positive")

    mlups = re.findall(r"average MLUPs :\s*([0-9.]+)", log)
    plat = re.findall(r"lattice platform: ([A-Z_]+)", log)
    head = (f"urban_flow showcase: {out}  grid {nx}x{ny}x{nz} = {nx*ny*nz/1e6:.2f} M cells, "
            f"platform {plat[-1] if plat else '?'}, {mlups[-1] if mlups else '?'} MLUPs")
    text = "\n".join([head] + rows + [f"  => {'ALL PASS' if ok_all else 'SOME CHECKS FAILED'}"])
    print(text)
    with open(os.path.join(out, "SHOWCASE.txt"), "w") as f:
        f.write(text + "\n")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
