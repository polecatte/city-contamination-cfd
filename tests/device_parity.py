#!/usr/bin/env python3
"""device_parity.py — compare two urban_flow output directories field by field.

    device_parity.py REF_DIR TEST_DIR TOL [--log-ref F --log-test F]

Used by tests/device_parity.sh to check the operator path (urban_ops.h) against the host
reference loops (CPU vs CPU), and the GPU build against the CPU build. For every output the
two runs both wrote it prints max|test-ref| / max|ref| (0 = identical), and fails when that
exceeds TOL:

    umean_full.f32   live velocity + nu_t (float32)          uavg.f32   time mean + variance
    theta.f32        Step-4 exposure integral                 deposition.f32
    exposure_timeseries.csv  every column, over the common steps
    meta_flow.txt    the mass budget (emitted / deposited / airborne / drained)

With --log-ref/--log-test it also compares the run logs' 'peak lattice |u|' and requires the
WALE_GRAD_CHECK line (when present) to say PASS in the test log.
"""
import os
import re
import sys

import numpy as np


def read5(path):
    b = open(path, "rb").read()
    h = np.frombuffer(b[:20], np.int32)
    nx, ny, nz, _, nc = (int(v) for v in h)
    a = np.frombuffer(b[20:], np.float32)
    return a[: nx * ny * nz * nc].astype(np.float64)


def read_csv(path):
    rows = [l.strip().split(",") for l in open(path) if l.strip()]
    head, data = rows[0], rows[1:]
    return head, {int(r[0]): np.array([float(v) for v in r[1:]]) for r in data}


def read_meta(path):
    out = {}
    for l in open(path):
        p = l.split()
        if len(p) == 2 and not p[0].startswith("#"):
            try:
                out[p[0]] = float(p[1])
            except ValueError:
                pass
    return out


def rel(a, b):
    scale = max(np.max(np.abs(a)) if a.size else 0.0, 1e-300)
    return float(np.max(np.abs(a - b)) / scale) if a.size else 0.0


def main():
    args = sys.argv[1:]
    logs = {}
    for k in ("--log-ref", "--log-test"):
        if k in args:
            i = args.index(k)
            logs[k] = args[i + 1]
            del args[i : i + 2]
    ref, test, tol = args[0], args[1], float(args[2])
    worst, checked, fails = 0.0, 0, []

    def report(name, r, n=""):
        nonlocal worst, checked
        checked += 1
        worst = max(worst, r)
        flag = "ok" if r <= tol else "FAIL"
        if r > tol:
            fails.append(name)
        print(f"  {name:<34s} rel.diff {r:10.3e}  {flag} {n}")

    for f in ("umean_full.f32", "uavg.f32", "theta.f32", "deposition.f32"):
        a, b = os.path.join(ref, f), os.path.join(test, f)
        if os.path.exists(a) and os.path.exists(b):
            x, y = read5(a), read5(b)
            if x.shape != y.shape:
                report(f, float("inf"), "(shape differs)")
                continue
            if not np.all(np.isfinite(y)):
                report(f, float("inf"), "(non-finite values in test)")
                continue
            report(f, rel(x, y), f"(max|ref| {np.max(np.abs(x)):.3e})")

    a, b = os.path.join(ref, "exposure_timeseries.csv"), os.path.join(test, "exposure_timeseries.csv")
    if os.path.exists(a) and os.path.exists(b):
        head, x = read_csv(a)
        _, y = read_csv(b)
        steps = sorted(set(x) & set(y))
        if not steps:
            report("exposure_timeseries.csv", float("inf"), "(no common steps)")
        else:
            X = np.array([x[s] for s in steps])
            Y = np.array([y[s] for s in steps])
            for j, name in enumerate(head[1:]):
                if name in ("t_s", "budget_resid"):
                    continue  # time is exact; the residual is a difference of large terms
                report(f"timeseries:{name}", rel(X[:, j], Y[:, j]), f"({len(steps)} rows)")

    a, b = os.path.join(ref, "meta_flow.txt"), os.path.join(test, "meta_flow.txt")
    if os.path.exists(a) and os.path.exists(b):
        x, y = read_meta(a), read_meta(b)
        for k in ("mass_emitted", "mass_deposited", "mass_airborne", "mass_drained", "burst_steps"):
            if k in x and k in y:
                report(f"meta:{k}", abs(x[k] - y[k]) / max(abs(x[k]), 1e-300))

    if logs:
        def peak(path):
            m = re.findall(r"peak lattice \|u\| over run = ([0-9.eE+-]+)", open(path, errors="replace").read())
            return float(m[-1]) if m else None
        pr, pt = peak(logs["--log-ref"]), peak(logs["--log-test"])
        if pr is not None and pt is not None:
            report("log:peak |u|", abs(pr - pt) / max(abs(pr), 1e-300))
        g = re.findall(r"WALE_GRAD_CHECK.*\[(PASS|FAIL)\]", open(logs["--log-test"], errors="replace").read())
        if g:
            checked += 1
            print(f"  {'log:WALE_GRAD_CHECK (test)':<34s} {g[-1]}")
            if g[-1] != "PASS":
                fails.append("WALE_GRAD_CHECK")

    if checked == 0:
        print("  nothing to compare (no common outputs)")
        return 1
    print(f"  worst {worst:.3e} vs tolerance {tol:.1e}: {'PASS' if not fails else 'FAIL ' + ', '.join(fails)}")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
