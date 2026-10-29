#!/usr/bin/env python3
"""builder_survey.py — what the parameter space builds, and how smoothly.

    python3 tests/builder_survey.py GEN_BINARY [N_SOBOL] [OUT_PNG]

GEN_BINARY is a gen_openlb_geom build (dx 4 recommended). Uses its STATS_ONLY mode (layout
only, ~10 ms per design), so it runs in seconds.

1. Survey: N_SOBOL scrambled-Sobol designs over param_space_density's bounds, builder
   revision 9 with the default height limit, and again with no limit: how often a design
   is infeasible, and the distribution of the tallest building (what the domain must hold).
2. Smoothness: one-parameter sweeps (others at mid-range) for revision 9 and the legacy
   builder (BUILDER=legacy). For each output (plan-area fraction lambda_p, frontal-area
   index lambda_f, floor-area-weighted height, tallest building) the largest jump between
   neighbouring sweep points, as a share of that output's range over the sweep.
"""
import os, sys, subprocess
import numpy as np
from scipy.stats import qmc

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import param_space_density as P

GEN = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else None
NS = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
PNG = sys.argv[3] if len(sys.argv) > 3 else None
KEYS = ["feasible", "buildings", "max_h", "fa_weighted_h", "lambda_p", "lambda_f", "gfa", "height_scale"]


def stats(phys, extra=None):
    env = dict(os.environ); env.update(P.gen_env(phys)); env["STATS_ONLY"] = "1"
    if extra: env.update(extra)
    out = subprocess.run([GEN], env=env, capture_output=True, text=True).stdout
    line = [l for l in out.splitlines() if l.startswith("STATS")]
    if not line: return None
    d = dict(kv.split("=") for kv in line[0].split()[1:])
    return {k: float(d[k]) for k in KEYS if k in d}


def survey():
    X = qmc.Sobol(len(P.PARAM_SPACE_DENSITY), scramble=True, seed=3).random(NS)
    res = {}
    for label, extra in (("limit 100 m", {}), ("no limit", {"H_MAX": "1e6"})):
        rows = [stats(P.to_physical_density(x), extra) for x in X]
        rows = [r for r in rows if r]
        feas = np.array([r["feasible"] for r in rows]); mh = np.array([r["max_h"] for r in rows])
        lp = np.array([r["lambda_p"] for r in rows]); lf = np.array([r["lambda_f"] for r in rows])
        print(f"[survey] {label}: {len(rows)} designs, infeasible {100*(1-feas.mean()):.1f} %; tallest building "
              f"p50 {np.percentile(mh,50):.0f} m, p95 {np.percentile(mh,95):.0f} m, p99 {np.percentile(mh,99):.0f} m, "
              f"max {mh.max():.0f} m; lambda_p {lp.min():.2f}-{lp.max():.2f}, lambda_f {lf.min():.2f}-{lf.max():.2f}")
        res[label] = mh
    return res


SWEEPS = {"block_w": 160, "street_width": 120, "park_fraction": 120, "roughness": 80, "height_concentration": 100}
OUTS = ["lambda_p", "lambda_f", "fa_weighted_h", "max_h"]


def sweeps():
    names = P.NAMES_DENSITY
    base = np.full(len(names), 0.5)
    results = {}
    for pname, n in SWEEPS.items():
        i = names.index(pname)
        for label, extra in (("rev 9", {}), ("legacy", {"BUILDER": "legacy"})):
            vals, rows = [], []
            for u in np.linspace(0, 1, n):
                x = base.copy(); x[i] = u
                ph = P.to_physical_density(x)
                if pname == "block_w": ph["block_d"] = ph["block_w"]          # square blocks
                ex = dict(extra)
                # the legacy builder's counterpart of height_concentration is PEAK_H 30-90 m
                if label == "legacy" and pname == "height_concentration": ex["PEAK_H"] = f"{30 + 60*u:.4g}"
                r = stats(ph, ex)
                if r: vals.append(ph[pname]); rows.append(r)
            results[(pname, label)] = (np.array(vals), rows)
            jumps = []
            for k in OUTS:
                y = np.array([r[k] for r in rows]); rng = y.max() - y.min()
                jumps.append(np.abs(np.diff(y)).max() / rng if rng > 0 else 0.0)
            print(f"[smooth] {pname:13s} {label:7s} largest jump / range: " +
                  ", ".join(f"{k} {100*j:4.0f} %" for k, j in zip(OUTS, jumps)))
    return results


def plot(sw, png):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(len(OUTS), len(SWEEPS), figsize=(3.2 * len(SWEEPS), 2.4 * len(OUTS)), squeeze=False)
    for j, pname in enumerate(SWEEPS):
        for label, c in (("legacy", "#b0b0b0"), ("rev 9", "#1f6fb4")):
            v, rows = sw[(pname, label)]
            for i, k in enumerate(OUTS):
                ax[i, j].plot(v, [r[k] for r in rows], color=c, lw=1.4, label=label)
        ax[-1, j].set_xlabel(pname)
    for i, k in enumerate(OUTS): ax[i, 0].set_ylabel(k)
    ax[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(png, dpi=120)
    print(f"[smooth] figure: {png}")


if __name__ == "__main__":
    if not GEN:
        print(__doc__); sys.exit(2)
    survey()
    sw = sweeps()
    if PNG: plot(sw, PNG)
