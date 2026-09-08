"""Continuity & extremes audit of the city builder.
For each of the 9 optimizer params, sweep [lo,hi] (log-spaced where the search
space says 'log'), holding the rest at canonical defaults, and measure the
response of every objective-relevant output. Flag discontinuities and dead
zones (regions where the objective is insensitive -> the optimizer wanders/
exploits)."""
import subprocess, io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import param_space as PS

# v8.3 baseline — the current 9-D search space (removed keys base_height,
# biz_inner_frac, coverage, target_density are gone; population is fixed).
DEFAULTS = dict(block_w=48, block_d=32, cbd_peak=45, cbd_decay=1.2e-5,
                patchiness=0.0, park_centrality=0.5, park_fraction=0.10,
                roughness=0.25, street_width=20)
N = 80

def run(param_lines):
    p = subprocess.run(["./probe"], input="\n".join(param_lines),
                       capture_output=True, text=True)
    return pd.read_csv(io.StringIO(p.stdout))

# Build one big batch: for each param, N samples across its range.
sweeps = {}
lines = []
index = []
for name, lo, hi, scale in PS.PARAM_SPACE:
    vals = (np.geomspace(lo, hi, N) if scale == "log" else np.linspace(lo, hi, N))
    sweeps[name] = vals
    for v in vals:
        d = dict(DEFAULTS); d[name] = v
        lines.append(" ".join(f"{k}={val:.6g}" for k, val in d.items()))
        index.append((name, v))

df = run(lines)
df["_sweep"] = [n for n, _ in index]
df["_val"] = [v for _, v in index]

# ── Continuity metric: max relative step between adjacent samples ──
KEY = ["pop", "biz_frac", "maxH", "fluid_frac", "nz", "park_frac",
       "inhab", "nblk", "indoor_inh_frac"]
print("=== Continuity report: largest adjacent jump per metric, per swept param ===")
print(f"(N={N} samples across each param's full optimizer range)\n")
rows = []
for name, lo, hi, scale in PS.PARAM_SPACE:
    sub = df[df["_sweep"] == name].sort_values("_val")
    rec = {"param": name}
    for k in KEY:
        y = sub[k].to_numpy(float)
        rng = np.nanmax(y) - np.nanmin(y)
        if rng <= 0:
            rec[k] = 0.0
            continue
        # largest single-step change as a fraction of the metric's full range
        rec[k] = float(np.nanmax(np.abs(np.diff(y))) / rng)
    rows.append(rec)
ctab = pd.DataFrame(rows).set_index("param")
pd.set_option("display.width", 200, "display.max_columns", 20)
print((ctab * 100).round(0).astype(int).to_string())
print("\n(values = largest single-step jump as % of that metric's full swept range;")
print(" 100% means the entire variation happens in ONE discontinuous step.)\n")

# ── Dead-zone metric: fraction of the swept range over which the population
#    objective driver (inhab) barely moves ──
print("=== Dead zones: fraction of each param's range where |d(inhab)| < 1% of range ===")
for name, lo, hi, scale in PS.PARAM_SPACE:
    sub = df[df["_sweep"] == name].sort_values("_val")
    y = sub["inhab"].to_numpy(float)
    rng = np.nanmax(y) - np.nanmin(y)
    if rng <= 0:
        print(f"  {name:16s}: inhab CONSTANT across range (objective-blind)")
        continue
    flat = np.mean(np.abs(np.diff(y)) < 0.01 * rng)
    print(f"  {name:16s}: {flat*100:4.0f}% of steps are flat   "
          f"(inhab range {np.nanmin(y):.0f}..{np.nanmax(y):.0f})")

# ── Plot: 12 params x 3 metric families ──
fig, axes = plt.subplots(len(PS.PARAM_SPACE), 3, figsize=(15, 2.5*len(PS.PARAM_SPACE)))
for i, (name, lo, hi, scale) in enumerate(PS.PARAM_SPACE):
    sub = df[df["_sweep"] == name].sort_values("_val")
    x = sub["_val"].to_numpy(float)
    logx = (scale == "log")

    ax = axes[i, 0]
    ax.plot(x, sub["pop"], "-", color="#2a5a9a", label="population")
    ax.plot(x, sub["target_pop"], "--", color="#888", label="density target")
    if logx: ax.set_xscale("log")
    ax.set_ylabel(name, fontsize=10, fontweight="bold")
    if i == 0: ax.set_title("population vs target", fontsize=10)
    ax.legend(fontsize=7, loc="best")

    ax = axes[i, 1]
    ax.plot(x, sub["biz_frac"], "-", color="#2a5a9a", label="biz frac")
    ax.plot(x, sub["park_frac"], "-", color="#4a8a3a", label="park frac")
    ax.plot(x, sub["indoor_inh_frac"], "-", color="#d45050", label="indoor inh frac")
    if logx: ax.set_xscale("log")
    if i == 0: ax.set_title("zoning fractions", fontsize=10)
    ax.legend(fontsize=7, loc="best")

    ax = axes[i, 2]
    ax.plot(x, sub["maxH"], "-", color="#c49030", label="max H (m)")
    ax.set_ylabel("max H (m)", fontsize=8)
    axt = ax.twinx()
    axt.plot(x, sub["fluid_frac"], "-", color="#3a9aa0", label="fluid frac")
    axt.plot(x, sub["nz"]/sub["nz"].max(), ":", color="#999", label="nz (norm)")
    axt.set_ylabel("fluid frac / nz", fontsize=8)
    if logx: ax.set_xscale("log")
    if i == 0: ax.set_title("geometry: height / blockage", fontsize=10)
    l1,lb1 = ax.get_legend_handles_labels(); l2,lb2 = axt.get_legend_handles_labels()
    ax.legend(l1+l2, lb1+lb2, fontsize=7, loc="best")

for ax in axes[-1]:
    ax.set_xlabel("parameter value", fontsize=9)
plt.suptitle("City-builder response across full parameter ranges "
             "(others fixed at canonical defaults)", fontsize=14, fontweight="bold", y=1.0)
plt.tight_layout(rect=[0,0,1,0.99])
plt.savefig("/mnt/user-data/outputs/builder_continuity_sweeps.png", dpi=130, bbox_inches="tight")
plt.close()
print("\nsaved builder_continuity_sweeps.png")
