#!/usr/bin/env python3
"""plot_stability.py — collision stability + WALE-engagement probe.

Reads stability_sweep.csv
  (H,collision,Cw,nu_floor,tau,stable,maxu,Xr,nut_peak,nut_mean,n_avg)
Panels:
  (A) stability map: green=stable/red=diverged per (operator,H) row vs tau.
  (B) Xr/H vs tau — reattachment vs Re, benchmark band shaded.
  (C) SGS engagement: peak nu_t/(U*H) vs tau (log) — does WALE engage as Re/H rise?
Colour = operator, marker = resolution H. Robust to a missing file.
"""
import os, csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = "stability_sweep.csv"
if not os.path.exists(CSV):
    print(f"{CSV} not found - run run_stability_sweep.sh first"); raise SystemExit

def fnum(x):
    try: return float(x)
    except (ValueError, TypeError): return float("nan")

rows = []
with open(CSV) as f:
    for r in csv.DictReader(f):
        rows.append(dict(H=int(fnum(r.get("H", 16))), col=r.get("collision", "mrt"),
                         tau=fnum(r.get("tau")), nf=fnum(r.get("nu_floor")),
                         stable=int(fnum(r.get("stable", 0)) or 0),
                         Xr=fnum(r.get("Xr")), npk=fnum(r.get("nut_peak")),
                         nmn=fnum(r.get("nut_mean"))))
rows = [r for r in rows if r["tau"] == r["tau"]]
if not rows:
    print("no parseable rows"); raise SystemExit

Hs   = sorted({r["H"] for r in rows})
cols = sorted({r["col"] for r in rows})
ccol = {"mrt": "#1f77b4", "reg": "#ff7f0e"}
mk   = ["o", "s", "^", "D", "v"]
hmk  = {H: mk[i % len(mk)] for i, H in enumerate(Hs)}

fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))

labels = [(c, H) for c in cols for H in Hs]
ypos = {lab: i for i, lab in enumerate(labels)}
for r in rows:
    y = ypos[(r["col"], r["H"])]
    ax[0].scatter(r["tau"], y, c=("#2ca02c" if r["stable"] else "#d62728"),
                  s=150, marker="s", edgecolors="k", zorder=3)
for (c, H), y in ypos.items():
    st = [r["tau"] for r in rows if r["col"] == c and r["H"] == H and r["stable"]]
    if st:
        ax[0].annotate(f"tau*={min(st):.3f}", (min(st), y), textcoords="offset points",
                       xytext=(-4, 0), ha="right", va="center", fontsize=7)
ax[0].axvline(0.5, ls="--", color="k", lw=1.2)
ax[0].set_yticks(list(ypos.values()))
ax[0].set_yticklabels([f"{c}.H{H}" for (c, H) in labels])
ax[0].set_xlabel("tau (=0.5+3nu_floor; lower => higher Re)")
ax[0].set_title("Stability map (green=stable, red=diverged)")
ax[0].invert_xaxis()

for c in cols:
    for H in Hs:
        pts = sorted([(r["tau"], r["Xr"]) for r in rows
                      if r["col"] == c and r["H"] == H and r["stable"] and r["Xr"] == r["Xr"]])
        if pts:
            ax[1].plot([p[0] for p in pts], [p[1] for p in pts],
                       marker=hmk[H], color=ccol.get(c, None), label=f"{c}.H{H}")
ax[1].axhspan(1.0, 2.3, color="green", alpha=0.08)
ax[1].axhline(1.6, ls=":", color="g", lw=1)
ax[1].set_xlabel("tau"); ax[1].set_ylabel("Xr/H"); ax[1].invert_xaxis()
ax[1].set_title("Reattachment vs Re (band=benchmark)")
ax[1].legend(fontsize=7, ncol=max(1, len(Hs)))

any_nut = False
for c in cols:
    for H in Hs:
        pts = sorted([(r["tau"], r["npk"]) for r in rows
                      if r["col"] == c and r["H"] == H and r["stable"]
                      and r["npk"] == r["npk"] and r["npk"] > 0])
        if pts:
            any_nut = True
            ax[2].plot([p[0] for p in pts], [p[1] for p in pts],
                       marker=hmk[H], color=ccol.get(c, None), label=f"{c}.H{H}")
if any_nut:
    ax[2].set_yscale("log")
ax[2].set_xlabel("tau"); ax[2].set_ylabel("peak nu_t/(U*H)"); ax[2].invert_xaxis()
ax[2].set_title("WALE engagement vs Re/resolution")
ax[2].legend(fontsize=7, ncol=max(1, len(Hs)))

fig.suptitle("Collision stability + WALE engagement: MRT vs regularized-MRT", fontweight="bold")
fig.tight_layout(); fig.savefig("fig_stability_sweep.png", dpi=130)
print("wrote fig_stability_sweep.png")
