#!/usr/bin/env python3
"""plot_zoning.py GEOM_DIR DOSE.npz OUT.png — the inner programme's result for one built form.

Panels: expected resident dose per building; the mean-optimal zoning; the zoning with the event
tail (CVaR90) cut by ~7 %; mean vs CVaR90 for 200 random zonings and the optimal front.
"""
import os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import zoning_milp as Z


def main():
    geom, dose, png = sys.argv[1:4]
    sys.argv = [sys.argv[0], geom, dose, "--out", os.path.join(os.path.dirname(os.path.abspath(png)), "zoning_out")]
    rows, capR, capE, a, c, prob, P, J, free, ref, front = Z.main()
    m0, cv0 = Z.evaluate(free["R"], free["E"], a, c, prob)
    tight = min(front, key=lambda f: abs(f[1] / cv0 - 0.93))
    abar = prob @ a
    fig, axs = plt.subplots(1, 4, figsize=(22, 5.4))

    def draw(ax, values, cmap, title, vmin=None, vmax=None, edges=None):
        for b, r in enumerate(rows):
            x0, x1, y0, y1 = (float(r[k]) for k in ("x0", "x1", "y0", "y1"))
            col = cmap(values[b]) if callable(cmap) else cmap[b]
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc=col, ec=(edges[b] if edges else "none"), lw=1.2))
        xs = [float(r["x0"]) for r in rows] + [float(r["x1"]) for r in rows]
        ys = [float(r["y0"]) for r in rows] + [float(r["y1"]) for r in rows]
        ax.set_xlim(min(xs) - 2, max(xs) + 2); ax.set_ylim(min(ys) - 2, max(ys) + 2)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=10)

    norm = plt.Normalize(abar.min(), abar.max())
    draw(axs[0], norm(abar), plt.cm.magma_r, "expected dose per resident, by building\n(proxy dose, uniform wind rose, all release zones)")
    sm = plt.cm.ScalarMappable(norm=norm, cmap="magma_r"); fig.colorbar(sm, ax=axs[0], shrink=0.8)
    for ax, sol, t in ((axs[1], free, f"mean-optimal zoning: mean {m0:.3g}, CVaR90 {cv0:.3g}"),
                       (axs[2], tight[2], f"with CVaR90 cut to {tight[1]/cv0:.2f}x: mean {tight[0]/m0:.3f}x")):
        occ = np.where(sol["u"] == 1, sol["R"] / capR, sol["E"] / np.maximum(capE, 1e-9))
        cols = [plt.cm.Blues(0.35 + 0.6 * o) if u else plt.cm.Oranges(0.35 + 0.6 * o) for u, o in zip(sol["u"], occ)]
        draw(axs[1] if ax is axs[1] else axs[2], None, cols, t + "\nblue = homes, orange = workplaces (darker = fuller)")
    ax = axs[3]
    ax.scatter(ref[:, 0], ref[:, 1], s=8, c="#adb5bd", label="200 random zonings")
    fr = np.array([(m, cv) for m, cv, _ in front])
    ax.plot(fr[:, 0], fr[:, 1], "o-", c="#4263eb", label="optimal: mean, then tighter CVaR90 bounds")
    ax.set_xlabel("expected population dose (mean)"); ax.set_ylabel("CVaR90 over release zone x wind")
    ax.set_title("mean vs event tail", fontsize=10); ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.suptitle("Inner level (zoning) on a fixed built form. PROXY DOSE: 2-D stand-in, not CFD", fontsize=11, color="#c92a2a")
    fig.tight_layout(); fig.savefig(png, dpi=100)


if __name__ == "__main__":
    main()
