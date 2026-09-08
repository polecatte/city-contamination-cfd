#!/usr/bin/env python3
"""plot_diffusion.py — visualize the numerical-diffusion comparison.

bench (default): cross-stream blob profiles (initial vs linear-upwind vs van-Leer)
  after pure advection — the exact answer is the initial profile, so the spread of
  each scheme is its numerical diffusion. Annotates the effective numerical
  diffusivities from diffusion_bench.csv.

city <NX> <NY>: side-by-side z=1 concentration fields from the production van Leer
  D3Q7 scalar vs the linear-upwind objective transport (same flow, same source),
  plus a downwind centreline comparison.

Usage:
  python3 plot_diffusion.py                 # bench
  python3 plot_diffusion.py city <NX> <NY>  # city fields
"""
import sys, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_profile(fn):
    eta, C = [], []
    with open(fn) as f:
        r = csv.reader(f); next(r)
        for row in r:
            eta.append(float(row[0])); C.append(float(row[1]))
    e = np.array(eta); c = np.array(C)
    o = np.argsort(e)
    return e[o], c[o]


def bench():
    vals = {}
    with open("diffusion_bench.csv") as f:
        r = csv.reader(f); hdr = next(r); row = next(r)
        for k, v in zip(hdr, row):
            try: vals[k] = float(v)
            except ValueError: vals[k] = v

    ei, ci = load_profile("prof_initial.csv")
    eu, cu = load_profile("prof_upwind.csv")
    ev, cv = load_profile("prof_vanleer.csv")
    # normalize each profile to unit peak for shape comparison
    norm = lambda a: a / (a.max() if a.max() > 0 else 1)

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))

    ax[0].plot(ei, norm(ci), "k--", lw=2, label=f"exact (no spread), σ₀={vals['sigma0']:.2f}")
    ax[0].plot(ev, norm(cv), color="#1b7837", lw=2,
               label=f"van-Leer TVD, σ={vals['sigma_vanleer']:.2f}")
    ax[0].plot(eu, norm(cu), color="#b2182b", lw=2,
               label=f"linear-upwind (objective), σ={vals['sigma_upwind']:.2f}")
    ax[0].set_xlabel("cross-stream coordinate η (cells)")
    ax[0].set_ylabel("normalized concentration")
    ax[0].set_title("Plume cross-section after pure advection\n(exact = no broadening)")
    ax[0].legend(); ax[0].grid(alpha=.3)
    ax[0].set_xlim(vals["sigma0"]*-8 + np.median(eu), np.median(eu) + vals["sigma0"]*8)

    labels = ["van-Leer\nTVD", "linear-upwind\n(objective)"]
    dnum = [vals["Dnum_vanleer"], vals["Dnum_upwind"]]
    bars = ax[1].bar(labels, dnum, color=["#1b7837", "#b2182b"])
    ax[1].axhspan(1e-2, 1e-1, color="gray", alpha=.2,
                  label="physical eddy diffusivity\nνt/Sc_t ~ 0.01–0.1")
    ax[1].set_yscale("log")
    ax[1].set_ylabel("effective numerical diffusivity (lattice units)")
    ax[1].set_title("Numerical-diffusion penalty\n(lower is better; band = physical mixing)")
    for b, d in zip(bars, dnum):
        ax[1].text(b.get_x()+b.get_width()/2, d*1.15, f"{d:.1e}", ha="center", fontsize=10)
    ax[1].legend(loc="lower left"); ax[1].grid(alpha=.3, which="both", axis="y")

    ratio = vals["Dnum_upwind"]/vals["Dnum_vanleer"] if vals["Dnum_vanleer"] else float("nan")
    fig.suptitle(f"Linear-upwind vs van-Leer numerical diffusion — penalty ≈ {ratio:.0f}×",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig("diffusion_bench.png", dpi=130, bbox_inches="tight")
    print("wrote diffusion_bench.png")
    print(f"  D_num linear-upwind = {vals['Dnum_upwind']:.3e} (LU)")
    print(f"  D_num van-Leer      = {vals['Dnum_vanleer']:.3e} (LU)")
    print(f"  penalty ratio       = {ratio:.0f}x")
    print("  NB: upwind D_num overlaps the physical eddy-diffusivity band → it")
    print("      over-diffuses the plume by ~the amount of real turbulent mixing.")


def read_field(fn):
    with open(fn, "rb") as f:
        nx, ny = np.fromfile(f, np.int32, 2)
        d = np.fromfile(f, np.float32, nx*ny).reshape(ny, nx)
    return d


def city(nx, ny):
    vl = read_field("tiac_vanleer_z1.bin")
    up = read_field("tiac_upwind_z1.bin")
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    for a, d, t in ((ax[0], vl, "production van Leer (physical std)"),
                    (ax[1], up, "linear-upwind (objective)")):
        n = d / (d.max() if d.max() > 0 else 1)
        im = a.imshow(n, origin="lower", cmap="magma", vmax=1)
        a.set_title(t); fig.colorbar(im, ax=a, fraction=.046)
    yc = vl.shape[0]//2
    ax[2].plot(vl[yc]/vl[yc].max(), color="#1b7837", label="van Leer")
    ax[2].plot(up[yc]/up[yc].max(), color="#b2182b", label="linear-upwind")
    ax[2].set_title("centreline (normalized)"); ax[2].set_xlabel("x (cells)")
    ax[2].legend(); ax[2].grid(alpha=.3)
    fig.suptitle("Same city, same flow, same source: production van Leer vs linear-upwind", y=1.02)
    fig.tight_layout(); fig.savefig("diffusion_city.png", dpi=130, bbox_inches="tight")
    print("wrote diffusion_city.png")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "city":
        city(int(sys.argv[2]), int(sys.argv[3]))
    else:
        bench()
