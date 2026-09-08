#!/usr/bin/env python3
"""airflow_validation_plots.py — figures for the rigorous airflow validation suite.

Reads the av_*.csv / av_*.bin artefacts written by airflow_validation.cpp and
produces one PNG per test. Every panel is guarded: a missing file is skipped with
a note, so this runs after a partial suite without crashing.

Outputs:  fig_inflow_turbulence.png, fig_cube_benchmark.png,
          fig_canyon.png, fig_grid_convergence.png
Usage:    python3 airflow_validation_plots.py
"""
import os, struct, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def _csv(path):
    """Load a headered CSV as (header_list, ndarray). None if absent/empty."""
    if not os.path.exists(path):
        print(f"  skip: {path} not found"); return None, None
    try:
        hdr = open(path).readline().strip().split(",")
        data = np.genfromtxt(path, delimiter=",", skip_header=1)
        if data.size == 0: return hdr, None
        if data.ndim == 1: data = data.reshape(1, -1)
        return hdr, data
    except Exception as e:
        print(f"  skip: {path} ({e})"); return None, None

def _slice_xz(path):
    """dump_xz: int32[nx,nz] then float32 ux,uz,speed,nut (each nx*nz, k=z*nx+x)."""
    if not os.path.exists(path):
        print(f"  skip: {path} not found"); return None
    with open(path, "rb") as f:
        nx, nz = struct.unpack("ii", f.read(8))
        a = np.frombuffer(f.read(), dtype=np.float32)
    n = nx * nz
    if a.size < 4 * n: return None
    fields = {k: a[i*n:(i+1)*n].reshape(nz, nx) for i, k in enumerate(("ux","uz","speed","nut"))}
    fields["nx"], fields["nz"] = nx, nz
    return fields

def fig_inflow():
    hdrp, P = _csv("av_inflow_profiles.csv")
    hdrs, S = _csv("av_inflow_spectrum.csv")
    if P is None and S is None: return
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    if P is not None:
        z, Iu, Iv, Iw = P[:,0], P[:,2], P[:,3], P[:,4]
        su, sv, sw = P[:,5], P[:,6], P[:,7]
        ax[0].plot(su, z, "o-", label=r"$\sigma_u/u_*$")
        ax[0].plot(sv, z, "s-", label=r"$\sigma_v/u_*$")
        ax[0].plot(sw, z, "^-", label=r"$\sigma_w/u_*$")
        for val, c in ((2.5,"C0"),(1.9,"C1"),(1.25,"C2")):
            ax[0].axvline(val, ls="--", lw=1, color=c, alpha=.7)
        ax[0].set_xlabel(r"$\sigma_i/u_*$ (dashed = surface-layer target)")
        ax[0].set_ylabel("z (m)"); ax[0].set_title("Reconstructed variances"); ax[0].legend()
        ax[1].plot(100*Iu, z, "o-", label=r"$I_u$")
        ax[1].plot(100*Iv, z, "s-", label=r"$I_v$")
        ax[1].plot(100*Iw, z, "^-", label=r"$I_w$")
        ax[1].set_xlabel("Turbulence intensity (%)"); ax[1].set_ylabel("z (m)")
        ax[1].set_title(r"TI profiles ($I_u>I_v>I_w$)"); ax[1].legend()
    if S is not None:
        f, Pw = S[:,0], S[:,1]
        m = (f > 0) & (Pw > 0)
        ax[2].loglog(f[m], Pw[m], "-", lw=1.2, label="RFG inflow spectrum")
        band = m & (f > 3*f[m].min()) & (f < 0.5*f[m].max())
        if band.any():
            fr = f[band]; ref = Pw[band][0] * (fr/fr[0])**(-5/3)
            ax[2].loglog(fr, ref, "k--", lw=1, label=r"$-5/3$ reference")
        ax[2].set_xlabel("frequency"); ax[2].set_ylabel("power")
        ax[2].set_title("Temporal spectrum (RFG ≈ Gaussian)"); ax[2].legend()
    fig.suptitle("Inflow turbulence statistics (T_inflow_turb)", fontweight="bold")
    fig.tight_layout(); fig.savefig("fig_inflow_turbulence.png", dpi=130); plt.close(fig)
    print("  wrote fig_inflow_turbulence.png")

def fig_cube():
    hw, W = _csv("av_cube_bench_windward.csv")
    hr, R = _csv("av_cube_bench_roof.csv")
    sl = _slice_xz("av_cube_bench_xz.bin")
    if W is None and R is None and sl is None: return
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    if W is not None:
        ax[0].plot(W[:,1], W[:,0], "o-"); ax[0].axvline(0, ls=":", color="k", lw=1)
        ax[0].set_xlabel(r"$C_p$"); ax[0].set_ylabel("z/H")
        ax[0].set_title("Windward face (stagnation)")
    if R is not None:
        ax[1].plot(R[:,0], R[:,1], "s-"); ax[1].axhline(0, ls=":", color="k", lw=1)
        ax[1].set_xlabel("x/H along roof"); ax[1].set_ylabel(r"$C_p$")
        ax[1].set_title("Roof centreline (separation suction)")
    if sl is not None:
        sp = sl["speed"]; im = ax[2].imshow(sp, origin="lower", aspect="auto", cmap="viridis")
        try:
            nx, nz = sl["nx"], sl["nz"]
            xx, zz = np.meshgrid(np.arange(nx), np.arange(nz))
            ax[2].streamplot(xx, zz, sl["ux"], sl["uz"], density=1.1, color="w", linewidth=.5, arrowsize=.6)
        except Exception as e:
            print(f"    (streamlines skipped: {e})")
        fig.colorbar(im, ax=ax[2], label="|u| (LU)")
        ax[2].set_title("Centreline x–z slice"); ax[2].set_xlabel("x (cells)"); ax[2].set_ylabel("z (cells)")
    fig.suptitle("Wall-mounted cube benchmark (T_cube_bench)", fontweight="bold")
    fig.tight_layout(); fig.savefig("fig_cube_benchmark.png", dpi=130); plt.close(fig)
    print("  wrote fig_cube_benchmark.png")

def fig_canyon():
    hp, Pr = _csv("av_canyon_profile.csv")
    sl = _slice_xz("av_canyon_xz.bin")
    if Pr is None and sl is None: return
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    if Pr is not None:
        ax[0].plot(Pr[:,1], Pr[:,0], "o-"); ax[0].axvline(0, ls="--", color="r", lw=1)
        ax[0].axhline(1.0, ls=":", color="k", lw=1, label="roof level")
        ax[0].set_xlabel(r"$u_x/U_{roof}$ (<0 ⇒ reversed)"); ax[0].set_ylabel("z/H")
        ax[0].set_title("Street-centre profile"); ax[0].legend()
    if sl is not None:
        sp = sl["speed"]; im = ax[1].imshow(sp, origin="lower", aspect="auto", cmap="viridis")
        try:
            nx, nz = sl["nx"], sl["nz"]
            xx, zz = np.meshgrid(np.arange(nx), np.arange(nz))
            ax[1].streamplot(xx, zz, sl["ux"], sl["uz"], density=1.3, color="w", linewidth=.5, arrowsize=.6)
        except Exception as e:
            print(f"    (streamlines skipped: {e})")
        fig.colorbar(im, ax=ax[1], label="|u| (LU)")
        ax[1].set_title("Canyon x–z slice (primary vortex)"); ax[1].set_xlabel("x (cells)"); ax[1].set_ylabel("z (cells)")
    fig.suptitle("Street canyon, W/H=1 skimming flow (T_canyon)", fontweight="bold")
    fig.tight_layout(); fig.savefig("fig_canyon.png", dpi=130); plt.close(fig)
    print("  wrote fig_canyon.png")

def fig_gridconv():
    hg, G = _csv("av_gridconv.csv")
    hk, K = _csv("av_gridconv_gci.csv")
    if G is None: return
    H, Xr = G[:,0], G[:,2]
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    inv = 1.0 / H
    ax.plot(inv, Xr, "o-", ms=8, label="Xr/H (computed)")
    if K is not None:
        asym = K[0, 5] if K.shape[1] > 5 else None
        p = K[0, 3] if K.shape[1] > 3 else None
        gci = K[0, 4] if K.shape[1] > 4 else None
        if asym is not None and np.isfinite(asym):
            ax.axhline(asym, ls="--", color="g",
                       label=f"Richardson asymptote = {asym:.2f}")
            txt = []
            if p   is not None: txt.append(f"observed order p = {p:.2f}")
            if gci is not None: txt.append(f"GCI(fine) = {gci:.1f}%")
            if txt: ax.text(0.05, 0.05, "\n".join(txt), transform=ax.transAxes,
                            va="bottom", ha="left", fontsize=9,
                            bbox=dict(boxstyle="round", fc="w", alpha=.8))
    ax.set_xlabel("1/H  (grid spacing ∝ 1/H)"); ax.set_ylabel("Xr/H")
    ax.set_title("Grid convergence of reattachment length (T_gridconv)", fontweight="bold")
    ax.legend(); fig.tight_layout(); fig.savefig("fig_grid_convergence.png", dpi=130); plt.close(fig)
    print("  wrote fig_grid_convergence.png")

def main():
    print("airflow_validation_plots.py — generating figures from av_*.csv / av_*.bin")
    for fn in (fig_inflow, fig_cube, fig_canyon, fig_gridconv):
        try: fn()
        except Exception as e: print(f"  {fn.__name__} failed: {e}")
    print("done.")

if __name__ == "__main__":
    main()
