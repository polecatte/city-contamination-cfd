"""rd_coarse.py — reaction-diffusion zoning patterns at CITY scale (coarse).

The earlier RD demo ran at 120² so features were ~6 cells → ~20 across the domain:
far finer than a real block grid (a 600 m city at 40 m blocks is ~15 blocks wide).
The Gray-Scott wavelength is fixed at a few cells, so the way to coarsen the pattern
relative to the domain is to run the simulation on a COARSER grid. Here R (the sim
resolution) is the length-scale knob: small R → few, large features. Patterns are
shown blocky (nearest) so they read as a zoning map.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

rng = np.random.default_rng(11)

def gray_scott(F, k, R=44, steps=9000, Du=0.16, Dv=0.08, aniso=1.0):
    """Gray-Scott on an R×R periodic grid. aniso>1 stretches x-diffusion → oriented."""
    u = np.ones((R, R)); v = np.zeros((R, R))
    nseed = max(12, R*R//90)
    for _ in range(nseed):
        i, j = rng.integers(1, R-2, size=2)
        v[i-1:i+2, j-1:j+2] = 0.5; u[i-1:i+2, j-1:j+2] = 0.25
    u += 0.01*rng.random((R, R)); v += 0.01*rng.random((R, R))
    # anisotropic diffusion, TRACE-PRESERVING (ax+ay=2) so stability matches the
    # isotropic case regardless of the anisotropy ratio.
    ax = 2.0*aniso/(1.0+aniso); ay = 2.0/(1.0+aniso)
    def lap(a):
        return (-(2*ax+2*ay)*a
                + ax*(np.roll(a,1,0)+np.roll(a,-1,0))
                + ay*(np.roll(a,1,1)+np.roll(a,-1,1)))
    for _ in range(steps):
        uvv = u*v*v
        u += Du*lap(u) - uvv + F*(1-u)
        v += Dv*lap(v) + uvv - (F+k)*v
    return v

def thresh(field, frac=0.30):
    return (field >= np.quantile(field, 1-frac)).astype(float)   # business = high-v

def corr_len(b, axis):
    b = b - b.mean(); n = b.shape[axis]
    B = np.moveaxis(b, axis, -1); ac = np.zeros(n)
    for row in B.reshape(-1, n):
        f = np.fft.rfft(row, 2*n); ac += np.fft.irfft(f*np.conj(f), 2*n)[:n]
    ac /= ac[0] if ac[0] else 1.0
    w = np.where(ac < 1/np.e)[0]
    return w[0] if len(w) else n
def lam(b): return corr_len(b,0)/max(1e-6, corr_len(b,1))

# (title, F, k, R, aniso, note)
CASES = [
    ("Large spots",      0.035, 0.065, 40, 1.0, "regular wake array — sparse tall clusters"),
    ("Coral / mitosis",  0.0367,0.0649,40, 1.0, "budding lobes — irregular mid-size masses"),
    ("Thick worms",      0.054, 0.063, 38, 1.0, "few fat finger-canyons"),
    ("Coarse labyrinth", 0.029, 0.057, 46, 1.0, "few wide tortuous corridors"),
    ("Open holes",       0.039, 0.058, 40, 1.0, "built fabric perforated by large courts"),
    ("Oriented worms",   0.054, 0.063, 40, 3.0, "anisotropic diffusion → wind-aligned canyons"),
]

fig, axes = plt.subplots(2, 3, figsize=(15, 12.0))
cmap = plt.matplotlib.colors.ListedColormap(["#e9e4d8", "#2e6f6a"])
for ax, (title, F, k, R, an, note) in zip(axes.ravel(), CASES):
    v = gray_scott(F, k, R=R, aniso=an)
    b = thresh(v)
    ax.imshow(b.T, origin="lower", cmap=cmap, interpolation="nearest")
    ax.set_title(f"{title}   (R={R}"+(f", aniso={an:g}" if an!=1 else "")+f")\nΛ = {lam(b):.2f}",
                 fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel(note, fontsize=9, style="italic", labelpad=8)
    ax.set_xticks([]); ax.set_yticks([])
    ax.arrow(1.5, 2, R*0.13, 0, head_width=R*0.03, head_length=R*0.03, fc="#b23", ec="#b23")
    print(f"  {title:16s} R={R} std={v.std():.3f} fill={b.mean():.2f} Lam={lam(b):.2f}")

fig.legend(handles=[Patch(facecolor="#2e6f6a", label="business / tall"),
                    Patch(facecolor="#e9e4d8", label="open / low")],
           loc="lower center", ncol=2, fontsize=12, frameon=True)
fig.suptitle("Reaction–diffusion zoning at city scale — coarsened  "
             "(R = sim resolution sets feature size · red arrow = wind · Λ = Lx/Ly)",
             fontsize=14, fontweight="bold")
plt.tight_layout(rect=[0, 0.04, 1, 0.95], h_pad=5.0)
out = "/mnt/user-data/outputs/rd_coarse.png"
plt.savefig(out, dpi=140, bbox_inches="tight"); print("saved", out)
