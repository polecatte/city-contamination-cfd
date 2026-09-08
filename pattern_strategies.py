"""pattern_strategies.py — candidate zoning-field GENERATORS beyond blobs/speckle.

The current Φ basis (radial, bipeak, corridor, maximin scatter) can only make a
disc, a few discs, a stripe, or a uniform speckle — all either low-frequency lumps
or structureless blue noise. The mid-frequency, anisotropic, contiguous-but-tortuous
regime (ridges, ramps, networks, labyrinths, streaks) is missing, and that regime is
exactly where interesting canopy CFD lives (channelling, developing internal boundary
layers, periodic recirculation, bicontinuous through-flow).

This script renders each candidate generator as a business/open pattern at a FIXED
~28% fill, and annotates a channelling index Λ = Lx/Ly (ratio of correlation lengths
along vs across wind, wind assumed +x). Λ≈1 isotropic, Λ≫1 wind-parallel canyons
(strong channelling), Λ≪1 wind-blocking ranks.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

N = 120
rng = np.random.default_rng(7)
xs = np.linspace(0, 1, N)
X, Y = np.meshgrid(xs, xs, indexing="ij")
cx = cy = 0.5

def thresh(field, frac=0.28):
    """Binary business mask: the `frac` lowest-field cells (business prefers low Φ)."""
    t = np.quantile(field, frac)
    return (field <= t).astype(float)

def corr_len(b, axis):
    """1/e correlation length of binary field along an axis (in cells)."""
    b = b - b.mean()
    n = b.shape[axis]
    # average 1-D autocorrelation along the axis
    B = np.moveaxis(b, axis, -1)
    ac = np.zeros(n)
    for row in B.reshape(-1, n):
        f = np.fft.rfft(row, 2*n)
        c = np.fft.irfft(f*np.conj(f), 2*n)[:n]
        ac += c
    ac /= ac[0] if ac[0] != 0 else 1.0
    below = np.where(ac < 1/np.e)[0]
    return below[0] if len(below) else n

def lam(b):
    return corr_len(b, 0) / max(1e-6, corr_len(b, 1))

# ── Gray–Scott reaction–diffusion ──────────────────────────────────────────
def gray_scott(F, k, steps=12000, Du=0.16, Dv=0.08):
    u = np.ones((N, N)); v = np.zeros((N, N))
    # nucleate domain-wide: many random 3×3 seed patches + low-amplitude field noise,
    # so the pattern fills the whole grid rather than growing from one centre.
    for _ in range(80):
        i, j = rng.integers(1, N-2, size=2)
        v[i-1:i+2, j-1:j+2] = 0.5; u[i-1:i+2, j-1:j+2] = 0.25
    u += 0.01*rng.random((N, N)); v += 0.01*rng.random((N, N))
    def lap(a):   # standard Gray–Scott 9-point stencil: 0.2 orthogonal, 0.05 diagonal
        return (-1.0*a
                + 0.2*(np.roll(a,1,0)+np.roll(a,-1,0)+np.roll(a,1,1)+np.roll(a,-1,1))
                + 0.05*(np.roll(np.roll(a,1,0),1,1)+np.roll(np.roll(a,1,0),-1,1)
                        +np.roll(np.roll(a,-1,0),1,1)+np.roll(np.roll(a,-1,0),-1,1)))
    for _ in range(steps):
        uvv = u*v*v
        u += (Du*lap(u) - uvv + F*(1-u))
        v += (Dv*lap(v) + uvv - (F+k)*v)
    return v

# ── Generators ──────────────────────────────────────────────────────────────
def radial():     return np.sqrt((X-cx)**2 + (Y-cy)**2)
def bipeak():     return -(np.cos(4*np.pi*(X-cx)) + np.cos(4*np.pi*(Y-cy)))
def corridor():   return np.abs(Y - cy)
def speckle():    return rng.random((N, N))
def ridge():      # anisotropic Gaussian: long in x, thin in y → wind-parallel wall
    return -np.exp(-(((X-cx)**2)/(2*0.32**2) + ((Y-cy)**2)/(2*0.05**2)))
def ramp():       return X                                   # monotone density gradient
def grid_net():   # network of streets: business on a lattice of corridors
    m = 4
    return -(np.exp(-(np.sin(np.pi*m*X)**2)/0.02) + np.exp(-(np.sin(np.pi*m*Y)**2)/0.02))
def shear():      # diagonal band field (oblique to wind)
    return np.cos(2*np.pi*3*((X-cx) + 0.6*(Y-cy)))
def trig_maze():  # trig level-set → periodic bicontinuous labyrinth
    m = 5
    return np.abs(np.cos(np.pi*m*X) + np.cos(np.pi*m*Y))
def spectral_streaks():   # band-limited anisotropic noise → wind-parallel streaks
    return gaussian_filter(rng.random((N, N)), sigma=(10, 2), mode="wrap")

PANELS = [
    # (title, field, cfd_note)   — Row A: current basis
    ("Radial disc\n(current)",        radial(),   "one bluff body → single wake"),
    ("Bipeak discs\n(current)",       bipeak(),   "few wakes, otherwise open"),
    ("Single corridor\n(current)",    corridor(), "one canyon: channels along wind, blocks across"),
    ("Maximin speckle\n(current)",    speckle(),  "homogeneous roughness, no large structure"),
    # Row B: directional / gradient
    ("Anisotropic ridge",             ridge(),    "wall: channels along wind, blocks across"),
    ("Density ramp",                  ramp(),     "developing internal boundary layer"),
    ("Corridor network",              grid_net(), "street grid → intersecting channels"),
    ("Oblique shear bands",           shear(),    "bands skew to wind → cross-stream mixing"),
    # Row C: pattern formation (organic, tunable)
    ("Reaction–diffusion: spots",     gray_scott(0.035, 0.065), "regular wake array (cube-array-like)"),
    ("Reaction–diffusion: labyrinth", gray_scott(0.029, 0.057), "tortuous canyons → high dispersion"),
    ("Reaction–diffusion: worms",     gray_scott(0.054, 0.063), "aligned finger canyons"),
    ("Band-limited streaks",          spectral_streaks(),       "tunable canyon length & anisotropy"),
]

fig, axes = plt.subplots(3, 4, figsize=(18, 14))
biz_cmap = plt.matplotlib.colors.ListedColormap(["#e9e4d8", "#2e6f6a"])  # open, business
for ax, (title, field, note) in zip(axes.ravel(), PANELS):
    b = thresh(field)
    ax.imshow(b.T, origin="lower", cmap=biz_cmap, interpolation="nearest")
    ax.set_title(f"{title}\nΛ = {lam(b):.2f}", fontsize=11, fontweight="bold")
    ax.set_xlabel(note, fontsize=9, style="italic")
    ax.set_xticks([]); ax.set_yticks([])
    ax.arrow(4, 6, 16, 0, head_width=4, head_length=4, fc="#b23", ec="#b23")  # wind +x

from matplotlib.patches import Patch
fig.legend(handles=[Patch(facecolor="#2e6f6a", label="business / tall"),
                    Patch(facecolor="#e9e4d8", label="open / low")],
           loc="lower center", ncol=2, fontsize=12, frameon=True)
fig.suptitle("Zoning-field generators beyond blobs & speckle  "
             "(fill ≈ 28% · red arrow = wind · Λ = channelling index Lx/Ly)",
             fontsize=15, fontweight="bold")
plt.tight_layout(rect=[0, 0.03, 1, 0.96])
out = "/mnt/user-data/outputs/pattern_strategies.png"
plt.savefig(out, dpi=135, bbox_inches="tight")
print("saved", out)
for title, field, note in PANELS:
    print(f"  {title.splitlines()[0]:32s} Λ={lam(thresh(field)):.2f}")
