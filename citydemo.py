"""citydemo.py — prototype of the aggregate-morphology grid-city generator.

Six scenario-robust parameters (see design note), each tied to a distinct
mechanism rather than to specific spatial placement:
  lambda_p   plan-area (footprint) density          [0.20, 0.55]
  HW         canyon aspect ratio  H/W (mean H / street width)  [0.4, 2.0]
  sigma_h    height heterogeneity sigma_h/hbar       [0.0, 0.5]
  theta      grid orientation vs prevailing wind     [0, 45] deg
  phi_open   open-space fraction                     [0.05, 0.30]
  conn       open-space connectivity (scattered->wind-aligned corridors) [0,1]
  rho_c      population centralization               [0,1]   (shading only here)

Construction is fully deterministic in the aggregate knobs: street width and
block pitch are DERIVED from (hbar, HW, lambda_p) so the canyon regime is set
directly, not as a byproduct. hbar is fixed here (the population-equality
constraint would otherwise pin total floor area); the demo shows morphology,
not the density nudging.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

HBAR = 24.0          # reference mean building height (m); pinned by floor-area constraint
L    = 512.0         # city extent (m)
DX   = 2.0           # raster resolution (m)
N    = int(L/DX)

def flow_regime(HW):
    if HW < 0.3:  return "isolated roughness"
    if HW < 0.65: return "wake interference"
    return "skimming flow"

def build_city(lambda_p, HW, sigma_h, theta_deg, phi_open, conn, rho_c, seed=0):
    rng = np.random.default_rng(seed)
    # Derive street width W and block size B from the aggregate knobs.
    W = HBAR / HW                                   # H/W = hbar/W
    sq = np.sqrt(np.clip(lambda_p, 0.04, 0.81))
    B = W * sq / (1.0 - sq)                          # lambda_p = B^2/(B+W)^2
    P = B + W                                        # block pitch
    nb = int(np.ceil(L / P)) + 2                     # blocks per side (+margin)

    # Per-block heights (lognormal, mean hbar, CoV sigma_h) and park flags.
    mu = np.log(HBAR) - 0.5*np.log(1.0 + sigma_h**2)
    sg = np.sqrt(np.log(1.0 + sigma_h**2)) if sigma_h > 1e-6 else 0.0
    bh = rng.lognormal(mu, sg, size=(nb, nb)) if sg > 0 else np.full((nb, nb), HBAR)

    # Open space: connectivity blends scattered parks (conn=0) with full
    # wind-aligned corridors (conn=1, columns => parallel to +x wind).
    park = np.zeros((nb, nb), bool)
    n_target = int(round(phi_open * nb * nb))
    n_corr = int(round(conn * n_target))
    placed = 0
    if n_corr > 0:
        cols = rng.choice(nb, size=max(1, n_corr // nb + 1), replace=False)
        for c in cols:
            park[:, c] = True
            placed += nb
            if placed >= n_corr: break
    while placed < n_target:                          # scatter the remainder
        i, j = rng.integers(nb), rng.integers(nb)
        if not park[i, j]: park[i, j] = True; placed += 1
    bh[park] = 0.0

    # Rasterize: rotate each pixel into the (rotated) block frame, decide
    # building-footprint vs street, look up that block's height.
    th = np.radians(theta_deg); ct, st = np.cos(th), np.sin(th)
    xs = (np.arange(N) + 0.5) * DX - L/2
    X, Y = np.meshgrid(xs, xs)
    Xr =  ct*X + st*Y + L/2
    Yr = -st*X + ct*Y + L/2
    bi = np.floor(Xr / P).astype(int) % nb
    bj = np.floor(Yr / P).astype(int) % nb
    fx = Xr - np.floor(Xr / P)*P                       # position within cell
    fy = Yr - np.floor(Yr / P)*P
    inside = (fx >= W/2) & (fx < W/2 + B) & (fy >= W/2) & (fy < W/2 + B)
    H = np.where(inside, bh[bi, bj], 0.0)
    is_park = inside & park[bi, bj]
    return dict(H=H, is_park=is_park, W=W, B=B, P=P,
                regime=flow_regime(HW), lambda_p=lambda_p, HW=HW)

def draw(ax, c, title):
    H = c["H"].copy()
    disp = np.ma.masked_where(c["is_park"], H)
    cmap = plt.cm.inferno.copy()
    ax.imshow(disp, origin="lower", cmap=cmap, vmin=0, vmax=HBAR*2.2,
              extent=[0, L, 0, L], interpolation="nearest")
    pk = np.ma.masked_where(~c["is_park"], np.ones_like(H))
    ax.imshow(pk, origin="lower", cmap=ListedColormap(["#2e7d32"]),
              extent=[0, L, 0, L], interpolation="nearest")
    ax.annotate("", xy=(70, 30), xytext=(20, 30),
                arrowprops=dict(arrowstyle="-|>", color="cyan", lw=2))
    ax.text(20, 42, "wind", color="cyan", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])

EX = [
 ("Low density / open\nlambda_p=0.25 H/W=0.5",
  dict(lambda_p=0.25, HW=0.5, sigma_h=0.1, theta_deg=0,  phi_open=0.12, conn=0.2, rho_c=0.5)),
 ("Compact canyons (skimming)\nlambda_p=0.50 H/W=1.6",
  dict(lambda_p=0.50, HW=1.6, sigma_h=0.1, theta_deg=0,  phi_open=0.10, conn=0.2, rho_c=0.5)),
 ("Tall+heterogeneous\nlambda_p=0.40 sigma_h=0.45",
  dict(lambda_p=0.40, HW=1.2, sigma_h=0.45, theta_deg=0, phi_open=0.10, conn=0.2, rho_c=0.5)),
 ("Rotated grid 30deg\n(oblique to wind)",
  dict(lambda_p=0.40, HW=1.0, sigma_h=0.15, theta_deg=30, phi_open=0.10, conn=0.2, rho_c=0.5)),
 ("Wind-aligned corridors\nphi_open=0.22 conn=0.9",
  dict(lambda_p=0.45, HW=1.2, sigma_h=0.15, theta_deg=0, phi_open=0.22, conn=0.9, rho_c=0.5)),
 ("Scattered parks\nphi_open=0.22 conn=0.1",
  dict(lambda_p=0.45, HW=1.2, sigma_h=0.15, theta_deg=0, phi_open=0.22, conn=0.1, rho_c=0.5)),
]

fig, axes = plt.subplots(2, 3, figsize=(13, 9))
print(f"{'example':<26}{'street W':>9}{'block B':>9}{'pitch':>7}  regime")
for ax, (title, prm) in zip(axes.ravel(), EX):
    c = build_city(seed=1, **prm)
    draw(ax, c, title)
    print(f"{title.splitlines()[0]:<26}{c['W']:8.1f}m{c['B']:8.1f}m{c['P']:6.0f}m  {c['regime']}")
handles = [mpatches.Patch(color="#2e7d32", label="open space / park"),
           mpatches.Patch(color="#bb3754", label="low building"),
           mpatches.Patch(color="#fcffa4", label="tall building")]
fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9)
fig.suptitle("Aggregate-parameter grid-city generator — morphology spanned by 6 robust knobs",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0.04, 1, 0.96])
fig.savefig("/mnt/user-data/outputs/city_param_demo.png", dpi=140, bbox_inches="tight")
print("\nsaved city_param_demo.png")
