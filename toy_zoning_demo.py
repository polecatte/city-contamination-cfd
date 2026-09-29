#!/usr/bin/env python3
"""toy_zoning_demo.py — geometry-then-zoning, end to end, on a toy transport model.

    python3 toy_zoning_demo.py [OUT_DIR]        (numpy, scipy, matplotlib; ~1 min)

The question: GIVEN a fixed city geometry, where should its fixed population live and work
so that exposure to an accidental ground-level release is lowest?  And how do we keep the
optimizer from "solving" it by moving everyone away from one assumed wind or source?

Pipeline (each stage is a stand-in for the production one, named in brackets):
  1. geometry     fixed blocks, heights, floors, parks, streets        [gen_openlb_geom]
  2. scenarios    wind directions x release patterns                   [urban_flow runs]
  3. dose         steady advection-diffusion-removal per scenario      [urban_flow STEP4]
                  (for a linear, time-invariant model the steady field of a unit continuous
                  source IS the time-integrated dose Theta of a unit burst)
  4. per-floor dose d[s, floor] = indoor + outdoor microenvironments    [EXPOSURE_METRIC.md]
  5. zoning LP    min sum_floor n*d_train  s.t. population fixed, per-floor capacity band,
                  relocation budget                                      [the inner problem]
  6. validation   score on HELD-OUT wind directions and release patterns [the guard]

TOY MODEL — NOT CFD. The 2-D transport below has the right ingredients (street-aligned
channelling, canyon trapping via H/W, vertical ventilation, a decaying vertical profile) but
none of it is calibrated. The point is the method: what the optimizer does, how it cheats,
and how the held-out test tells the two apart.
"""
import os, sys
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from scipy.optimize import linprog
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

OUT = sys.argv[1] if len(sys.argv) > 1 else "toy_zoning"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(7)

# ───────────────────────────── 1. geometry ─────────────────────────────
DX = 4.0                         # m per cell
NB = 8                           # 8 x 8 blocks
BLK = 9                          # block edge in cells (36 m)
ST = 3                           # ordinary street (12 m)
BLV = 8                          # boulevard (32 m) after block index 3 in x and in y
MARGIN = 20                      # 80 m open fetch around the city
FLOOR_H = 3.5                    # m per storey
M2_PER_PERSON = 30.0

def axis_layout():
    pos, x = [], MARGIN
    for i in range(NB):
        pos.append((x, x + BLK)); x += BLK
        if i < NB - 1: x += BLV if i == 3 else ST
    return pos, x + MARGIN
bpos, N = axis_layout()

solid = np.zeros((N, N), bool)          # [x, y]
height = np.zeros((N, N))
park = np.zeros((N, N), bool)
blocks = []
parks = {(1, 6), (6, 1), (5, 5), (2, 2)}
for i, (x0, x1) in enumerate(bpos):
    for j, (y0, y1) in enumerate(bpos):
        if (i, j) in parks:
            park[x0:x1, y0:y1] = True
            continue
        r = np.hypot(i - 3.5, j - 3.5) / 5.0          # taller core, log-normal scatter
        h = float(np.clip(12 + 60 * np.exp(-3 * r * r) * rng.lognormal(0, 0.45), 9, 110))
        nf = int(h // FLOOR_H)
        h = nf * FLOOR_H
        solid[x0:x1, y0:y1] = True; height[x0:x1, y0:y1] = h
        cap_floor = (BLK * DX) ** 2 * 0.8 / M2_PER_PERSON
        blocks.append(dict(i=i, j=j, x0=x0, x1=x1, y0=y0, y1=y1, h=h, nf=nf, cap=cap_floor))
open_ = ~solid
city = np.zeros((N, N), bool); city[bpos[0][0]:bpos[-1][1], bpos[0][0]:bpos[-1][1]] = True

# street orientation: an x-running street cell sits between blocks in y, etc.
xrun = np.zeros((N, N), bool); yrun = np.zeros((N, N), bool)
xs_in = np.zeros(N, bool); ys_in = np.zeros(N, bool)
for (a, b) in bpos: xs_in[a:b] = True; ys_in[a:b] = True
for x in range(N):
    for y in range(N):
        if not open_[x, y] or park[x, y] or not city[x, y]: continue
        if xs_in[x] and not ys_in[y]: xrun[x, y] = True     # between block rows -> runs along x
        elif ys_in[y] and not xs_in[x]: yrun[x, y] = True
        elif not xs_in[x] and not ys_in[y]: xrun[x, y] = yrun[x, y] = True   # intersection

# canyon openness W/H for the vertical exchange: street width over neighbouring height
def local_H(x, y, rad=4):
    sl = height[max(0, x - rad):x + rad + 1, max(0, y - rad):y + rad + 1]
    return sl.max()
width = np.where(xrun & ~yrun, 0, 0).astype(float)
openness = np.ones((N, N))
for x in range(N):
    for y in range(N):
        if open_[x, y] and city[x, y] and not park[x, y]:
            w = (BLV if (bpos[3][1] <= x < bpos[4][0] or bpos[3][1] <= y < bpos[4][0]) else ST) * DX
            H = local_H(x, y)
            openness[x, y] = min(1.0, w / max(H, 1.0))

# ───────────────────────────── 2. scenarios ─────────────────────────────
U = 3.0                                                # m/s above-canopy reference
K = 2.0                                                # m^2/s eddy diffusivity
LAM0 = 0.03                                            # 1/s vertical exchange when fully open
LZ = 20.0                                              # m, vertical decay scale of a ground release

src_uniform = (open_ & city).astype(float)             # every open ground cell (the project's Omega)
src_traffic = np.zeros((N, N))                         # the two boulevards + ring street
bx0, bx1 = bpos[3][1], bpos[4][0]
src_traffic[bx0:bx1, bpos[0][0]:bpos[-1][1]] = 1; src_traffic[bpos[0][0]:bpos[-1][1], bx0:bx1] = 1
src_traffic *= open_
def src_clusters(seed, n=5, rad=4):
    r = np.random.default_rng(seed); s = np.zeros((N, N))
    cand = np.argwhere(open_ & city)
    for k in r.choice(len(cand), n, replace=False):
        cx, cy = cand[k]; s[max(0, cx - rad):cx + rad, max(0, cy - rad):cy + rad] = 1
    return s * open_
SOURCES = {"uniform": src_uniform, "traffic": src_traffic,
           "clusters_a": src_clusters(1), "clusters_b": src_clusters(2), "clusters_c": src_clusters(3)}
# independent draws from the SAME cluster family, used only for training (never tested on)
for k, seed in enumerate(range(11, 16)): SOURCES[f"train_clusters_{k}"] = src_clusters(seed)

# wind rose: direction the wind blows FROM (met. convention), prevailing westerly
ROSE = {270: .24, 247.5: .10, 292.5: .09, 225: .08, 315: .07, 180: .06, 202.5: .06, 337.5: .05,
        0: .04, 90: .04, 45: .04, 135: .04, 22.5: .03, 67.5: .02, 112.5: .02, 157.5: .02}
wsum = sum(ROSE.values()); ROSE = {k: v / wsum for k, v in ROSE.items()}

# ───────────────────────────── 3. dose solver ─────────────────────────────
idx = -np.ones((N, N), int); cells = np.argwhere(open_); idx[open_] = np.arange(len(cells))
def solve(wind_from_deg, src):
    th = np.deg2rad(270.0 - wind_from_deg)             # blowing-towards angle, +x = from west
    ux0, uy0 = U * np.cos(th), U * np.sin(th)
    # street channelling: the along-street component survives, the cross-street one is damped
    ax = np.where(xrun | ~city | park, 1.0, 0.25); ay = np.where(yrun | ~city | park, 1.0, 0.25)
    ux, uy = ux0 * ax, uy0 * ay
    lam = LAM0 * openness + 1e-4
    n = len(cells); rows, cols, vals = [], [], []
    diag = np.zeros(n); rhs = np.zeros(n)
    inv = 1.0 / DX
    for k, (x, y) in enumerate(cells):
        d = lam[x, y]
        for (dx_, dy_, uc) in ((1, 0, ux[x, y]), (-1, 0, -ux[x, y]), (0, 1, uy[x, y]), (0, -1, -uy[x, y])):
            xx, yy = x + dx_, y + dy_
            inside = 0 <= xx < N and 0 <= yy < N
            # diffusion (zero-flux into buildings, C=0 outside the domain)
            if inside and open_[xx, yy]:
                d += K * inv * inv; rows.append(k); cols.append(idx[xx, yy]); vals.append(-K * inv * inv)
            elif not inside:
                d += K * inv * inv
            # upwind advection u.grad C: upstream neighbour is at -u direction
            if uc < 0:                                  # flow comes from (xx,yy)
                a = -uc * inv; d += a
                if inside and open_[xx, yy]:
                    rows.append(k); cols.append(idx[xx, yy]); vals.append(-a)
        diag[k] = d; rhs[k] = src[x, y]
    A = sp.csr_matrix((vals, (rows, cols)), shape=(n, n)) + sp.diags(diag)
    c = spsolve(A.tocsc(), rhs)
    C = np.zeros((N, N)); C[open_] = c
    return C

# ─────────────── 4. per-floor dose (two microenvironments) ───────────────
F_IN, F_OUT, F_INF = 0.87, 0.075, 0.62
floors = []                                            # (block index, floor k, capacity)
for b, B in enumerate(blocks):
    for k in range(B["nf"]): floors.append((b, k, B["cap"]))
FB = np.array([f[0] for f in floors]); FK = np.array([f[1] for f in floors]); CAP = np.array([f[2] for f in floors])
def envelope(B):
    m = np.zeros((N, N), bool)
    m[max(0, B["x0"] - 1):B["x1"] + 1, max(0, B["y0"] - 1):B["y1"] + 1] = True
    return m & open_
ENV = [envelope(B) for B in blocks]
def floor_dose(C):
    env = np.array([C[m].mean() for m in ENV])         # ground-level envelope dose per block
    z = (FK + 0.5) * FLOOR_H
    return F_IN * F_INF * env[FB] * np.exp(-z / LZ) + F_OUT * env[FB]

print(f"geometry: {N}x{N} cells ({N*DX:.0f} m), {len(blocks)} buildings, {len(floors)} floors, "
      f"capacity {CAP.sum():.0f} people")
scen = [(w, s) for w in ROSE for s in SOURCES]
D = {}; FIELDS = {}
for (w, s) in scen:
    C = solve(w, SOURCES[s]); C /= SOURCES[s].sum()     # per unit released mass
    FIELDS[(w, s)] = C; D[(w, s)] = floor_dose(C)
print(f"solved {len(scen)} scenarios ({len(ROSE)} wind directions x {len(SOURCES)} release patterns)")

# ───────────────────────────── 5. zoning LP ─────────────────────────────
POP = 0.6 * CAP.sum()
n0 = POP * CAP / CAP.sum()                              # baseline: occupancy proportional to capacity
LO, HI = 0.3, 1.0                                       # per-floor occupancy band (fraction of capacity)
def optimize(dvec, budget_frac=0.3):
    """min d.n  s.t. sum n = POP, LO*cap <= n <= HI*cap, sum |n - n0| <= 2*budget*POP."""
    m = len(dvec)
    # variables: n (m), t (m) with t >= |n - n0|
    c = np.concatenate([dvec, np.zeros(m)])
    A_ub = sp.vstack([sp.hstack([sp.eye(m), -sp.eye(m)]), sp.hstack([-sp.eye(m), -sp.eye(m)]),
                      sp.hstack([sp.csr_matrix((1, m)), sp.csr_matrix(np.ones((1, m)))])])
    b_ub = np.concatenate([n0, -n0, [2 * budget_frac * POP]])
    A_eq = sp.hstack([sp.csr_matrix(np.ones((1, m))), sp.csr_matrix((1, m))])
    bounds = [(LO * cp, HI * cp) for cp in CAP] + [(0, None)] * m
    r = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[POP], bounds=bounds, method="highs")
    assert r.success, r.message
    return r.x[:m]
def expected(scens, weights=None):
    ws = np.array([ROSE[w] for (w, s) in scens]) if weights is None else np.asarray(weights)
    return sum(wi * D[sc] for wi, sc in zip(ws, scens)) / ws.sum()
def J(n, sc): return float(n @ D[sc]) / POP             # dose per person

# train / test split
TRAIN_W = [w for w in ROSE if float(w) % 45 == 0]       # the 8 cardinal/intercardinal directions
TEST_W = [w for w in ROSE if float(w) % 45 != 0]        # the 8 in between: never seen in training
TRAIN_S = ["uniform", "traffic"]; TEST_S = ["clusters_a", "clusters_b", "clusters_c"]
TRAIN_CL = [s for s in SOURCES if s.startswith("train_clusters")]
test_scen = [(w, s) for w in TEST_W for s in TEST_S]
tw = np.array([ROSE[w] for (w, s) in test_scen])
# two held-out tests, to separate "new wind" from "new kind of release":
TEST_A = [(w, s) for w in TEST_W for s in TRAIN_S]      # unseen winds, familiar release types
TEST_B = test_scen                                       # unseen winds AND unseen cluster draws

strategies = {
    "naive (one wind, W, uniform source)": optimize(D[(270, "uniform")]),
    "robust (8-direction rose x 2 source patterns)": optimize(expected([(w, s) for w in TRAIN_W for s in TRAIN_S])),
    "robust+clusters (rose x uniform, traffic, 5 cluster draws)":
        optimize(expected([(w, s) for w in TRAIN_W for s in TRAIN_S + TRAIN_CL])),
}
# oracle: the best zoning had we known each test scenario exactly (for regret)
oracle = {sc: J(optimize(D[sc]), sc) for sc in test_scen}

def gain(n, scs):   # % reduction vs baseline, per scenario
    return np.array([100 * (1 - J(n, sc) / J(n0, sc)) for sc in scs])
print("\nheld-out = 8 unseen wind directions x 3 unseen release patterns (24 scenarios), rose-weighted")
report = {}
for name, n in strategies.items():
    g_train = 100 * (1 - (n @ D[(270, 'uniform')]) / (n0 @ D[(270, 'uniform')]))
    g = gain(n, test_scen); worst = g.min()
    regret = np.array([J(n, sc) / oracle[sc] - 1 for sc in test_scen]) * 100
    report[name] = (g, regret)
    print(f"  {name}\n      gain on the W/uniform case: {g_train:5.1f} %   "
          f"held-out: mean {np.average(g, weights=tw):5.1f} %, worst {worst:5.1f} %, "
          f"scenarios made WORSE: {(g < 0).sum()}/{len(g)}, mean regret vs oracle {np.average(regret, weights=tw):5.1f} %")


print("\nseparating the two kinds of novelty (mean held-out reduction; 'flat' = every direction weighted equally)")
print(f"  {'':58s} {'A: new winds':>14s} {'A flat':>8s} {'B: new winds+draws':>19s} {'B flat':>8s} {'B worse':>8s}")
for name, n in strategies.items():
    ga, gb = gain(n, TEST_A), gain(n, TEST_B)
    wa = np.array([ROSE[w] for (w, s) in TEST_A]); wb = np.array([ROSE[w] for (w, s) in TEST_B])
    print(f"  {name:58s} {np.average(ga, weights=wa):13.1f}% {ga.mean():7.1f}% {np.average(gb, weights=wb):18.1f}% "
          f"{gb.mean():7.1f}% {(gb < 0).sum():5d}/{len(gb)}")
nn = strategies["naive (one wind, W, uniform source)"]
print("  naive zoning by held-out wind direction (uniform release): " +
      ", ".join(f"{w:g}° {100*(1-J(nn,(w,'uniform'))/J(n0,(w,'uniform'))):+.0f}%" for w in TEST_W))

# covariance decomposition on one held-out scenario and on the held-out average
def decomp(n, sc):
    d = D[sc]; W = POP
    mean_d = d.mean()                               # average over floors = where people COULD be
    per_person = n @ d / W
    return mean_d, per_person - mean_d
print("\ncovariance decomposition, per-person dose = mean over floors + alignment term (x1e-3)")
for name, n in [("baseline", n0)] + list(strategies.items()):
    md = np.average([decomp(n, sc)[0] for sc in test_scen], weights=tw)
    al = np.average([decomp(n, sc)[1] for sc in test_scen], weights=tw)
    print(f"  {name:58s} mean {1e3*md:.4f}  alignment {1e3*al:+.4f}")

# what did each zoning actually do? between buildings vs between floors, and where
def split_gain(n, scs):
    """Between-buildings part: same building totals, floors filled proportionally (no vertical
    sorting). The rest of the gain is the between-floors part."""
    w_ = np.array([ROSE[w] for (w, s) in scs])
    tot = np.bincount(FB, weights=n, minlength=len(blocks)); tot0 = np.bincount(FB, weights=n0, minlength=len(blocks))
    n_h = n0 * (tot / tot0)[FB]
    gh = np.average(gain(n_h, scs), weights=w_); gt = np.average(gain(n, scs), weights=w_)
    return gh, gt - gh
dstreet = np.array([openness[ENV[b]].mean() for b in range(len(blocks))])
print("\nwhere the held-out gain comes from (test B), and what kind of building gains people")
for name, n in strategies.items():
    gh, gv = split_gain(n, TEST_B)
    tot = np.bincount(FB, weights=n, minlength=len(blocks)); tot0 = np.bincount(FB, weights=n0, minlength=len(blocks))
    up, dn = tot > tot0 * 1.02, tot < tot0 * 0.98
    print(f"  {name:58s} between buildings {gh:+5.1f} %, between floors {gv:+5.1f} %;  "
          f"street W/H of gaining vs losing buildings {dstreet[up].mean():.2f} vs {dstreet[dn].mean():.2f}")
budgets = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8]
wB = np.array([ROSE[w] for (w, s) in TEST_B])
BCURVE = {}
for name, tr in (("robust", TRAIN_S), ("robust+clusters", TRAIN_S + TRAIN_CL)):
    dtr = expected([(w, s) for w in TRAIN_W for s in tr])
    BCURVE[name] = [np.average(gain(optimize(dtr, b), TEST_B), weights=wB) for b in budgets]
    print(f"  {name:16s} held-out gain vs relocation budget: " +
          ", ".join(f"{int(100*b)}% -> {g:.1f}%" for b, g in zip(budgets, BCURVE[name])))

# ───────────────────────────── figures ─────────────────────────────
ext = [0, N * DX, 0, N * DX]
def base_axes(ax, title):
    hm = np.ma.masked_where(~solid, height).T
    ax.imshow(np.where(park, 1.0, np.nan).T, origin="lower", extent=ext, cmap="Greens", vmin=0, vmax=1.5)
    ax.imshow(hm, origin="lower", extent=ext, cmap="Greys", vmin=0, vmax=120)
    ax.set_title(title, fontsize=10); ax.set_xticks([]); ax.set_yticks([])

fig, axs = plt.subplots(1, 3, figsize=(16, 5.6))
ax = axs[0]; hm = np.ma.masked_where(~solid, height).T
ax.imshow(np.where(park, 1.0, np.nan).T, origin="lower", extent=ext, cmap="Greens", vmin=0, vmax=1.5)
im = ax.imshow(hm, origin="lower", extent=ext, cmap="plasma", vmin=0, vmax=110)
ax.contour(np.arange(N) * DX, np.arange(N) * DX, src_traffic.T, levels=[0.5], colors="#e8590c", linewidths=1)
ax.set_title("1. fixed geometry: 60 buildings (colour = height),\n"
             "4 parks (green), boulevards (orange = 'traffic' release)", fontsize=10); ax.set_xticks([]); ax.set_yticks([])
fig.colorbar(im, ax=ax, shrink=0.8, label="height (m)")
ax = axs[1]; ax.remove(); ax = fig.add_subplot(1, 3, 2, projection="polar")
th = np.deg2rad([w for w in ROSE]); ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
ax.bar(th, [ROSE[w] for w in ROSE], width=np.deg2rad(20),
       color=["#4263eb" if w in TRAIN_W else "#fab005" for w in ROSE], alpha=.85)
ax.set_title("2. wind rose (direction wind comes FROM)\nblue = training, yellow = held out", fontsize=10)
ax = axs[2]; base_axes(ax, "2. release patterns: grey = uniform Omega (training),\nred = held-out clusters")
for s, col in (("clusters_a", "#c92a2a"), ("clusters_b", "#e03131"), ("clusters_c", "#fa5252")):
    ax.contourf(np.arange(N) * DX, np.arange(N) * DX, SOURCES[s].T, levels=[0.5, 1.5], colors=[col], alpha=.8)
fig.tight_layout(); fig.savefig(f"{OUT}/1_geometry_scenarios.png", dpi=110); plt.close(fig)

fig, axs = plt.subplots(1, 3, figsize=(15, 5.6))
for ax, sc, t in zip(axs, [(270, "uniform"), (0, "uniform"), (247.5, "clusters_a")],
                     ["wind from W, uniform release", "wind from N, uniform release", "wind from WSW, clusters (held out)"]):
    C = np.ma.masked_where(~open_, FIELDS[sc]).T
    im = ax.imshow(C, origin="lower", extent=ext, cmap="magma_r", vmax=np.percentile(FIELDS[sc][open_], 99))
    ax.imshow(np.ma.masked_where(~solid, height).T, origin="lower", extent=ext, cmap="Greys", vmin=-50, vmax=120)
    ax.set_title(f"3. ground-level dose\n{t}", fontsize=10); ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout(); fig.savefig(f"{OUT}/2_dose_fields.png", dpi=110); plt.close(fig)

SHORT = ["naive", "robust", "robust+clusters"]
fig, axs = plt.subplots(2, 2, figsize=(13, 11))
for ax, (name, n), sh in zip(axs.flat[:3], strategies.items(), SHORT):
    delta = np.bincount(FB, weights=n - n0, minlength=len(blocks))
    img = np.full((N, N), np.nan)
    for b, B in enumerate(blocks): img[B["x0"]:B["x1"], B["y0"]:B["y1"]] = delta[b]
    ax.imshow(np.where(park, 1.0, np.nan).T, origin="lower", extent=ext, cmap="Greens", vmin=0, vmax=1.5)
    im = ax.imshow(img.T, origin="lower", extent=ext, cmap="RdBu", norm=TwoSlopeNorm(0, -400, 400))
    ax.set_title(f"5. {sh}: {name.split('(')[1][:-1]}\npeople moved per building (blue gains, red loses)", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    ax.annotate("prevailing wind", xy=(N * DX * 0.3, N * DX * 0.05), xytext=(N * DX * 0.02, N * DX * 0.05),
                va="center", fontsize=8, arrowprops=dict(arrowstyle="->", lw=2))
    fig.colorbar(im, ax=ax, shrink=0.75, label="Δ people")
ax = axs[1, 1]
xs = np.arange(len(strategies))
for off, (lab, scs, col) in zip((-0.18, 0.18), (("A: unseen winds, familiar releases", TEST_A, "#4263eb"),
                                                ("B: unseen winds + unseen cluster draws", TEST_B, "#e8590c"))):
    for i, n in enumerate(strategies.values()):
        g = gain(n, scs)
        ax.scatter(np.full(len(g), i + off) + rng.normal(0, .025, len(g)), g, s=9, c=col, alpha=.5,
                   label=lab if i == 0 else None)
        ax.plot([i + off - .1, i + off + .1], [g.mean()] * 2, c="k", lw=2)
ax.axhline(0, color="k", lw=0.8); ax.set_xticks(xs); ax.set_xticklabels(SHORT)
ax.set_ylabel("exposure reduction vs baseline (%)"); ax.legend(fontsize=8, loc="lower left")
ax.set_title("6. held-out tests (dots = scenarios, bars = flat means)\nbelow 0 = zoning made that scenario worse", fontsize=10)
fig.tight_layout(); fig.savefig(f"{OUT}/3_zoning_and_validation.png", dpi=110); plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(13, 4.4))
ax = axs[0]
labs = ["baseline"] + SHORT
md = [np.average([decomp(n, sc)[0] for sc in test_scen], weights=tw) for n in [n0] + list(strategies.values())]
al = [np.average([decomp(n, sc)[1] for sc in test_scen], weights=tw) for n in [n0] + list(strategies.values())]
pp = 1e3 * (np.array(md) + np.array(al))
ax.bar(labs, pp, color=["#adb5bd", "#4263eb", "#4263eb", "#4263eb"])
ax.axhline(1e3 * md[0], ls="--", c="k", lw=1, label="mean dose over all floors (set by geometry, identical for all)")
for i, (p_, a_) in enumerate(zip(pp, al)):
    ax.text(i, p_ + 0.01, f"alignment {1e3*a_:+.3f}", ha="center", fontsize=8)
ax.set_ylim(0.8 * pp.min(), 1.08 * 1e3 * md[0])
ax.set_ylabel("dose per person, test B (x1e-3)"); ax.legend(fontsize=8, loc="lower left")
ax.set_title("per-person dose = geometry's mean dose + alignment (zoning)\nzoning only ever moves the alignment term", fontsize=10)
ax = axs[1]
for name, col in (("robust", "#4263eb"), ("robust+clusters", "#e8590c")):
    ax.plot([100 * b for b in budgets], BCURVE[name], "o-", c=col, label=name)
ax.set_xlabel("relocation budget (% of population moved)"); ax.set_ylabel("held-out (B) exposure reduction (%)")
ax.set_title("value of zoning vs how much of the city may change", fontsize=10); ax.grid(alpha=.3); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/4_decomposition_budget.png", dpi=110); plt.close(fig)
print(f"\nfigures in {OUT}/")
