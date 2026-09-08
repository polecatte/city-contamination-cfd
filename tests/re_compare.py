"""re_compare.py — is the dispersion Reynolds-independent?

Loads the per-floor z=1 TIAC fields produced by re_test.sh, normalizes each by
its emitted mass (so we compare the SHAPE of the plume, not its magnitude), and
measures how much the normalized pattern changes as the effective Reynolds number
varies. Reports the standard dispersion-model agreement metrics against the
highest-Re (lowest-floor) reference:
  FAC2  — fraction of cells within a factor of 2 (Chang & Hanna 2004; >0.5 is
          "good" between models, so >0.8 across a 6x Re span = effectively
          Re-independent for this output).
  R     — Pearson correlation of the normalized fields.
  NMSE  — normalized mean-square error.
and overlays the plume centerline decay (max over the cross-stream direction vs
downwind distance) — if the decay rate is invariant to Re, the mixing is too.

Usage:  python3 re_compare.py disp_tau0.505.bin disp_tau0.515.bin disp_tau0.530.bin
"""
import sys, os, struct, re
import numpy as np
import matplotlib.pyplot as plt

def load(fn):
    with open(fn, "rb") as f:
        nx, ny = struct.unpack("2i", f.read(8))
        a = np.frombuffer(f.read(nx*ny*4), dtype=np.float32).reshape(ny, nx).astype(np.float64)
    return a

files = sys.argv[1:]
if len(files) < 2:
    sys.exit("need >=2 fields, e.g.: python3 re_compare.py disp_tau*.bin")
def taukey(fn):
    m = re.search(r"tau(\d+\.\d+)", fn); return float(m.group(1)) if m else 0.0
files = sorted(files, key=taukey)                      # ascending tau = descending Re
labels = [f"tau={taukey(f):.3f}" for f in files]

fields = [load(f) for f in files]
norm = [a / a.sum() if a.sum() > 0 else a for a in fields]   # compare shape
ref = norm[0]                                                 # highest Re (lowest tau)
ny, nx = ref.shape

def metrics(a, b):
    m = (a > 0) | (b > 0)
    if not m.any(): return (np.nan, np.nan, np.nan)
    x, y = a[m], b[m]
    both = (x > 0) & (y > 0)
    fac2 = np.mean(np.maximum(x[both]/y[both], y[both]/x[both]) <= 2.0) if both.any() else 0.0
    R = np.corrcoef(x, y)[0, 1]
    nmse = np.mean((x-y)**2) / (x.mean()*y.mean() + 1e-300)
    return fac2, R, nmse

print(f"\nReference (highest Re): {labels[0]}")
print(f"{'field':>12} {'FAC2':>7} {'R':>7} {'NMSE':>9}")
rows = []
for lab, a in zip(labels, norm):
    fac2, R, nmse = metrics(ref, a)
    rows.append((lab, fac2, R, nmse))
    print(f"{lab:>12} {fac2:7.3f} {R:7.3f} {nmse:9.3e}")

# verdict from the most-different (highest-tau) case vs the reference
worst = rows[-1]
re_indep = (worst[1] >= 0.8 and worst[2] >= 0.95)
print("\nVERDICT:", "effectively Re-INDEPENDENT for this output "
      "(normalized plume ~invariant across the Re span)" if re_indep else
      "Re-DEPENDENT — the normalized plume changes with the floor; "
      "bias is quantified above and the floor matters for absolute results.")

# ── figure: centerline decay + agreement scatter ──
fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
xs = np.arange(nx) * 4.0   # m (4 m cells)
for lab, a in zip(labels, norm):
    center = a.max(axis=0)                       # plume envelope vs downwind x
    ax[0].semilogy(xs, np.maximum(center, 1e-12), label=lab)
ax[0].set_xlabel("downwind x (m)"); ax[0].set_ylabel("normalized centerline conc. (log)")
ax[0].set_title("Plume centerline decay vs effective Re\n(curves collapsing = Re-independent)")
ax[0].legend(); ax[0].grid(alpha=0.3)

m = (ref > 0) | (norm[-1] > 0)
ax[1].loglog(np.maximum(ref[m], 1e-12), np.maximum(norm[-1][m], 1e-12), ".", ms=2, alpha=0.3)
lim = [1e-10, max(ref.max(), norm[-1].max())]
ax[1].plot(lim, lim, "k-", lw=1); ax[1].plot(lim, [2*l for l in lim], "r--", lw=0.8)
ax[1].plot(lim, [0.5*l for l in lim], "r--", lw=0.8, label="factor 2")
ax[1].set_xlabel(f"normalized conc. — {labels[0]} (ref)")
ax[1].set_ylabel(f"normalized conc. — {labels[-1]}")
ax[1].set_title(f"Cell-by-cell agreement\nFAC2={worst[1]:.3f}  R={worst[2]:.3f}")
ax[1].legend(loc="upper left")
fig.suptitle("Reynolds-independence test", fontsize=14, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.95])
out = os.environ.get("FIG_DIR", "figures"); os.makedirs(out, exist_ok=True)
plt.savefig(f"{out}/re_independence.png", dpi=150, bbox_inches="tight")
print(f"\nsaved {out}/re_independence.png")
