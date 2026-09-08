"""verify_redundancy.py — demonstrate parameter redundancies in the v2 builder.

Each test runs gen_zoning_demo with two (or more) DIFFERENT parameter vectors and
checks whether they map to the SAME city (identical, or identical up to a domain
symmetry). Identical output from different inputs == redundant variation.
"""
import os, subprocess, collections

BIN = "./gen_zoning_demo"
TMP = "/tmp/redun"
os.makedirs(TMP, exist_ok=True)

def run(tag, morph, biz, park):
    out = os.path.join(TMP, f"{tag}.txt")
    cmd = f"{BIN} {morph} {biz} {park} {out}"
    r = subprocess.run(cmd.split(), capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return out

def parse(path):
    """Return {(col,row): usage} plus ordered usage list."""
    cells = {}
    xs, ys = set(), set()
    rows = []
    for line in open(path):
        if not line.startswith("B "): continue
        t = line.split()
        bx0, by0 = int(t[5]), int(t[6]); usage = int(t[10])
        rows.append((bx0, by0, usage)); xs.add(bx0); ys.add(by0)
    xmap = {v: i for i, v in enumerate(sorted(xs))}
    ymap = {v: i for i, v in enumerate(sorted(ys))}
    grid = {(xmap[bx], ymap[by]): u for bx, by, u in rows}
    return grid, len(xmap), len(ymap)

def hist(grid):
    return collections.Counter(grid.values())

# Standard morphology, roughness=0 so zoning is clean (roughness only perturbs
# heights, which do not feed the business/park SELECTION).
STD0 = "40 32 55 6e-6 0 0.15 0.0 20"
SQ0  = "40 40 55 6e-6 0 0.15 0.0 20"   # square blocks → 90°-symmetric grid

print("="*70)
print("TEST 1 — global 90° rotation:  (biz⟂green) is orientation-redundant")
print("="*70)
# A: biz corridor θ=0 (E-W), park corridor θ=90 (N-S)
# B: biz corridor θ=90 (N-S), park corridor θ=0 (E-W)   [= A rotated 90°]
a = run("rotA", SQ0, "0 0 0 0 0 1 0 0",  "0.5 0 0 0 0 1 90 0")
b = run("rotB", SQ0, "0 0 0 0 0 1 90 0", "0.5 0 0 0 0 1 0 0")
ga, nx, ny = parse(a); gb, _, _ = parse(b)

# the 8 dihedral maps of a square grid (n×n)
def dihedral(grid, n):
    maps = {
        "id":  lambda c, r: (c, r),
        "r90": lambda c, r: (r, n-1-c),
        "r180":lambda c, r: (n-1-c, n-1-r),
        "r270":lambda c, r: (n-1-r, c),
        "mx":  lambda c, r: (n-1-c, r),
        "my":  lambda c, r: (c, n-1-r),
        "d":   lambda c, r: (r, c),
        "ad":  lambda c, r: (n-1-r, n-1-c),
    }
    return {name: {f(c, r): u for (c, r), u in grid.items()} for name, f in maps.items()}

def frac_match(g1, g2):
    keys = set(g1) & set(g2)
    return sum(g1[k] == g2[k] for k in keys) / max(1, len(keys))

variants = dihedral(ga, nx)
best = max(variants, key=lambda nm: frac_match(variants[nm], gb))
print(f"  histograms A={dict(hist(ga))}  B={dict(hist(gb))}  equal={hist(ga)==hist(gb)}")
print(f"  best dihedral map of A onto B = '{best}'  cell-match = {frac_match(variants[best], gb):.1%}")
# business-only (usage code 1, no jitter) under that map
BIZ = 1
ba = {k for k, u in variants[best].items() if u == BIZ}
bb = {k for k, u in gb.items() if u == BIZ}
print(f"  business blocks match under '{best}': {ba == bb}  (|A|={len(ba)}, |B|={len(bb)})")

print("="*70)
print("TEST 2 — scale invariance of Φ weights at scatter=0")
print("="*70)
# Same field DIRECTION, different magnitudes → identical ranking → identical city
a = run("scl1", STD0, "1 0 0 0 0 0 0 0", "0.5 0 0 0 0 0 0 0")
b = run("scl5", STD0, "5 0 0 0 0 0 0 0", "0.5 0 0 0 0 0 0 0")
c = run("sclr", STD0, "1 0.5 0 0 0 0 0 0", "0.5 0 0 0 0 0 0 0")
d = run("sclr2",STD0, "2 1.0 0 0 0 0 0 0", "0.5 0 0 0 0 0 0 0")
ga,_,_=parse(a); gb,_,_=parse(b); gc,_,_=parse(c); gd,_,_=parse(d)
print(f"  w_radial 1 vs 5 (pure radial):        identical city = {ga==gb}")
print(f"  (1,0.5) vs (2,1.0) (same ratio):      identical city = {gc==gd}")

print("="*70)
print("TEST 3 — park_centrality magnitude is redundant at park_scatter=0")
print("="*70)
# Only the SIGN of a single-term weight matters when it is the sole ranking term
outs = {}
for pc in ["0.75", "0.90", "1.00"]:
    outs[pc] = parse(run(f"pc{pc}", STD0, "1 0 0 0 0 0 0 0", f"{pc} 0 0 0 0 0 0 0"))[0]
allsame = outs["0.75"] == outs["0.90"] == outs["1.00"]
print(f"  park_centrality 0.75 == 0.90 == 1.00 (scatter=0): identical parks = {allsame}")

print("="*70)
print("TEST 4 — near-flat field: tiny weights swamped by scatter")
print("="*70)
# With scatter=1 the maximin distance term dominates; tiny radial weights vanish
a = parse(run("flatA", STD0, "0.02 0 0 0 0 0 0 1", "0.5 0 0 0 0 0 0 1"))[0]
b = parse(run("flatB", STD0, "0.05 0 0 0 0 0 0 1", "0.5 0 0 0 0 0 0 1"))[0]
print(f"  phi_radial 0.02 vs 0.05 at biz_scatter=1: identical city = {a==b}")

print("="*70)
print("TEST 5 — grid transpose:  block_w↔block_d is a 90° rotation")
print("="*70)
# 40×32 blocks vs 32×40 blocks, same everything else → transpose of each other
a = run("gA", "40 32 55 6e-6 0 0.15 0.0 20", "1 0 0 0 0 0 0 0", "0.5 0 0 0 0 0 0 0")
b = run("gB", "32 40 55 6e-6 0 0.15 0.0 20", "1 0 0 0 0 0 0 0", "0.5 0 0 0 0 0 0 0")
ha = hist(parse(a)[0]); hb = hist(parse(b)[0])
print(f"  usage histogram 40×32 == 32×40: {ha==hb}   ({dict(ha)})")
