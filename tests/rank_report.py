#!/usr/bin/env python3
"""rank_report.py — does a spread of city designs give a spread of exposure larger than the noise,
and does the ranking survive a change of model?

    python3 tests/rank_report.py MANIFEST [MANIFEST ...]

Each manifest line:  config  design  seed  geom_dir  out_dir   (lab_openlb.sh rank writes them;
'#' starts a comment). A run without theta.f32 yet is listed as missing and skipped.

Per run: J = <w, Theta> / M_emitted (as stage_c_J.py) and the intake fraction
iF = B * J / dx^3 (B = 1.6e-4 m^3/s, an adult's mean breathing rate, ~14 m^3/day; dx^3 turns
lattice C into concentration), per million: the share of the released mass the city's
population inhales. iF is comparable across dx; J is not.

Per configuration:
  - design means of iF and the seed noise: pooled within-design standard deviation of iF over
    inlet seeds (sigma_noise), relative to the grand mean;
  - spread: (max - min) / mean of the design means, and the one-way ANOVA F = between-design
    variance / within-design variance with its p-value (F distribution, by numerical
    integration; no scipy needed);
  - distinguishable pairs: design pairs whose mean difference exceeds 2 * sqrt(2/n) * sigma_noise
    (about a 95 % two-sided test with n seeds each);
  - the ranking, lowest exposure first.
Across configurations (designs present in both): Spearman rho and Kendall tau of the design
means, and whether the best and worst designs agree.

"Meaningful spread" here: F well above 1 (p < 0.05) AND the spread several times the seed
noise. With 2 seeds the within-design variance has few degrees of freedom; add seeds if F is
borderline.
"""
import sys, os, math, itertools
from collections import defaultdict, OrderedDict
import numpy as np

B_BREATH = 1.6e-4   # m^3/s


def read5(fn, dtype):
    with open(fn, 'rb') as f:
        h = np.frombuffer(f.read(20), dtype=np.int32)
        a = np.frombuffer(f.read(), dtype=dtype)
    return h, a


def meta(path):
    d = {}
    for line in open(path):
        p = line.split()
        if len(p) == 2 and not p[0].startswith('#'):
            try: d[p[0]] = float(p[1])
            except ValueError: pass
    return d


def run_J(geom, out):
    hw, w = read5(os.path.join(geom, 'receptor_w.f32'), np.float32)
    ht, th = read5(os.path.join(out, 'theta.f32'), np.float32)
    if tuple(ht[:3]) != tuple(hw[:3]):
        raise ValueError(f"theta grid {ht[:3]} != receptor grid {hw[:3]}")
    m = meta(os.path.join(out, 'meta_flow.txt'))
    emit = m.get('mass_emitted_total', m['mass_emitted'])
    J = float(np.dot(w.astype(np.float64), th.astype(np.float64)) / emit)
    dx = m.get('dx_m', hw[3] / 1000.0)
    return J, B_BREATH * J / dx**3 * 1e6, m


def betainc_reg(a, b, x, n=4000):
    """regularised incomplete beta I_x(a,b) by Simpson's rule on t = x * s^(1/a) (smooth integrand)."""
    if x <= 0: return 0.0
    if x >= 1: return 1.0
    lnB = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    # integral_0^x t^(a-1)(1-t)^(b-1) dt with t = x*u^(1/a): = x^a/a * integral_0^1 (1 - x u^(1/a))^(b-1) du
    s = 0.0
    for i in range(n + 1):
        u = i / n
        f = (1 - x * u ** (1 / a)) ** (b - 1)
        s += f * (1 if i in (0, n) else (4 if i % 2 else 2))
    s *= 1 / (3 * n)
    return math.exp(a * math.log(x) - math.log(a) - lnB) * s


def f_pvalue(F, d1, d2):
    if not (F > 0) or d1 <= 0 or d2 <= 0: return float('nan')
    x = d2 / (d2 + d1 * F)
    return betainc_reg(d2 / 2, d1 / 2, x)


def ranks(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    for k, i in enumerate(order): r[i] = k + 1.0
    return r


def spearman(a, b):
    ra, rb = np.array(ranks(a)), np.array(ranks(b))
    return float(np.corrcoef(ra, rb)[0, 1]) if len(a) > 2 else float('nan')


def kendall(a, b):
    c = d = 0
    for i, j in itertools.combinations(range(len(a)), 2):
        s = np.sign(a[i] - a[j]) * np.sign(b[i] - b[j])
        c += s > 0; d += s < 0
    return (c - d) / max(1, c + d)


def main():
    if len(sys.argv) < 2:
        print(__doc__); return 2
    runs = defaultdict(lambda: defaultdict(list))     # cfg -> design -> [iF]
    order = OrderedDict()
    for man in sys.argv[1:]:
        for line in open(man):
            line = line.split('#')[0].split()
            if len(line) < 5: continue
            cfg, des, seed, geom, out = line[:5]
            order.setdefault(cfg, OrderedDict()).setdefault(des, None)
            if not os.path.exists(os.path.join(out, 'theta.f32')):
                print(f"[rank] {cfg} {des} seed {seed}: missing ({out})"); continue
            try:
                J, iF, m = run_J(geom, out)
            except Exception as e:
                print(f"[rank] {cfg} {des} seed {seed}: unreadable ({e})"); continue
            runs[cfg][des].append(iF)
            print(f"[rank] {cfg:10s} {des:8s} seed {seed}: iF = {iF:8.3f} ppm  (J {J:.4e}, "
                  f"closure {m.get('budget_closure', float('nan')):.1e}, steps {int(m.get('burst_steps', 0))})")
    means = {}
    for cfg in order:
        D = [d for d in order[cfg] if runs[cfg][d]]
        if len(D) < 2:
            print(f"[rank] {cfg}: fewer than two designs done"); continue
        mu = {d: float(np.mean(runs[cfg][d])) for d in D}
        means[cfg] = mu
        grand = float(np.mean([x for d in D for x in runs[cfg][d]]))
        ss_w = sum(((np.array(runs[cfg][d]) - mu[d]) ** 2).sum() for d in D)
        df_w = sum(len(runs[cfg][d]) - 1 for d in D)
        n_all = sum(len(runs[cfg][d]) for d in D)
        ss_b = sum(len(runs[cfg][d]) * (mu[d] - grand) ** 2 for d in D)
        df_b = len(D) - 1
        sig = math.sqrt(ss_w / df_w) if df_w > 0 else float('nan')
        spread = (max(mu.values()) - min(mu.values())) / grand
        print(f"\n[rank] == {cfg}: {len(D)} designs, {n_all} runs; grand mean iF {grand:.3f} ppm")
        for d in sorted(D, key=lambda d: mu[d]):
            v = runs[cfg][d]
            print(f"[rank]    {d:8s} iF {mu[d]:8.3f} ppm  ({100*(mu[d]/grand-1):+6.1f} % of mean; seeds: "
                  + ", ".join(f"{x:.3f}" for x in v) + ")")
        print(f"[rank]    spread (max-min)/mean = {100*spread:.1f} %")
        if df_w > 0 and sig > 0:
            F = (ss_b / df_b) / (ss_w / df_w)
            p = f_pvalue(F, df_b, df_w)
            n_seed = n_all / len(D)
            thr = 2 * math.sqrt(2 / n_seed) * sig
            pairs = list(itertools.combinations(D, 2))
            dist = sum(abs(mu[a] - mu[b]) > thr for a, b in pairs)
            print(f"[rank]    seed noise sigma = {sig:.3f} ppm ({100*sig/grand:.2f} % of mean); spread / noise = {spread*grand/sig:.1f}")
            print(f"[rank]    ANOVA F({df_b},{df_w}) = {F:.1f}, p = {p:.2g}; distinguishable pairs {dist}/{len(pairs)} "
                  f"(|diff| > {thr:.3f} ppm)")
            ok = p < 0.05 and spread * grand > 3 * sig
            print(f"[rank]    meaningful spread: {'YES' if ok else 'NO'} (p < 0.05 and spread > 3 sigma)")
        else:
            print("[rank]    seed noise: needs >= 2 seeds of some design")
    cfgs = list(means)
    for a, b in itertools.combinations(cfgs, 2):
        D = [d for d in means[a] if d in means[b]]
        if len(D) < 3: continue
        va, vb = [means[a][d] for d in D], [means[b][d] for d in D]
        best = (min(D, key=lambda d: means[a][d]), min(D, key=lambda d: means[b][d]))
        worst = (max(D, key=lambda d: means[a][d]), max(D, key=lambda d: means[b][d]))
        print(f"\n[rank] {a} vs {b} over {len(D)} designs: Spearman rho = {spearman(va, vb):+.2f}, "
              f"Kendall tau = {kendall(va, vb):+.2f}; best {best[0]} / {best[1]}, worst {worst[0]} / {worst[1]}; "
              f"mean iF ratio {np.mean(vb)/np.mean(va):.2f}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
