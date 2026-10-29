#!/usr/bin/env python3
"""precision_report.py — is the float-lattice GPU build as good as the double one for exposure?

    python3 tests/precision_report.py MANIFEST

Manifest lines (lab_openlb.sh precision writes them):  precision  design  seed  geom_dir  out_dir
with precision 'double' or 'single'. Expected per design: double at two inlet seeds (s1, s2) and
single at s1.

Why this test and not "float == double": the flow is turbulent, so a float run and a double run
from the same seed part ways after a few flow-throughs and become two realisations of the same
statistics. They cannot agree better than two seeds of the double build do. So the double seed
pair measures the run-to-run noise sigma (relative, one run), and the float-minus-double
difference is judged against it:
  - each design:  |J_single(s1)/J_double(s1) - 1| < 3 sqrt(2) sigma   (a difference of two runs)
  - all designs:  |mean of those| < 2 sqrt(2) sigma / sqrt(n)          (no systematic bias)
  - every run passes opt/evaluate.py's run checks; the float budget closes to < 1e-3
sigma is pooled over the designs' seed pairs, floored at 0.3 % so that a lucky pair cannot make
the test impossible. Also printed: the ranking of the designs in each column (Spearman), and the
wall time of each build (spin-up + burst, from meta.txt / meta_flow.txt) with the speed-up.
Exit 0 = PASS, 1 = FAIL, 2 = incomplete input.
"""
import os, sys, math
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
from rank_report import run_J            # noqa: E402  (J = <w,Theta>/M, iF ppm, numeric meta)
from opt.evaluate import run_checks      # noqa: E402

SIGMA_FLOOR = 0.003


def meta_all(path):
    m = {}
    if os.path.exists(path):
        for line in open(path):
            p = line.split()
            if len(p) == 2 and not p[0].startswith('#'):
                try: m[p[0]] = float(p[1])
                except ValueError: m[p[0]] = p[1]
    return m


def spearman(a, b):
    def ranks(v):
        o = sorted(range(len(v)), key=lambda i: v[i]); r = [0] * len(v)
        for k, i in enumerate(o): r[i] = k
        return r
    if len(a) < 3: return float('nan')
    ra, rb = ranks(a), ranks(b); n = len(a)
    return 1 - 6 * sum((x - y) ** 2 for x, y in zip(ra, rb)) / (n * (n * n - 1))


def main():
    if len(sys.argv) != 2:
        print(__doc__); return 2
    runs = defaultdict(dict)                 # design -> {(precision, seed): record}
    order = []
    for line in open(sys.argv[1]):
        p = line.split()
        if len(p) != 5 or p[0].startswith('#'): continue
        prec, design, seed, geom, out = p[0], p[1], int(p[2]), p[3], p[4]
        if design not in order: order.append(design)
        if not os.path.exists(os.path.join(out, 'theta.f32')):
            print(f"  missing: {prec} {design} s{seed} ({out})"); continue
        J, iF, _ = run_J(geom, out)
        mf = meta_all(os.path.join(out, 'meta_flow.txt')); ma = meta_all(os.path.join(out, 'meta.txt'))
        why = run_checks(J, mf)
        st, sw = ma.get('spinup_steps'), ma.get('spinup_wall_s')
        bt, bw = mf.get('burst_steps'), mf.get('burst_wall_s')
        runs[design][(prec, seed)] = dict(J=J, iF=iF, closure=mf.get('budget_closure'), stop=mf.get('stop_reason'),
                                          built=mf.get('precision', '?'), why=why,
                                          wall=(sw + bw) if isinstance(sw, float) and isinstance(bw, float) else None,
                                          ms_spin=1e3 * sw / st if isinstance(sw, float) and st else None,
                                          ms_burst=1e3 * bw / bt if isinstance(bw, float) and bt else None)
    print(f"{'design':8s} {'run':12s} {'built':6s} {'iF ppm':>9s} {'closure':>9s} {'stop':>9s} "
          f"{'spin ms/st':>10s} {'burst ms/st':>11s} {'wall s':>8s}  checks")
    d_seed, d_prec, rows, fails, speed = [], [], [], [], []
    for design in order:
        r = runs[design]
        for k in sorted(r, key=lambda k: (k[0] != 'double', k[1])):
            x = r[k]; f = lambda v, fmt: (fmt % v) if isinstance(v, float) else '-'
            print(f"{design:8s} {k[0] + ' s' + str(k[1]):12s} {str(x['built']):6s} {x['iF']:9.3f} {f(x['closure'], '%.1e'):>9s} "
                  f"{str(x['stop']):>9s} {f(x['ms_spin'], '%.2f'):>10s} {f(x['ms_burst'], '%.2f'):>11s} {f(x['wall'], '%.0f'):>8s}  "
                  f"{'ok' if not x['why'] else '; '.join(x['why'])}")
            if x['why']: fails.append(f"{design} {k[0]} s{k[1]}: {'; '.join(x['why'])}")
            if k[0] == 'single' and x['built'] != 'float': fails.append(f"{design} single s{k[1]}: built as {x['built']}, not float")
            if k[0] == 'double' and x['built'] not in ('double', '?'): fails.append(f"{design} double s{k[1]}: built as {x['built']}")
            if k[0] == 'single' and isinstance(x['closure'], float) and x['closure'] >= 1e-3:
                fails.append(f"{design} single: budget closes only to {x['closure']:.1e} (>= 1e-3)")
        dbl = sorted(s for p, s in r if p == 'double'); sgl = sorted(s for p, s in r if p == 'single')
        if len(dbl) < 2 or not sgl or sgl[0] not in dbl:
            print(f"  {design}: needs double at two seeds and single at the first of them; skipped in the verdict"); continue
        s1, s2 = (sgl[0], [s for s in dbl if s != sgl[0]][0])
        a, b, c = r[('double', s1)], r[('double', s2)], r[('single', s1)]
        d_seed.append(b['J'] / a['J'] - 1); d_prec.append(c['J'] / a['J'] - 1)
        rows.append((design, a['J'], b['J'], c['J']))
        if a['wall'] and c['wall']: speed.append(a['wall'] / c['wall'])
    if not rows:
        print("no complete design (double s1, s2 + single s1)"); return 2
    n = len(rows)
    sig_meas = math.sqrt(sum(d * d for d in d_seed) / (2 * n))
    sig = max(sig_meas, SIGMA_FLOOR)
    print(f"\nseed noise (double, pooled over {n} design pair(s)): sigma = {100 * sig_meas:.2f} % per run"
          f"{f' (floored to {100 * sig:.2f} %)' if sig > sig_meas else ''}")
    lim_one, lim_mean = 3 * math.sqrt(2) * sig, 2 * math.sqrt(2) * sig / math.sqrt(n)
    print(f"{'design':8s} {'seed diff':>10s} {'float-double':>13s}   (limit per design {100 * lim_one:.2f} %)")
    for (design, *_), ds, dp in zip(rows, d_seed, d_prec):
        ok = abs(dp) < lim_one
        print(f"{design:8s} {100 * ds:+9.2f} % {100 * dp:+12.2f} %   {'ok' if ok else 'OUTSIDE'}")
        if not ok: fails.append(f"{design}: float-double {100 * dp:+.2f} % beyond {100 * lim_one:.2f} %")
    mean_dp = sum(d_prec) / n
    print(f"mean float-double {100 * mean_dp:+.2f} % (limit {100 * lim_mean:.2f} %)")
    if abs(mean_dp) >= lim_mean: fails.append(f"systematic float-double offset {100 * mean_dp:+.2f} %")
    if n >= 3:
        print(f"ranking: Spearman double s1 vs double s2 {spearman([r[1] for r in rows], [r[2] for r in rows]):+.2f}, "
              f"double s1 vs single s1 {spearman([r[1] for r in rows], [r[3] for r in rows]):+.2f}")
    if speed:
        g = math.exp(sum(math.log(v) for v in speed) / len(speed))
        print(f"speed-up float over double (wall time, same design and seed): {g:.2f}x (per design: "
              + ", ".join(f"{v:.2f}x" for v in speed) + ")")
    print("\nPRECISION " + ("PASS" if not fails else "FAIL"))
    for f in fails: print("  - " + f)
    return 0 if not fails else 1


if __name__ == '__main__':
    sys.exit(main())
