#!/usr/bin/env python3
"""test_driver_synth.py — the optimisation loop end to end on a known function (no CFD).

    python3 tests/test_driver_synth.py [WORKDIR]      (needs requirements-opt.txt)

opt.evaluate.SyntheticEvaluator stands in for the CFD: a rose-mean exposure F(design) with a known
optimum and a categorical effect, a design-dependent cos(4 theta) street-alignment term as large
as the design spread, a direction-asymmetric term, and 0.5 % run noise. Checks:
  1. the loop runs over the real 11-D parameter space (unit cube, emp_centers categorical),
     and resumes: a second invocation continues the history without re-running anything;
  2. the design it reports as best has a TRUE rose mean F better than 95 % of random designs;
  3. it spends directions around contenders: the 10 runs nearest the reported best design are
     close to it in design space and cover at least 5 different wind directions;
  4. verify: the top design run at all 16 directions agrees with the surrogate's F within
     its stated uncertainty (3 sigma) or 2 %.
"""
import os, sys, shutil, argparse
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from opt import driver                                   # noqa: E402
from opt.evaluate import SyntheticEvaluator              # noqa: E402

WD = sys.argv[1] if len(sys.argv) > 1 else "/tmp/opt_synth"
shutil.rmtree(WD, ignore_errors=True)
ev = SyntheticEvaluator(WD, noise=0.005)
quiet = lambda *a: None
args = argparse.Namespace(workdir=WD, budget=30, n_init=20, seed=1, seed_base=1000, show=5, top=1, verify_seed=90000)

driver.run(args, ev, log=quiet)                          # first 30 runs
n1 = ev.n_calls
args.budget = 60
driver.run(args, ev, log=print)                          # resume to 60
H = driver.History(WD)
ok = H.ok()
resumed = len(ok) == 60 and ev.n_calls == 60 and n1 == 30
print(f"[synth] resume: {len(ok)} runs in history, {ev.n_calls} evaluations in total -> {'OK' if resumed else 'FAIL'}")

top = driver.report(args, log=print)
best_true = ev.F(top[0]["x"])
rng = np.random.default_rng(5)
rand = np.array([ev.F(x) for x in rng.random((4000, driver.Space().d))])
pct = 100.0 * np.mean(rand < best_true)
good = pct <= 5.0
print(f"[synth] best design: true F {best_true:.3f} (optimum region ~20.0), better than {100-pct:.1f} % of random designs "
      f"-> {'OK' if good else 'FAIL'}")

# Directions are spent around contenders: in a continuous space the optimizer samples a contender
# through near neighbours at other directions (exact repeats are rarely worth more), so check the
# runs nearest the reported best design: close in design space and many directions.
X = np.array([r["x"] for r in ok]); dist = np.linalg.norm(X - np.array(top[0]["x"]), axis=1)
near = np.argsort(dist)[:10]
typical = np.median(np.linalg.norm(rng.random((500, X.shape[1])) - rng.random((500, X.shape[1])), axis=1))
ndir = len({ok[i]["phi"] for i in near})
uneven = ndir >= 5 and dist[near].max() < 0.6 * typical
reruns = len(ok) - len({driver.design_id(r["x"]) for r in ok})
print(f"[synth] 10 runs nearest the best design: within {dist[near].max():.2f} (random pairs ~{typical:.2f}), "
      f"{ndir} distinct directions; exact repeats of a design: {reruns} -> {'OK' if uneven else 'FAIL'}")

vres = driver.verify(args, ev, log=print)[0]
agree = abs(vres["F_verified"] - top[0]["F_mean"]) <= max(3 * top[0]["F_sd"], 0.02 * vres["F_verified"])
print(f"[synth] verify: 16-direction F {vres['F_verified']:.3f} vs surrogate {top[0]['F_mean']:.3f} +- {top[0]['F_sd']:.3f} "
      f"(true {best_true:.3f}) -> {'OK' if agree else 'FAIL'}")
ok_all = resumed and good and uneven and agree
print(f"[synth] {'PASS' if ok_all else 'FAIL'}")
sys.exit(0 if ok_all else 1)
