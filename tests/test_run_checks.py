#!/usr/bin/env python3
"""test_run_checks.py — evaluate.run_checks and the driver's handling of rejected runs.

    python3 tests/test_run_checks.py              (numpy for 1-2; 3 also needs scipy)

1. run_checks on meta_flow.txt contents: a clean clearance stop, a tail stop and a capped run
   with a small tail fit pass; a capped run without a tail fit, a tail fit over 15 % of J, an
   open mass budget, a non-finite J, an unknown stop reason and an old-format (no stop_reason)
   run that did not clear are rejected, each with its reason.
2. intake_fraction reads stop_reason (a text value) from meta_flow.txt.
3. the driver: a rejected run stays in the history but out of the GP (History.ok), and three
   rejected runs in a row stop the search with a RuntimeError instead of spending the budget.
   The runs fail inside the Sobol start (n_init), before any GP is fitted, so torch is not needed.
"""
import os, sys, tempfile, argparse
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from opt.evaluate import run_checks, intake_fraction      # noqa: E402

fails = 0
def check(name, ok, detail=""):
    global fails
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(': ' + detail) if detail else ''}")
    fails += not ok

GOOD = {"budget_closure": 5e-15, "stop_reason": "clearance", "tail_applied": 0.0, "tail_fit_frac": 0.0,
        "tail_air_frac_at_stop": 0.009, "burst_cap": 61000.0}

print("1. run_checks")
cases = [
    ("clearance stop",              8.7, {},                                                         None),
    ("tail stop, tail 4 % of J",    8.7, {"stop_reason": "tail", "tail_applied": 1.0, "tail_fit_frac": 0.04}, None),
    ("cap, small tail fit",         8.7, {"stop_reason": "cap", "tail_applied": 1.0, "tail_fit_frac": 0.08}, None),
    ("cap, no tail fit",            8.7, {"stop_reason": "cap", "tail_air_frac_at_stop": 0.12},      "MAX_BURST_STEPS"),
    ("cap, tail 30 % of J",         8.7, {"stop_reason": "cap", "tail_applied": 1.0, "tail_fit_frac": 0.30}, "extrapolated tail"),
    ("budget open by 3 %",          8.7, {"budget_closure": 0.03},                                   "mass budget"),
    ("no budget_closure",           8.7, {"budget_closure": None},                                   "no budget_closure"),
    ("J not finite",       float("nan"), {},                                                         "not finite"),
    ("J zero",                      0.0, {},                                                         "not finite"),
    ("unknown stop reason",         8.7, {"stop_reason": "weird"},                                   "unknown stop_reason"),
    ("old format, cleared",         8.7, {"stop_reason": None, "tail_air_frac_at_stop": 0.009},      None),
    ("old format, not cleared",     8.7, {"stop_reason": None, "tail_air_frac_at_stop": 0.25},       "did not clear"),
]
for name, J, upd, expect in cases:
    m = {**GOOD, **upd}; m = {k: v for k, v in m.items() if v is not None}
    why = run_checks(J, m)
    if expect is None: check(name, why == [], "; ".join(why))
    else: check(name, any(expect in w for w in why), "; ".join(why) or "accepted")

print("2. intake_fraction reads text values")
with tempfile.TemporaryDirectory() as d:
    hdr = np.array([3, 2, 1, 4000, 1], np.int32)
    for fn, a in (("receptor_w.f32", [1, 0, 2, 0, 0, 1]), ("theta.f32", [2, 5, 1, 0, 0, 3])):
        with open(os.path.join(d, fn), "wb") as f: hdr.tofile(f); np.array(a, np.float32).tofile(f)
    open(os.path.join(d, "meta_flow.txt"), "w").write(
        "dx_m 4.0\nburst_steps 61000\nburst_cap 61000\nstop_reason cap\nmass_emitted 2\nmass_emitted_total 4\n"
        "budget_closure 1e-14\ntail_applied 0\n")
    J, iF, m = intake_fraction(d, d)
    check("J = <w,Theta>/emitted_total", abs(J - (1*2 + 2*1 + 1*3) / 4) < 1e-12, f"{J}")
    check("stop_reason kept as text", m.get("stop_reason") == "cap", f"{m.get('stop_reason')!r}")
    check("capped run rejected", any("MAX_BURST_STEPS" in w for w in run_checks(J, m)))

print("3. driver: rejected runs")
try:
    import scipy  # noqa: F401  (the Sobol start; the GP, and torch, come after n_init)
    from opt import driver
    have_driver = True
except ImportError:
    have_driver = False
    print("  (skipped: scipy not installed)")
if have_driver:
    from opt.evaluate import SyntheticEvaluator

    class Flaky(SyntheticEvaluator):
        """Synthetic evaluator whose runs from call `bad_from` on fail the run checks."""
        def __init__(self, wd, bad_from):
            super().__init__(wd, noise=0.005); self.bad_from = bad_from
        def __call__(self, x, phi, seed):
            r = super().__call__(x, phi, seed)
            if self.n_calls > self.bad_from:
                r = {**r, "valid": False, "reject": ["hit MAX_BURST_STEPS before clearing, no tail fit"],
                     "iF_raw_ppm": r["iF_ppm"], "iF_ppm": None}
            return r

    with tempfile.TemporaryDirectory() as wd:
        ev = Flaky(wd, bad_from=6)
        args = argparse.Namespace(workdir=wd, budget=20, n_init=12, seed=1, seed_base=1000)
        stopped = None
        try: driver.run(args, ev, log=lambda *a: None)
        except RuntimeError as e: stopped = str(e)
        H = driver.History(wd)
        check("three rejected runs in a row stop the search", stopped is not None and "in a row" in stopped,
              (stopped or "did not stop")[:90])
        check("rejected runs kept in the history", sum(r.get("valid") is False for r in H.rows) == 3,
              f"{sum(r.get('valid') is False for r in H.rows)} rejected rows")
        check("rejected runs never reach the GP", all(r.get("iF_ppm") is not None for r in H.ok()) and len(H.ok()) == 6,
              f"{len(H.ok())} rows in History.ok()")

print(f"\n{'ALL PASS' if fails == 0 else f'{fails} FAILED'}")
sys.exit(1 if fails else 0)
