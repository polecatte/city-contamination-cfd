"""Logic tests for the optimizer orchestration (no C++ solver involved)."""
import numpy as np
import param_space as PS
from run_optimization import run_multiresolution, summarize

# ── (1) Multi-resolution: synthetic deterministic objective ──
# shifted sphere in PS.DIM dims, min at unit-cube point 0.5
DIM = PS.DIM
def make_synth(stage):
    # "coarse" stage adds a small bias to mimic lower fidelity; fine is exact.
    bias = 0.0 if stage['cell'] <= 4.0 else 0.02
    def fn(x):
        return float(np.sum((np.asarray(x)-0.5)**2)) + bias
    return fn

stages = [dict(name='coarse', cell=8.0, n_init=18, n_iter=40, seed_topk=5, seed=1),
          dict(name='fine',   cell=4.0, n_init=5,  n_iter=30, seed_topk=0, seed=2)]
res = run_multiresolution(stages, DIM, make_synth, PS.EARLY_STOP, verbose=False)
coarse_best = res[0][1]['y_best']; fine_best = res[1][1]['y_best']
print(f"(1) multi-res: coarse_best={coarse_best:.4e}  fine_best={fine_best:.4e}")
assert fine_best <= coarse_best + 1e-9, "fine stage should not be worse than coarse (seeding)"
out = summarize(res)
assert set(out['best_params'].keys()) == {n for n,_,_,_ in PS.PARAM_SPACE}, "param mapping mismatch"
print("    multi-resolution + seeding + summary OK")

# ── (2) Warm-restart nearest-neighbour selection ──
import os
from evaluate import Evaluator
ev = Evaluator(binary='/bin/true', cell=8.0, workroot='/tmp/wrtest', warm_restart=True, verbose=False)
ev.prior = [dict(x_unit=np.full(DIM,0.1), ckpt='A'),
            dict(x_unit=np.full(DIM,0.9), ckpt='B')]
near_lo = ev._nearest_ckpt(np.full(DIM,0.15))
near_hi = ev._nearest_ckpt(np.full(DIM,0.85))
print(f"(2) warm-restart NN: near(0.15)->{near_lo}  near(0.85)->{near_hi}")
assert near_lo=='A' and near_hi=='B', "nearest-neighbour ckpt selection wrong"
# empty prior → None (cold start)
ev2 = Evaluator(binary='/bin/true', cell=8.0, workroot='/tmp/wrtest2', warm_restart=True, verbose=False)
assert ev2._nearest_ckpt(np.zeros(DIM)) is None
print("    warm-restart NN + cold-start fallback OK")

# ── (3) to_physical mapping incl. log-scale ──
xlo = PS.to_physical(np.zeros(DIM)); xhi = PS.to_physical(np.ones(DIM))
d = dict((n,(lo,hi,sc)) for n,lo,hi,sc in PS.PARAM_SPACE)
assert abs(xlo['cbd_decay']-d['cbd_decay'][0])<1e-12          # log low edge
assert abs(xhi['cbd_decay']-d['cbd_decay'][1])<1e-12          # log high edge
assert abs(xlo['block_w']-d['block_w'][0])<1e-9              # lin low edge
print(f"(3) mapping: cbd_decay {xlo['cbd_decay']:.1e}..{xhi['cbd_decay']:.1e}, "
      f"block_w {xlo['block_w']:.0f}..{xhi['block_w']:.0f}  OK")

print("\nALL ORCHESTRATION LOGIC TESTS PASSED")
