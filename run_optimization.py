"""run_optimization.py — Multi-resolution Bayesian optimization of city layout
to minimize the exposure objective.

The objective is the effective-inhabitance-weighted concentration:
    J = Σ_locations  C(location) · inh_eff(location)
the inner product of the dispersed concentration field (mass-weighted TIAC)
and the effective inhabitance map (occupancy-time-weighted population). No exposure
weights or filtration are applied in this raw objective — just the concentration·inhabitance
product. (Source is a single point source for now; the source treatment is a
separate later change.)

Pipeline per layout candidate:
    params → solver_eval (warm-up flow → release source) → C field
           → Σ C·inh_eff → scalar J → Bayesian optimizer.

Features:
  • Multi-resolution: a coarse-cell stage explores; the best `seed_topk` points
    seed a fine-cell stage that refines. Unit-cube points are resolution-
    independent, so seeds transfer directly. (Build one solver_eval binary per
    cell size; see param_space.STAGES.)
  • Warm restarts: each evaluation saves its flow field; the next reuses the
    nearest prior field when the grid matches (handled in evaluate.Evaluator;
    the driver cold-starts automatically on a grid mismatch).
  • Early termination (per stage, see bayesopt.optimize): budget exhausted,
    no relative improvement > rel_tol for `patience` steps, or EI collapse.

Run:   python3 run_optimization.py
(Builds resolution binaries first — see build_optimizer.sh.)
"""
import os, json
import numpy as np
import param_space as PS
import bayesopt


def run_multiresolution(stages, dim, make_evaluator, early_stop, verbose=True):
    """Core orchestration. make_evaluator(stage) -> callable(x_unit)->J.
    Returns list of (stage_name, optimize-result)."""
    seeds = []           # unit-cube points carried from the previous stage
    results = []
    for st in stages:
        if verbose:
            print(f"\n══════ STAGE '{st['name']}'  cell={st['cell']}m  "
                  f"(n_init={st['n_init']}, n_iter={st['n_iter']}, seeds={len(seeds)}) ══════")
        ev = make_evaluator(st)
        res = bayesopt.optimize(ev, dim,
                                n_init=st['n_init'], n_iter=st['n_iter'],
                                init_points=seeds, seed=st.get('seed', 0),
                                verbose=verbose, **early_stop)
        order = np.argsort(res['Y'])
        seeds = [res['X'][i] for i in order[:st.get('seed_topk', 0)]]
        results.append((st['name'], res))
        if verbose:
            print(f"  stage best J={res['y_best']:.4e}  "
                  f"({res['n_eval']} evals, stop={res['stop_reason']})")
    return results


def summarize(results):
    name, res = results[-1]
    best_named = PS.to_physical(res['x_best'])
    out = dict(stage=name, J_best=res['y_best'], n_eval_total=sum(r['n_eval'] for _,r in results),
               best_params=best_named)
    print("\n══════ OPTIMIZATION COMPLETE ══════")
    print(f"  best J = {res['y_best']:.4e}  ({out['n_eval_total']} total evaluations)")
    print("  best layout parameters:")
    for k, v in best_named.items():
        print(f"    {k:18s} = {v:.5g}")
    return out


def make_subprocess_evaluator(stage):
    return __import__('evaluate').Evaluator(
        binary=stage['binary'], cell=stage['cell'],
        workroot=os.path.join('opt_work', stage['name']),
        fixed=PS.FIXED,
        omp_threads=int(os.environ.get('OMP_NUM_THREADS', 8)),
        warm_restart=True, verbose=True)


if __name__ == '__main__':
    results = run_multiresolution(PS.STAGES, PS.DIM, make_subprocess_evaluator,
                                  PS.EARLY_STOP, verbose=True)
    out = summarize(results)
    with open('optimization_result.json', 'w') as f:
        json.dump(out, f, indent=2)
    print("  written optimization_result.json")
