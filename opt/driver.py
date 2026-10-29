#!/usr/bin/env python3
"""driver.py — Bayesian optimisation of the city design against the rose-averaged intake fraction.

    python3 -m opt.driver run    --workdir W --gen GEN --solver BIN [--budget 150 --n-init 24 ...]
    python3 -m opt.driver report --workdir W
    python3 -m opt.driver verify --workdir W --gen GEN --solver BIN [--top 3]

Objective (minimised): F(design) = mean over a uniform 16-direction wind rose of the intake fraction
iF(design, direction) (opt/evaluate.py). One CFD run = one (design, direction, inlet seed).

Method (OPENLB_PHASE9_FIDELITY.md section 8, OPTIMIZER.md):
  surrogate  GP over (design, direction): k = k_d + k_d' * k_theta, Matern-5/2 on the continuous
             knobs x a categorical kernel on emp_centers, periodic in the direction (one turn).
             The rose mean is the GP averaged over the 16 directions (BoTorch AppendFeatures +
             Expectation), the construction of Cakmak et al. (2020).
  design     log expected improvement (analytic, BoTorch LogExpectedImprovement) on the rose-mean
             posterior, incumbent = best posterior mean (plug-in EI for noisy runs); candidates
             are the acquisition optimum per emp_centers level AND every design already run, so
             a contender can win a further direction instead of a new design being forced.
  direction  for the chosen design, the direction whose run would most reduce the posterior
             variance of its rose mean: argmax_j Cov(F, J_j)^2 / (Var J_j + noise), exact for a GP.
  noise      a fresh inlet seed per run; the GP's noise term absorbs the seed scatter.
  feasibility designs the generator cannot fit under the height limit are never run (pre-check).

History (WORKDIR/history.jsonl) and the evaluator's per-run cache make every command resumable.
"""
import argparse, json, math, os, sys, time, warnings
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import param_space_density as P                  # noqa: E402

N_DIR = 16
DIRS_DEG = [360.0 * i / N_DIR for i in range(N_DIR)]
warnings.filterwarnings("ignore")


# ── design space: unit cube (generator) <-> model coordinates (categorical levels as integers) ──
class Space:
    def __init__(self):
        self.names = P.NAMES_DENSITY
        self.d = len(self.names)
        self.cat = [i for i, (n, lo, hi, s) in enumerate(P.PARAM_SPACE_DENSITY) if s == "int"]
        self.levels = {i: int(P.PARAM_SPACE_DENSITY[i][2] - P.PARAM_SPACE_DENSITY[i][1] + 1) for i in self.cat}

    def to_model(self, x):
        z = np.array(x, float).copy()
        for i in self.cat:
            z[i] = float(min(self.levels[i] - 1, int(np.clip(z[i], 0, 1) * self.levels[i])))
        return z

    def to_unit(self, z):
        x = np.array(z, float).copy()
        for i in self.cat:
            x[i] = (round(z[i]) + 0.5) / self.levels[i]
        return np.clip(x, 0.0, 1.0)

    def bounds(self):
        import torch
        lo = np.zeros(self.d); hi = np.ones(self.d)
        for i in self.cat: hi[i] = self.levels[i] - 1
        return torch.tensor(np.stack([lo, hi]), dtype=torch.double)


# ── history ──
class History:
    def __init__(self, workdir):
        self.fn = os.path.join(workdir, "history.jsonl"); os.makedirs(workdir, exist_ok=True)
        self.rows = [json.loads(l) for l in open(self.fn)] if os.path.exists(self.fn) else []

    def add(self, row):
        self.rows.append(row)
        with open(self.fn, "a") as f: f.write(json.dumps(row) + "\n")

    def ok(self):
        return [r for r in self.rows if r.get("feasible", True) and r.get("iF_ppm") is not None and r.get("phase") != "verify"]


def design_id(x):
    return tuple(round(float(v), 6) for v in x)


# ── model ──
def fit_model(space, rows):
    import torch
    from botorch.models import SingleTaskGP
    from botorch.models.transforms import Standardize
    from botorch.models.transforms.input import AppendFeatures
    from botorch.models.kernels.categorical import CategoricalKernel
    from botorch.fit import fit_gpytorch_mll
    from gpytorch.mlls import ExactMarginalLogLikelihood
    from gpytorch.kernels import MaternKernel, PeriodicKernel, ScaleKernel
    d = space.d
    Z = np.array([space.to_model(r["x"]) for r in rows])
    th = np.array([[r["phi"] / 360.0] for r in rows])
    X = torch.tensor(np.hstack([Z, th]), dtype=torch.double)
    Y = torch.tensor([[-r["iF_ppm"]] for r in rows], dtype=torch.double)       # BoTorch maximises
    cont = [i for i in range(d) if i not in space.cat]

    def kd():
        k = MaternKernel(nu=2.5, ard_num_dims=len(cont), active_dims=cont)
        if space.cat:
            k = k * CategoricalKernel(ard_num_dims=len(space.cat), active_dims=space.cat)
        return k
    kth = PeriodicKernel(active_dims=[d]); kth.period_length = 1.0
    kth.raw_period_length.requires_grad_(False)
    covar = ScaleKernel(kd()) + ScaleKernel(kd() * kth)
    dirs = torch.tensor([[a / 360.0] for a in DIRS_DEG], dtype=torch.double)
    m = SingleTaskGP(X, Y, covar_module=covar, outcome_transform=Standardize(m=1),
                     input_transform=AppendFeatures(feature_set=dirs))
    fit_gpytorch_mll(ExactMarginalLogLikelihood(m.likelihood, m))
    return m


def rose_posterior(model, Z):
    """Posterior mean and sd of F (intake fraction, ppm) at model-coordinate designs Z (n x d)."""
    import torch
    from botorch.acquisition.objective import ExpectationPosteriorTransform
    with torch.no_grad():
        p = model.posterior(torch.as_tensor(np.atleast_2d(Z), dtype=torch.double),
                            posterior_transform=ExpectationPosteriorTransform(n_w=N_DIR))
    return (-p.mean.squeeze(-1)).numpy(), p.variance.clamp_min(0).sqrt().squeeze(-1).numpy()


def choose_direction(model, z):
    """argmax_j Cov(F, J_j)^2 / (Var J_j + noise): the run that most reduces Var F at design z."""
    import torch
    with torch.no_grad():
        post = model.posterior(torch.as_tensor(z[None, :], dtype=torch.double))
        C = post.distribution.covariance_matrix.squeeze()          # 16 x 16, original scale
        noise = float(model.likelihood.noise.mean()) * float(model.outcome_transform.stdvs.squeeze()) ** 2
        cf = C.mean(dim=0)                                          # Cov(F, J_j) with F = mean_i J_i
        score = cf ** 2 / (torch.diagonal(C) + noise)
    return DIRS_DEG[int(torch.argmax(score))], score.numpy()


def propose(space, model, rows, feasible, restarts=4, raw=128):
    """Next design (model coordinates): analytic log expected improvement on the rose-mean
    posterior, with the best posterior mean among the designs run as the incumbent (plug-in EI,
    the usual choice for noisy evaluations). Infeasible candidates are skipped."""
    import torch
    from botorch.acquisition.analytic import LogExpectedImprovement
    from botorch.acquisition.objective import ExpectationPosteriorTransform
    from botorch.optim import optimize_acqf
    designs = {}
    for r in rows: designs.setdefault(design_id(r["x"]), space.to_model(r["x"]))
    Zd = np.array(list(designs.values()))
    mu, _ = rose_posterior(model, Zd)
    acq = LogExpectedImprovement(model, best_f=float(-mu.min()),
                                 posterior_transform=ExpectationPosteriorTransform(n_w=N_DIR))
    bounds = space.bounds(); cands = []
    levels = [dict()] if not space.cat else [{space.cat[0]: float(v)} for v in range(space.levels[space.cat[0]])]
    for ff in levels:
        try:
            X, v = optimize_acqf(acq, bounds, q=1, num_restarts=restarts, raw_samples=raw,
                                 fixed_features=ff or None, return_best_only=False)
            for k in range(X.shape[0]): cands.append((float(v[k]), X[k, 0].detach().numpy(), "new"))
        except Exception as e:                                       # keep going on a numerical hiccup
            print(f"[opt] acquisition optimisation failed for {ff}: {e}")
    with torch.no_grad():
        vb = acq(torch.tensor(Zd[:, None, :], dtype=torch.double))
    for k in range(len(Zd)): cands.append((float(vb[k]), Zd[k], "rerun"))
    cands.sort(key=lambda c: -c[0])
    for val, z, kind in cands:
        z = space.to_model(space.to_unit(z)) if kind == "new" else z
        if kind == "rerun" or feasible(space.to_unit(z)):
            return z, kind, val
    raise RuntimeError("no feasible candidate")


# ── commands ──
def run(args, evaluator, space=None, log=print):
    space = space or Space(); H = History(args.workdir)
    from scipy.stats import qmc
    sob = qmc.Sobol(space.d, scramble=True, seed=args.seed).random(max(args.n_init, 1))
    rng = np.random.default_rng(args.seed)
    init_dirs = list(rng.permutation(N_DIR)) * (args.n_init // N_DIR + 1)
    seed_next = args.seed_base + len(H.rows)
    while len([r for r in H.rows if r.get("phase") != "verify"]) < args.budget:
        n = len([r for r in H.rows if r.get("phase") != "verify"])
        t0 = time.time()
        if n < args.n_init:
            x = sob[n]; phi = DIRS_DEG[init_dirs[n]]; kind = "init"; acqv = float("nan")
            if not evaluator.feasible(x):
                H.add({"x": list(map(float, x)), "phi": phi, "feasible": False, "phase": "init"}); continue
        else:
            model = fit_model(space, H.ok())
            z, kind, acqv = propose(space, model, H.ok(), evaluator.feasible)
            x = space.to_unit(z); phi, _ = choose_direction(model, z)
        res = evaluator(x, phi, seed_next); seed_next += 1
        res.update({"phase": kind, "acq": acqv, "t_decide_s": time.time() - t0 - res.get("wall_s", 0)})
        H.add(res)
        ok = H.ok()
        nd = len({design_id(r["x"]) for r in ok})
        log(f"[opt] {len(ok):4d} runs, {nd:3d} designs | {kind:5s} phi {phi:6.1f} -> iF {res.get('iF_ppm', float('nan')):.4f} ppm"
            f"{'' if res.get('feasible', True) else ' (infeasible)'}")
    return H


def report(args, space=None, log=print, model=None):
    space = space or Space(); H = History(args.workdir); ok = H.ok()
    if len(ok) < 3: log("[opt] fewer than 3 runs"); return []
    model = model or fit_model(space, ok)
    designs = {}
    for r in ok: designs.setdefault(design_id(r["x"]), []).append(r)
    ids = list(designs); Z = np.array([space.to_model(designs[i][0]["x"]) for i in ids])
    mu, sd = rose_posterior(model, Z)
    order = np.argsort(mu)
    log(f"[opt] {len(ok)} runs on {len(ids)} designs; directions per design: mean {np.mean([len(designs[i]) for i in ids]):.2f}, "
        f"max {max(len(designs[i]) for i in ids)}")
    log(f"[opt] {'rank':>4} {'F mean':>9} {'F sd':>7} {'dirs':>4}  design")
    out = []
    for k, j in enumerate(order[: args.show]):
        ph = P.to_physical_density(designs[ids[j]][0]["x"])
        desc = ", ".join(f"{n}={ph[n]:.3g}" for n in space.names)
        log(f"[opt] {k+1:>4} {mu[j]:9.4f} {sd[j]:7.4f} {len(designs[ids[j]]):>4}  {desc}")
        out.append({"x": designs[ids[j]][0]["x"], "F_mean": float(mu[j]), "F_sd": float(sd[j]), "n_dir": len(designs[ids[j]])})
    json.dump(out, open(os.path.join(args.workdir, "report.json"), "w"), indent=1)
    return out


def verify(args, evaluator, space=None, log=print):
    """Top designs by posterior mean, run at all 16 directions with new seeds: the reported F."""
    space = space or Space(); H = History(args.workdir)
    top = report(argparse.Namespace(workdir=args.workdir, show=args.top), space, log=lambda *a: None)[: args.top]
    seed = args.verify_seed
    res = []
    for t in top:
        vals = []
        for phi in DIRS_DEG:
            r = evaluator(t["x"], phi, seed); seed += 1
            r["phase"] = "verify"; H.add(r); vals.append(r["iF_ppm"])
        Fv = float(np.mean(vals))
        log(f"[verify] design F = {Fv:.4f} ppm over 16 directions (surrogate {t['F_mean']:.4f} +- {t['F_sd']:.4f}); "
            f"directions min {min(vals):.4f} max {max(vals):.4f}")
        res.append({"x": t["x"], "F_verified": Fv, "per_direction": vals, "F_surrogate": t["F_mean"]})
    json.dump(res, open(os.path.join(args.workdir, "verified.json"), "w"), indent=1)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run", "report", "verify"])
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--gen"); ap.add_argument("--solver")
    ap.add_argument("--budget", type=int, default=150); ap.add_argument("--n-init", type=int, default=24)
    ap.add_argument("--seed", type=int, default=1); ap.add_argument("--seed-base", type=int, default=10000)
    ap.add_argument("--top", type=int, default=3); ap.add_argument("--verify-seed", type=int, default=90000)
    ap.add_argument("--show", type=int, default=10)
    ap.add_argument("--run-env", default="", help="extra solver settings, e.g. 'TAU=0.5001 D_LOCAL=1'")
    ap.add_argument("--gen-env", default="", help="extra generator settings, e.g. 'H_MAX=100'")
    a = ap.parse_args(argv)
    if a.cmd in ("run", "verify"):
        from opt.evaluate import CFDEvaluator
        if not (a.gen and a.solver): ap.error("--gen and --solver are required")
        kv = lambda s: dict(t.split("=", 1) for t in s.split()) if s else {}
        ev = CFDEvaluator(a.gen, a.solver, a.workdir, run_env=kv(a.run_env), gen_env=kv(a.gen_env))
        (run if a.cmd == "run" else verify)(a, ev)
        if a.cmd == "run": report(a)
    else:
        report(a)


if __name__ == "__main__":
    main()
