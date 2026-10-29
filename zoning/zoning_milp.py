#!/usr/bin/env python3
"""zoning_milp.py — the inner level of the two-tier design (TWO_TIER_DESIGN.md).

Given a fixed built form (gen_form_city: buildings.csv) and a dose table (dose_table.py), choose
each building's use (home or work) and occupancy to minimise the expected population dose.

    python3 zoning/zoning_milp.py GEOM_DIR DOSE.npz [--cvar-bound X] [--kappa 2.0] [--out DIR]

Per building b:  u_b in {0 work, 1 home};  residents R_b, jobs E_b >= 0
    R_b <= cap_res_b u_b,   R_b >= OCC_MIN cap_res_b u_b        (occupancy band when a home)
    E_b <= cap_job_b (1-u_b), E_b >= OCC_MIN cap_job_b (1-u_b)  (and when a workplace)
    sum R = P,  sum E = jobs                                     (from meta_geom.txt)
Dose per person in scenario s (NHAPS time budget, single-zone infiltration):
    resident: T_HOME F_INF env[s,b] + T_OUT street[s,b]          (outdoor time spent near home)
    worker:   T_WORK F_INF env[s,b]
Objective: expected population dose  sum_s prob_s L_s,  L_s = sum_b R_b a[s,b] + E_b c[s,b].
Event tail: CVaR_beta over scenarios of L_s <= bound, linear through Rockafellar & Uryasev (2000):
    t + 1/(1-beta) sum_s prob_s z_s <= bound,  z_s >= L_s - t,  z_s >= 0.
Per-person tail: no homes where the expected resident dose exceeds kappa x its mean over the
city's buildings (u_b fixed to 0 there).
Solved with HiGHS (scipy.optimize.milp).
"""
import os, sys, csv, json
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

T_HOME, T_WORK, T_OUT = 0.69, 0.18, 0.131      # city_builder7.h NHAPS: home, work, park + street
F_INF = 0.62                                   # infiltration factor (repo default; size-dependent later)
OCC_MIN = 0.5
BETA = 0.9


def load(geom, dose):
    rows = list(csv.DictReader(open(os.path.join(geom, "buildings.csv"))))
    meta = {}
    for l in open(os.path.join(geom, "meta_geom.txt")):
        p = l.split()
        if len(p) == 2 and not p[0].startswith("#"):
            try: meta[p[0]] = float(p[1])
            except ValueError: pass
    T = dict(np.load(dose))
    capR = np.array([float(r["cap_residents"]) for r in rows]); capE = np.array([float(r["cap_jobs"]) for r in rows])
    a = T_HOME * F_INF * T["env"] + T_OUT * T["street"]          # [S, B] per resident
    c = T_WORK * F_INF * T["env"]                                 # [S, B] per worker
    return rows, meta, capR, capE, a, c, T["prob"]


def solve(capR, capE, a, c, prob, P, J, cvar_bound=None, kappa=None, beta=BETA, fix_use=None, time_limit=60):
    S, B = a.shape
    abar, cbar = prob @ a, prob @ c
    # variable layout: u[B] | R[B] | E[B] | t | z[S]
    nv = 3 * B + 1 + S
    iu, iR, iE, it, iz = 0, B, 2 * B, 3 * B, 3 * B + 1
    cost = np.zeros(nv); cost[iR:iR + B] = abar; cost[iE:iE + B] = cbar
    lb = np.zeros(nv); ub = np.full(nv, np.inf); ub[iu:iu + B] = 1
    lb[it] = -np.inf
    integ = np.zeros(nv); integ[iu:iu + B] = 1
    if kappa is not None:
        ub[iu:iu + B] = np.where(abar > kappa * abar.mean(), 0, 1)
    if fix_use is not None:
        lb[iu:iu + B] = fix_use; ub[iu:iu + B] = fix_use
    A, lo, hi = [], [], []
    def row(entries, l, h):
        r = np.zeros(nv)
        for i, v in entries: r[i] += v
        A.append(r); lo.append(l); hi.append(h)
    for b in range(B):
        row([(iR + b, 1), (iu + b, -capR[b])], -np.inf, 0)
        row([(iR + b, 1), (iu + b, -OCC_MIN * capR[b])], 0, np.inf)
        row([(iE + b, 1), (iu + b, capE[b])], -np.inf, capE[b])
        row([(iE + b, 1), (iu + b, OCC_MIN * capE[b])], OCC_MIN * capE[b], np.inf)
    row([(iR + b, 1) for b in range(B)], P, P)
    row([(iE + b, 1) for b in range(B)], J, J)
    if cvar_bound is not None:
        for s in range(S):     # z_s - L_s + t >= 0
            row([(iz + s, 1), (it, 1)] + [(iR + b, -a[s, b]) for b in range(B)] + [(iE + b, -c[s, b]) for b in range(B)], 0, np.inf)
        row([(it, 1)] + [(iz + s, prob[s] / (1 - beta)) for s in range(S)], -np.inf, cvar_bound)
    else:
        ub[iz:iz + S] = 0; lb[it] = 0; ub[it] = 0
    res = milp(cost, integrality=integ, bounds=Bounds(lb, ub),
               constraints=LinearConstraint(np.array(A), lo, hi), options={"time_limit": time_limit})
    if res.x is None:
        return None
    x = res.x
    return dict(u=np.round(x[iu:iu + B]).astype(int), R=x[iR:iR + B], E=x[iE:iE + B], status=res.message)


def evaluate(R, E, a, c, prob, beta=BETA):
    L = a @ R + c @ E                                            # population dose per scenario
    mean = float(prob @ L)
    order = np.argsort(-L); cum = np.cumsum(prob[order]); tail = order[cum <= (1 - beta) + 1e-12]
    if len(tail) == 0: tail = order[:1]
    cvar = float((prob[tail] @ L[tail]) / prob[tail].sum())
    return mean, cvar


def random_zoning(capR, capE, P, J, rng):
    """Uses at random in proportion to need, occupancy uniform: the exposure-blind reference."""
    B = len(capR); need = P / (P + J)
    for _ in range(1000):
        u = (rng.random(B) < need).astype(int)
        cr, ce = (capR * u).sum(), (capE * (1 - u)).sum()
        if OCC_MIN * cr <= P <= cr and OCC_MIN * ce <= J <= ce:
            return u, capR * u * P / cr, capE * (1 - u) * J / ce
    return None


def main():
    geom, dose = sys.argv[1:3]
    arg = lambda k, d: type(d)(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d
    out = arg("--out", os.path.join(geom, "zoning"))
    os.makedirs(out, exist_ok=True)
    rows, meta, capR, capE, a, c, prob = load(geom, dose)
    P, J = meta["population"], meta["jobs"]
    rng = np.random.default_rng(0)
    refs = [random_zoning(capR, capE, P, J, rng) for _ in range(200)]
    refs = [r for r in refs if r is not None]
    ref = np.array([evaluate(R, E, a, c, prob) for _, R, E in refs])
    free = solve(capR, capE, a, c, prob, P, J)
    m0, cv0 = evaluate(free["R"], free["E"], a, c, prob)
    print(f"buildings {len(rows)}, scenarios {a.shape[0]}, population {P:.0f}, jobs {J:.0f}")
    print(f"random zonings (200): mean dose {ref[:,0].mean():.4e} (range {ref[:,0].min():.3e}..{ref[:,0].max():.3e}), "
          f"CVaR90 {ref[:,1].mean():.4e}")
    print(f"optimal zoning:       mean dose {m0:.4e} ({100*(1-m0/ref[:,0].mean()):.1f} % below the random mean), CVaR90 {cv0:.4e}")
    # Pareto front: tighten the event-tail bound step by step
    front = [(m0, cv0, free)]
    for f in np.linspace(0.98, 0.80, 10):
        sol = solve(capR, capE, a, c, prob, P, J, cvar_bound=f * cv0)
        if sol is None: break
        front.append((*evaluate(sol["R"], sol["E"], a, c, prob), sol))
    print("mean vs CVaR90 front: " + ", ".join(f"({m/m0:.3f}, {cv/cv0:.3f})" for m, cv, _ in front))
    kap = solve(capR, capE, a, c, prob, P, J, kappa=arg("--kappa", 1.5))
    if kap is not None:
        mk, ck = evaluate(kap["R"], kap["E"], a, c, prob)
        print(f"with per-person cap (no homes above {arg('--kappa', 1.5)} x mean resident dose): mean {mk/m0:.3f} x optimum")
    np.savez(os.path.join(out, "zoning.npz"), u=free["u"], R=free["R"], E=free["E"], ref=ref,
             front=np.array([(m, cv) for m, cv, _ in front]))
    json.dump(dict(mean=m0, cvar90=cv0, random_mean=float(ref[:, 0].mean()), random_cvar90=float(ref[:, 1].mean()),
                   homes=int(free["u"].sum()), buildings=len(rows)), open(os.path.join(out, "zoning_summary.json"), "w"), indent=1)
    return rows, capR, capE, a, c, prob, P, J, free, ref, front


if __name__ == "__main__":
    main()
