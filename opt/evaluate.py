"""evaluate.py — one CFD evaluation: (design, wind direction, inlet seed) -> intake fraction.

    design    x in [0,1]^d over param_space_density (the generator's knobs)
    direction phi, degrees, the direction the wind blows TOWARD, counter-clockwise from the
              city's own +x axis; mapped to the generator's (ORIENT, WIND_DEG) by
              tools/wind_rose.py (an exact grid symmetry of the city + an inflow angle in [0, 45])
    seed      inlet turbulence seed (ABL_SEED): the run-to-run noise

Pipeline: gen_openlb_geom (STATS_ONLY pre-check, then the geometry) -> urban_flow (live flow,
burst, tail extrapolation) -> J = <w, Theta>/M -> intake fraction iF = B J / dx^3 (ppm), the
number the optimizer minimises (comparable across dx).

Every evaluation lives in WORKDIR/evals/<key>/ with result.json; a finished key is never run
again, so the driver can be stopped and restarted at any point. Large fields the optimizer does
not need are deleted after the run (the flow snapshots); Theta and the receptor field are kept,
so J can be re-scored later (e.g. with different receptor weights) without new CFD.
"""
import hashlib, json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tools")); sys.path.insert(0, os.path.join(ROOT, "tests"))
import param_space_density as P          # noqa: E402
import wind_rose                         # noqa: E402

B_BREATH = 1.6e-4   # m^3/s (as tests/rank_report.py)


def _read5(fn, dtype):
    import numpy as np
    with open(fn, "rb") as f:
        h = np.frombuffer(f.read(20), dtype=np.int32)
        a = np.frombuffer(f.read(), dtype=dtype)
    return h, a


def intake_fraction(geom, out):
    """(J, iF ppm, meta) from a finished run (the same formula as tests/rank_report.py)."""
    import numpy as np
    hw, w = _read5(os.path.join(geom, "receptor_w.f32"), np.float32)
    ht, th = _read5(os.path.join(out, "theta.f32"), np.float32)
    if tuple(ht[:3]) != tuple(hw[:3]):
        raise ValueError(f"theta grid {ht[:3]} != receptor grid {hw[:3]}")
    m = {}
    for line in open(os.path.join(out, "meta_flow.txt")):
        p = line.split()
        if len(p) == 2 and not p[0].startswith("#"):
            try: m[p[0]] = float(p[1])
            except ValueError: m[p[0]] = p[1]                  # text values, e.g. stop_reason
    emit = m.get("mass_emitted_total", m["mass_emitted"])
    J = float(np.dot(w.astype(np.float64), th.astype(np.float64)) / emit)
    dx = m.get("dx_m", hw[3] / 1000.0)
    return J, B_BREATH * J / dx ** 3 * 1e6, m


def run_checks(J, m, max_closure=1e-2, clear_frac=0.01, max_tail_frac=0.15):
    """Reasons a finished run must not reach the optimizer (empty list = valid).

    A run whose exit code is 0 can still carry a wrong J: one that hit MAX_BURST_STEPS before
    clearing has truncated Theta, and urban_flow applies the tail fit at the cap without its
    TAIL_MAXFRAC limit. Checks: J finite and positive; mass budget closes (gate 7a's 1 %); the
    burst cleared (stop_reason clearance or tail), or hit the cap with a tail fit that is a
    minor share of J; any applied tail within max_tail_frac."""
    import math
    why = []
    if not (isinstance(J, float) and math.isfinite(J) and J > 0):
        why.append(f"J = {J} is not finite and positive")
    closure = m.get("budget_closure")
    if not isinstance(closure, float):
        why.append("meta_flow.txt has no budget_closure")
    elif closure > max_closure:
        why.append(f"mass budget closes only to {closure:.2e} (> {max_closure:g})")
    tail_applied = m.get("tail_applied", 0.0) == 1.0
    tail_frac = m.get("tail_fit_frac", 0.0) if tail_applied else 0.0
    if tail_applied and isinstance(tail_frac, float) and tail_frac > max_tail_frac:
        why.append(f"extrapolated tail is {100 * tail_frac:.1f} % of J (> {100 * max_tail_frac:.0f} %)")
    reason = m.get("stop_reason")
    if reason is None:                                         # solver older than stop_reason
        air = m.get("tail_air_frac_at_stop")
        if not tail_applied and isinstance(air, float) and air > clear_frac:
            why.append(f"burst did not clear: {100 * air:.1f} % of peak airborne at stop, no tail fit")
    elif reason == "cap" and not tail_applied:
        air, cap = m.get("tail_air_frac_at_stop"), m.get("burst_cap")
        air_txt = f", {100 * air:.1f} % of peak still airborne" if isinstance(air, float) else ""
        cap_txt = f" ({cap:.0f} steps)" if isinstance(cap, float) else ""
        why.append(f"hit MAX_BURST_STEPS{cap_txt} before clearing{air_txt}, no tail fit")
    elif reason not in ("clearance", "tail", "cap"):
        why.append(f"unknown stop_reason {reason!r}")
    return why


class CFDEvaluator:
    """Runs the generator and the solver. `run_env` adds solver settings (e.g. TAU, D_LOCAL)."""

    def __init__(self, gen, solver, workdir, run_env=None, gen_env=None, keep=("theta.f32", "deposition.f32")):
        self.gen = os.path.abspath(gen); self.solver = os.path.abspath(solver)
        self.workdir = os.path.abspath(workdir); os.makedirs(os.path.join(self.workdir, "evals"), exist_ok=True)
        self.run_env = {"STEP4": "1", "TAIL": "1", "TAIL_CLEAR": "0.05", "CLEAR_FRAC": "0.01",
                        "SPINUP_FT": "3", "AVG_FT": "0", "CHECK_EVERY": "2000", "TS_EVERY": "200"}
        self.run_env.update(run_env or {})
        self.gen_env = dict(gen_env or {})
        self.keep = set(keep)

    # ── helpers ──
    def _env(self, extra):
        e = dict(os.environ); e.update(extra); return e

    def design_env(self, x):
        return {**P.gen_env(P.to_physical_density(x)), **self.gen_env}

    def key(self, x, phi, seed):
        s = json.dumps({"x": [round(float(v), 9) for v in x], "phi": round(float(phi) % 360.0, 6), "seed": int(seed),
                        "gen": self.gen, "solver": self.solver, "run": self.run_env, "genenv": self.gen_env}, sort_keys=True)
        return hashlib.sha1(s.encode()).hexdigest()[:16]

    def feasible(self, x):
        """The generator's floor-area / height-limit check (STATS_ONLY, ~10 ms)."""
        r = subprocess.run([self.gen], env=self._env({**self.design_env(x), "STATS_ONLY": "1"}),
                           capture_output=True, text=True)
        return r.returncode == 0

    # ── one evaluation ──
    def __call__(self, x, phi, seed):
        k = self.key(x, phi, seed)
        d = os.path.join(self.workdir, "evals", k)
        res_fn = os.path.join(d, "result.json")
        if os.path.exists(res_fn):
            return json.load(open(res_fn))
        os.makedirs(d, exist_ok=True)
        orient, wdeg = wind_rose.case_for_direction(phi)
        res = {"key": k, "x": [float(v) for v in x], "phi": float(phi) % 360.0, "seed": int(seed),
               "orient": orient, "wind_deg": wdeg, "feasible": True}
        t0 = time.time()
        geom, out = os.path.join(d, "geom"), os.path.join(d, "run")
        g = subprocess.run([self.gen], env=self._env({**self.design_env(x), "ORIENT": str(orient),
                                                      "WIND_DEG": f"{wdeg:g}", "OUT_DIR": geom}),
                           capture_output=True, text=True)
        open(os.path.join(d, "gen.log"), "w").write(g.stdout + g.stderr)
        if g.returncode == 3:
            res["feasible"] = False
        elif g.returncode != 0:
            raise RuntimeError(f"generator failed ({g.returncode}), see {d}/gen.log")
        else:
            with open(os.path.join(d, "run.log"), "w") as log:
                r = subprocess.run([self.solver], env=self._env({**self.run_env, "GEOM_DIR": geom, "OUT_DIR": out,
                                                                 "ABL_SEED": str(seed)}),
                                   stdout=log, stderr=subprocess.STDOUT, cwd=d)
            if r.returncode != 0:
                raise RuntimeError(f"solver failed ({r.returncode}), see {d}/run.log")
            J, iF, m = intake_fraction(geom, out)
            why = run_checks(J, m)
            res.update({"J": J, "iF_ppm": iF, "burst_steps": m.get("burst_steps"),
                        "budget_closure": m.get("budget_closure"), "tail_J_part": m.get("tail_J_part", 0.0),
                        "stop_reason": m.get("stop_reason"), "precision": m.get("precision", "double"),
                        "valid": not why, "reject": why})
            if why:                                     # keep the numbers for diagnosis, out of the GP
                res.update({"J_raw": J, "iF_raw_ppm": iF, "J": None, "iF_ppm": None})
            for fn in os.listdir(out):                      # drop what the optimizer never reads
                if fn.endswith(".f32") and fn not in self.keep:
                    os.remove(os.path.join(out, fn))
        res["wall_s"] = time.time() - t0
        json.dump(res, open(res_fn + ".tmp", "w"), indent=1); os.replace(res_fn + ".tmp", res_fn)
        return res


class SyntheticEvaluator:
    """Same interface, a known function instead of CFD (tests/test_driver_synth.py). Exposure
    rises away from an optimum, with a design-dependent street-alignment term in the wind
    direction and a categorical effect."""

    def __init__(self, workdir, noise=0.005, seed=0):
        import numpy as np
        self.workdir = workdir; os.makedirs(workdir, exist_ok=True)
        self.noise = noise; self.rng = np.random.default_rng(seed)
        self.cat = [P.NAMES_DENSITY.index(n) for n, lo, hi, s in P.PARAM_SPACE_DENSITY if s == "int"]
        self.n_calls = 0

    def F(self, x):
        import numpy as np
        x = np.asarray(x, float)
        cont = [i for i in range(len(x)) if i not in self.cat]
        opt = np.linspace(0.25, 0.75, len(cont))
        f = 20.0 * (1 + 0.6 * np.sum((x[cont] - opt) ** 2) / len(cont) * 4)
        cval = int(np.clip(int(x[self.cat[0]] * 4), 0, 3))
        return f * (1.0 + 0.06 * (cval != 1))

    def J(self, x, phi):
        import numpy as np
        x = np.asarray(x, float); th = np.radians(phi)
        A = 0.03 + 0.10 * x[2]                                   # wider streets: stronger alignment effect
        return self.F(x) * (1 + A * np.cos(4 * th) + 0.03 * np.cos(th - 0.7))

    def feasible(self, x):
        return True

    def __call__(self, x, phi, seed):
        self.n_calls += 1
        rng = __import__("numpy").random.default_rng(seed)
        v = self.J(x, phi) * (1 + self.noise * rng.standard_normal())
        return {"key": f"syn{self.n_calls}", "x": [float(t) for t in x], "phi": float(phi) % 360, "seed": int(seed),
                "feasible": True, "iF_ppm": float(v), "wall_s": 0.0}
