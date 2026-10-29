#!/usr/bin/env python3
"""must_score.py — score urban_flow against measured data on an obstacle array (MUST).

    python3 tests/must_score.py GEOM_DIR OUT_DIR [OUT_DIR ...] [--flow OBS.csv] [--conc OBS.csv]
                                [--uref-point X,Y,Z --uref-obs U] [--uref U --H H] [--frame file|grid]
    python3 tests/must_score.py --selftest

GEOM_DIR from gen_gate6_geom CASE=obstacles (its meta_geom.txt holds the rotation and origin that
map the data's frame onto the grid); OUT_DIR(s) from urban_flow on it, several = inlet seeds,
averaged. Observation files are CSV with a header; columns by name, case-insensitive:
  --flow   x, y, z and any of u, v, w (time-mean velocity, the data's frame; z above ground)
  --conc   x, y, z, c (time-mean concentration, normalised: C* = c U_ref H^2 / Q)
The data's frame is the one OBST_FILE uses; --frame grid takes points already in grid metres.

Model side. Flow: the time mean uavg.f32 (run with AVG_FT > 0), components turned back into the
data's frame; with --uref-point/--uref-obs, model velocities are scaled by U_obs_ref / U_model_ref
(the speed at that point), so normalised data and a model at another inflow speed compare. Scalar:
a puff's Theta is the steady field of a continuous source of the same rate (tests/plume_check.py),
so C*_model = Theta / dx^3 / M_emitted * U_ref * H^2 (--uref, --H; H defaults to the tallest
block). In a turbulent flow one puff's Theta is one realisation of that mean: give several seeds.

Metrics (Chang & Hanna 2004; Hanna & Chang 2012 for the urban acceptance; VDI 3783-9 / COST 732
for the hit rate):
  FB = (mean O - mean P) / (0.5 (mean O + mean P))      NMSE = mean (O - P)^2 / (mean O mean P)
  FAC2 = share of points with 0.5 <= P/O <= 2           NAD = mean |O - P| / (mean O + mean P)
  MG = exp(mean ln O - mean ln P), VG = exp(mean (ln O - ln P)^2)   (positive pairs only)
  hit rate q = share of points with |P - O| <= D |O| or |P - O| <= W   (D 0.25, W --hit-w)
Acceptance printed: velocity q >= 0.66 per component (COST 732); concentration |FB| <= 0.67,
NMSE <= 6, FAC2 >= 0.30, NAD <= 0.50 (urban, Hanna & Chang 2012).
Exit 0 = all printed criteria met, 1 = some not, 2 = input problem.
"""
import sys, os, math, csv, argparse
import numpy as np


def read5(fn, dtype):
    with open(fn, "rb") as f:
        h = np.frombuffer(f.read(20), dtype=np.int32)
        a = np.frombuffer(f.read(), dtype=dtype)
    nx, ny, nz, dxmm, nc = (int(v) for v in h)
    return a.reshape(nc, nz, ny, nx).transpose(0, 3, 2, 1), dxmm / 1000.0       # [c, x, y, z]


def meta(path):
    d = {}
    if os.path.exists(path):
        for line in open(path):
            p = line.split()
            if len(p) == 2 and not p[0].startswith("#"):
                try: d[p[0]] = float(p[1])
                except ValueError: d[p[0]] = p[1]
    return d


def metrics(O, P, D=0.25, W=0.0):
    O = np.asarray(O, float); P = np.asarray(P, float)
    mo, mp = O.mean(), P.mean()
    r = {"n": len(O), "FB": (mo - mp) / (0.5 * (mo + mp)) if mo + mp != 0 else float("nan"),
         "NMSE": float(np.mean((O - P) ** 2) / (mo * mp)) if mo * mp > 0 else float("nan"),
         "NAD": float(np.mean(np.abs(O - P)) / (mo + mp)) if mo + mp != 0 else float("nan"),
         "hit": float(np.mean((np.abs(P - O) <= D * np.abs(O)) | (np.abs(P - O) <= W)))}
    with np.errstate(divide="ignore", invalid="ignore"):
        q = P / O
    r["FAC2"] = float(np.mean((O != 0) & (q >= 0.5) & (q <= 2)))
    pos = (O > 0) & (P > 0)
    if pos.any():
        lo, lp = np.log(O[pos]), np.log(P[pos])
        r["MG"] = float(math.exp(lo.mean() - lp.mean())); r["VG"] = float(math.exp(np.mean((lo - lp) ** 2)))
    return r


class Grid:
    """data frame <-> grid: rotation about the array centre, origin, cell centres (k - 1/2) dx up"""
    def __init__(self, geom, frame):
        m = meta(os.path.join(geom, "meta_geom.txt"))
        mat, dx = read5(os.path.join(geom, "material_map.dat"), np.int32)
        self.mat = mat[0]; self.dx = dx; self.shape = self.mat.shape
        self.frame = frame
        self.rot = math.radians(m.get("rot_deg", 0.0)) if frame == "file" else 0.0
        self.c = (m.get("array_cx", 0.0), m.get("array_cy", 0.0)) if frame == "file" else (0.0, 0.0)
        self.o = (m.get("x0_rot", 0.0), m.get("y0_rot", 0.0)) if frame == "file" else (0.0, 0.0)
        self.Hmax = m.get("H_max_m")

    def index(self, x, y, z):
        cr, sr = math.cos(self.rot), math.sin(self.rot)
        px, py = x - self.c[0], y - self.c[1]
        return ((cr * px - sr * py - self.o[0]) / self.dx, (sr * px + cr * py - self.o[1]) / self.dx, z / self.dx + 0.5)

    def to_data(self, u, v):           # grid-frame vector components -> data frame
        cr, sr = math.cos(self.rot), math.sin(self.rot)
        return cr * u + sr * v, -sr * u + cr * v

    def interp(self, f, x, y, z):
        """trilinear on cell centres; None if a corner is not fluid (inside a block, or a face)"""
        fi, fj, fk = self.index(x, y, z)
        i, j, k = int(math.floor(fi)), int(math.floor(fj)), int(math.floor(fk))
        if i < 0 or j < 0 or k < 1 or i + 1 >= self.shape[0] or j + 1 >= self.shape[1] or k + 1 >= self.shape[2]:
            return None
        if np.any(self.mat[i:i + 2, j:j + 2, k:k + 2] != 1):
            return None
        a, b, c = fi - i, fj - j, fk - k
        w = np.array([(1 - a) * (1 - b) * (1 - c), a * (1 - b) * (1 - c), (1 - a) * b * (1 - c), a * b * (1 - c),
                      (1 - a) * (1 - b) * c, a * (1 - b) * c, (1 - a) * b * c, a * b * c])
        cube = f[..., i:i + 2, j:j + 2, k:k + 2]
        vals = np.stack([cube[..., 0, 0, 0], cube[..., 1, 0, 0], cube[..., 0, 1, 0], cube[..., 1, 1, 0],
                         cube[..., 0, 0, 1], cube[..., 1, 0, 1], cube[..., 0, 1, 1], cube[..., 1, 1, 1]], -1)
        return vals @ w


def read_obs(fn, need):
    rows = list(csv.DictReader(open(fn)))
    if not rows: raise SystemExit(f"{fn}: no rows")
    keys = {k.strip().lower(): k for k in rows[0]}
    miss = [k for k in need if k not in keys]
    if miss: raise SystemExit(f"{fn}: missing column(s) {miss} (have {list(keys)})")
    out = {k: np.array([float(r[keys[k]]) for r in rows]) for k in keys if k in ("x", "y", "z", "u", "v", "w", "c")}
    return out


def show(name, r, crit):
    ok = True; parts = []
    for k in ("n", "FB", "NMSE", "FAC2", "NAD", "MG", "VG", "hit"):
        if k in r: parts.append(f"{k} {r[k]:.3g}" if k != "n" else f"n {r[k]}")
    verdict = []
    for k, test, txt in crit:
        good = bool(test(r[k])) if k in r and not (isinstance(r[k], float) and math.isnan(r[k])) else False
        ok &= good; verdict.append(f"{txt} {'ok' if good else 'NOT MET'}")
    print(f"[must] {name:14s} " + "  ".join(parts))
    if verdict: print(f"[must] {'':14s} " + "; ".join(verdict))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("geom", nargs="?"); ap.add_argument("outs", nargs="*")
    ap.add_argument("--flow"); ap.add_argument("--conc")
    ap.add_argument("--uref-point"); ap.add_argument("--uref-obs", type=float)
    ap.add_argument("--uref", type=float, help="U_ref (m/s) in C* = c U H^2 / Q, model side")
    ap.add_argument("--H", type=float, help="H (m) in C*; default: the tallest block")
    ap.add_argument("--hit-w", type=float, default=0.008, help="hit-rate absolute tolerance W (units of the data)")
    ap.add_argument("--frame", choices=("file", "grid"), default="file")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest: return selftest()
    if not a.geom or not a.outs or not (a.flow or a.conc):
        ap.print_help(); return 2
    G = Grid(a.geom, a.frame)
    ok = True
    if a.flow:
        obs = read_obs(a.flow, ["x", "y", "z"])
        U = None
        for o in a.outs:
            f = os.path.join(o, "uavg.f32")
            if not os.path.exists(f): print(f"[must] {o}: no uavg.f32 (run with AVG_FT > 0)"); return 2
            u, dx = read5(f, np.float32); u = u[:3].astype(np.float64)
            U = u if U is None else U + u
        U /= len(a.outs)
        scale = 1.0
        if a.uref_point and a.uref_obs:
            xr, yr, zr = (float(v) for v in a.uref_point.split(","))
            ur = G.interp(U, xr, yr, zr)
            if ur is None: print("[must] the reference point is not in the fluid"); return 2
            scale = a.uref_obs / float(np.linalg.norm(ur))
            print(f"[must] model speed at the reference point {np.linalg.norm(ur):.3f} m/s -> scale {scale:.4f}")
        P = {"u": [], "v": [], "w": []}; O = {"u": [], "v": [], "w": []}; skipped = 0
        for n in range(len(obs["x"])):
            m = G.interp(U, obs["x"][n], obs["y"][n], obs["z"][n])
            if m is None: skipped += 1; continue
            pu, pv = G.to_data(m[0], m[1]); pw = m[2]
            for k, pval in (("u", pu), ("v", pv), ("w", pw)):
                if k in obs: O[k].append(obs[k][n]); P[k].append(scale * pval)
        print(f"[must] flow: {len(obs['x'])} points, {skipped} not in the fluid (inside a block or the domain faces)")
        for k in ("u", "v", "w"):
            if O[k]:
                ok &= show(f"velocity {k}", metrics(O[k], P[k], W=a.hit_w), [("hit", lambda q: q >= 0.66, "q >= 0.66")])
        if O["u"] and O["v"]:
            so = np.hypot(O["u"], O["v"]); sp = np.hypot(P["u"], P["v"])
            show("speed (u,v)", metrics(so, sp, W=a.hit_w), [])
    if a.conc:
        obs = read_obs(a.conc, ["x", "y", "z", "c"])
        if not a.uref: print("[must] --conc needs --uref (U_ref of the data's C* normalisation)"); return 2
        H = a.H or G.Hmax
        TH = None
        for o in a.outs:
            th, dx = read5(os.path.join(o, "theta.f32"), np.float32)
            mf = meta(os.path.join(o, "meta_flow.txt"))
            Mrel = mf.get("mass_emitted_total", mf.get("mass_emitted"))
            c = th[0].astype(np.float64) / dx ** 3 / Mrel * a.uref * H * H
            TH = c if TH is None else TH + c
        TH /= len(a.outs)
        O_, P_, skipped = [], [], 0
        for n in range(len(obs["x"])):
            m = G.interp(TH, obs["x"][n], obs["y"][n], obs["z"][n])
            if m is None: skipped += 1; continue
            O_.append(obs["c"][n]); P_.append(float(m))
        print(f"[must] concentration: {len(obs['x'])} points, {skipped} not in the fluid; C* = c U H^2/Q with U {a.uref} m/s, H {H} m;"
              f" {len(a.outs)} run(s) averaged")
        ok &= show("C*", metrics(O_, P_), [("FB", lambda v: abs(v) <= 0.67, "|FB| <= 0.67"), ("NMSE", lambda v: v <= 6, "NMSE <= 6"),
                                           ("FAC2", lambda v: v >= 0.30, "FAC2 >= 0.30"), ("NAD", lambda v: v <= 0.50, "NAD <= 0.50")])
    print(f"[must] {'all criteria met' if ok else 'some criteria NOT met'}")
    return 0 if ok else 1


def selftest():
    fails = 0
    def chk(name, cond):
        nonlocal fails; print(f"  [{'PASS' if cond else 'FAIL'}] {name}"); fails += not cond
    O = np.array([1.0, 2.0, 4.0, 8.0])
    r = metrics(O, O)
    chk("P = O: FB 0, NMSE 0, FAC2 1, NAD 0, MG 1, VG 1, hit 1",
        abs(r["FB"]) < 1e-12 and r["NMSE"] < 1e-12 and r["FAC2"] == 1 and r["NAD"] < 1e-12
        and abs(r["MG"] - 1) < 1e-12 and abs(r["VG"] - 1) < 1e-12 and r["hit"] == 1)
    r = metrics(O, 2 * O)
    chk("P = 2 O: FB = -2/3, FAC2 = 1, MG = 1/2, NMSE = mean O^2 / (2 mean(O)^2)", abs(r["FB"] + 2 / 3) < 1e-12 and r["FAC2"] == 1
        and abs(r["MG"] - 0.5) < 1e-12 and abs(r["NMSE"] - np.mean(O ** 2) / (2 * np.mean(O) ** 2)) < 1e-12)
    r = metrics(O, 2.01 * O)
    chk("P = 2.01 O: FAC2 = 0", r["FAC2"] == 0)
    r = metrics([1.0, 1.0], [1.2, 1.3], D=0.25)
    chk("hit rate: 20 % off is a hit, 30 % is not", abs(r["hit"] - 0.5) < 1e-12)
    # interpolation and frame: a linear field is reproduced exactly; the rotation round-trips
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        nx, ny, nz, dx = 12, 10, 8, 0.5
        mat = np.ones((nx, ny, nz), np.int32); mat[:, :, 0] = 7
        with open(os.path.join(d, "material_map.dat"), "wb") as f:
            np.array([nx, ny, nz, 500, 1], np.int32).tofile(f); mat.transpose(2, 1, 0).astype(np.int32).tofile(f)
        open(os.path.join(d, "meta_geom.txt"), "w").write("rot_deg 30\narray_cx 1.0\narray_cy 2.0\nx0_rot -2.5\ny0_rot -2.0\nH_max_m 1\n")
        G = Grid(d, "file")
        ii, jj, kk = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
        f = 3.0 * ii + 2.0 * jj - kk
        x, y, z = 0.7, 2.4, 1.3
        fi, fj, fk = G.index(x, y, z)
        chk("trilinear reproduces a linear field", abs(G.interp(f, x, y, z) - (3 * fi + 2 * fj - fk)) < 1e-9)
        u, v = G.to_data(math.cos(math.radians(30)), math.sin(math.radians(30)))
        chk("rotation 30 deg: a grid vector at +30 deg is the data's +x", abs(u - 1) < 1e-12 and abs(v) < 1e-12)
    print("ALL PASS" if not fails else f"{fails} FAILED")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
