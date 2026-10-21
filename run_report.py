#!/usr/bin/env python3
# run_report.py — post-run analysis of a stored forward_city run (no LBM, no weights dir).
#
# Rebuilds each building's receptor sets exactly as forward_city does (env(b): outdoor
# fluid cells face-adjacent to the building's shell, z in 1..hz; street(b): those at
# z_ped) from geom_type.u8 + city_density.txt, then reports from the stored dose fields:
#   1. pathway totals J_in, J_out, J (self-check: must equal exposure_split.py)
#   2. spread check: last member vs ensemble mean, corrected and run's own weights
#   3. who is exposed: per-person dose by building — inequality (top-10% share, Gini),
#      Spearman rank correlation vs height and downwind position (permutation p-values)
#   4. timing: street exposure rate rebuilt from frames/ (the z_ped slice), time to
#      50% / 90% of the outdoor dose
# Writes per-building values to RUN_DIR/run_report_buildings.csv.
#
# Usage:  python3 run_report.py RUN_DIR
# Receptor constants follow forward_city's env knobs (FINF, F_IN, F_OUT, Z_PED, FLAT_W);
# set them the same way as for the run if it used non-defaults.
import sys, os, csv, glob
import numpy as np

FLUID, SHELL = 0, 2
USAGE = {1: "business", 2: "res_high", 3: "res_low"}

def load(path, shape_only=False):
    """forward_city binary: 5 int32 [nx, ny, nz, dx*1000, ncomp] + payload, as (nz, ny, nx)."""
    nx, ny, nz, dxm, nc = (int(v) for v in np.fromfile(path, dtype=np.int32, count=5))
    a = np.fromfile(path, dtype=np.uint8 if path.endswith(".u8") else np.float32, offset=20)
    if a.size != nx * ny * nz * nc:
        sys.exit(f"{path}: payload {a.size} != {nx}x{ny}x{nz}x{nc}: file truncated?")
    return a.reshape(nz, ny, nx), dxm / 1000.0

def meta(run):
    m = {}
    for line in open(os.path.join(run, "meta.txt")):
        if not line.startswith("#"):
            k, _, v = line.strip().partition(" "); m[k] = v
    return m

def blocks(run):
    out = []
    for line in open(os.path.join(run, "city_density.txt")):
        f = line.split()
        if f and f[0] == "B":
            x0, y0, x1, y1, bx0, by0, bx1, by1, hc, u = map(int, f[1:11])
            out.append(dict(x0=x0, y0=y0, x1=x1, y1=y1, hc=hc, usage=u, eff_inh=float(f[12])))
    return out

def receptor_sets(geom, b, zped):
    """forward_city's env(b) and street(b) as (z, y, x) index arrays, or None if skipped."""
    nz, ny, nx = geom.shape
    if b["usage"] == 0 or b["hc"] < 1 or b["eff_inh"] <= 0: return None
    x0, x1, y0, y1 = max(0, b["x0"]), min(nx, b["x1"]), max(0, b["y0"]), min(ny, b["y1"])
    if x1 <= x0 or y1 <= y0: return None
    hz = min(b["hc"], nz - 1)
    X0, X1, Y0, Y1, Z0, Z1 = max(0, x0 - 1), min(nx, x1 + 1), max(0, y0 - 1), min(ny, y1 + 1), 1, min(nz, hz + 2)
    sub = geom[Z0:Z1, Y0:Y1, X0:X1]
    src = np.zeros(sub.shape, bool)
    zs, ys, xs = slice(1 - Z0, hz + 1 - Z0), slice(y0 - Y0, y1 - Y0), slice(x0 - X0, x1 - X0)
    src[zs, ys, xs] = sub[zs, ys, xs] == SHELL
    dil = np.zeros_like(src)
    dil[1:] |= src[:-1]; dil[:-1] |= src[1:]
    dil[:, 1:] |= src[:, :-1]; dil[:, :-1] |= src[:, 1:]
    dil[:, :, 1:] |= src[:, :, :-1]; dil[:, :, :-1] |= src[:, :, 1:]
    z, y, x = np.nonzero(dil & (sub == FLUID))
    env = (z + Z0, y + Y0, x + X0)
    on = env[0] == zped
    return env, (env[0][on], env[1][on], env[2][on])

def ranks(a):
    a = np.asarray(a, float); o = np.argsort(a, kind="mergesort"); r = np.empty(len(a)); s = a[o]; i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and s[j + 1] == s[i]: j += 1
        r[o[i:j + 1]] = 0.5 * (i + j) + 1; i = j + 1
    return r

def spearman(x, y, n_perm=20000, seed=0):
    rx, ry = ranks(x), ranks(y)
    rho = np.corrcoef(rx, ry)[0, 1]
    rng = np.random.default_rng(seed)
    perm = np.array([np.corrcoef(rx, rng.permutation(ry))[0, 1] for _ in range(n_perm)])
    return rho, (1 + np.sum(np.abs(perm) >= abs(rho) - 1e-12)) / (n_perm + 1)

def main():
    if len(sys.argv) != 2: sys.exit("usage: run_report.py RUN_DIR")
    run = sys.argv[1]; m = meta(run)
    n_omega = int(m["omega_source_cells"]); wind = np.deg2rad(float(m.get("wind_deg", 0.0)))
    env = lambda k, d: float(os.environ.get(k, d))
    f_in, f_out = np.float32(env("F_IN", 0.87)), np.float32(env("F_OUT", 0.075))
    finf = min(np.float32(env("FINF", 0.62)), np.float32(1.0))
    flat = int(os.environ.get("FLAT_W", "0")) != 0

    geom, dx = load(os.path.join(run, "geom_type.u8"))
    nz, ny, nx = geom.shape
    zped = min(nz - 1, max(1, int(np.floor(env("Z_PED", 2.0) / dx + 0.5))))
    th, _ = load(os.path.join(run, "theta_mean.f32"))
    has_last = os.path.exists(os.path.join(run, "theta_last.f32"))
    thl = load(os.path.join(run, "theta_last.f32"))[0] if has_last else None
    print(f"run {run}: grid {nx}x{ny}x{nz} @ {dx:g} m, N_ens {m.get('N_ens', '?')}, |Omega| {n_omega}, "
          f"z_ped slice {zped}, f_in {f_in:g} F_inf {finf:g} f_out {f_out:g}")

    # ── per-building receptor sets and doses ─────────────────────────────────
    # city_density.txt rounds occupancy to 0.1 person; the run's receptor_w.f32 holds
    # N_b·f_in·F_inf/|env(b)| exactly on every envelope cell above z_ped (no street term
    # there, in any forward_city version), so N_b is recovered from it when present.
    wpath = os.path.join(run, "receptor_w.f32")
    w_run = load(wpath)[0] if os.path.exists(wpath) else None
    if w_run is None: print("   (no receptor_w.f32: occupancy from city_density.txt, rounded to 0.1)")
    rows, w_out2d, n_nostreet, worst = [], np.zeros((ny, nx)), 0, 0.0
    for i, b in enumerate(blocks(run)):
        sets = receptor_sets(geom, b, zped)
        if sets is None: continue
        e, s = sets
        if len(e[0]) == 0: continue
        Nb = 1.0 if flat else b["eff_inh"]
        if w_run is not None:
            up = e[0] != zped
            Nb = float(np.median(w_run[e][up])) * len(e[0]) / (float(f_in) * float(finf)) if not flat else 1.0
            worst = max(worst, abs(Nb - (1.0 if flat else b["eff_inh"])))
        d_env = float(th[e].astype(np.float64).mean())
        d_st = float(th[s].astype(np.float64).mean()) if len(s[0]) else 0.0
        if len(s[0]): np.add.at(w_out2d, (s[1], s[2]), Nb * float(f_out) / len(s[0]))
        else: n_nostreet += 1
        p_in = float(f_in) * float(finf) * d_env / n_omega
        p_out = float(f_out) * d_st / n_omega if len(s[0]) else 0.0
        xc, yc = 0.5 * (b["x0"] + b["x1"]) * dx, 0.5 * (b["y0"] + b["y1"]) * dx
        r = dict(block=i, usage=USAGE.get(b["usage"], b["usage"]), height_m=b["hc"] * dx, people=Nb,
                 x_m=xc, y_m=yc, env_cells=len(e[0]), street_cells=len(s[0]),
                 mean_env_dose=d_env, mean_street_dose=d_st, dose_per_person=p_in + p_out,
                 dose_per_person_in=p_in, dose_per_person_out=p_out)
        if has_last:
            r["last_in"] = float(f_in) * float(finf) * float(thl[e].astype(np.float64).mean()) / n_omega
            r["last_out"] = float(f_out) * float(thl[s].astype(np.float64).mean()) / n_omega if len(s[0]) else 0.0
        rows.append(r)
    if worst > 0.06:
        sys.exit(f"occupancy from receptor_w.f32 differs from city_density.txt by up to {worst:.2f} people: "
                 "the run used other FINF / F_IN / FLAT_W values; set them to match and rerun")
    N = np.array([r["people"] for r in rows]); p = np.array([r["dose_per_person"] for r in rows])
    J_in = sum(r["people"] * r["dose_per_person_in"] for r in rows)
    J_out = sum(r["people"] * r["dose_per_person_out"] for r in rows)
    J = J_in + J_out
    s0 = np.array([r["x_m"] * np.cos(wind) + r["y_m"] * np.sin(wind) for r in rows])
    for r, s in zip(rows, s0): r["downwind_m"] = s - s0.min(); r["share_of_J"] = r["people"] * r["dose_per_person"] / J

    print(f"\n1. pathways ({len(rows)} inhabited buildings, {N.sum():.0f} people; "
          f"{n_nostreet} without street cells)")
    print(f"   J_in  = {J_in:.6e}  ({100 * J_in / J:.1f}%)\n   J_out = {J_out:.6e}  ({100 * J_out / J:.1f}%)\n"
          f"   J     = {J:.6e}   <- must match exposure_split.py for this run")
    if "J_in_mean" in m:
        print(f"   solver's own split: J_in {float(m['J_in_mean']):.6e}, J_out {float(m['J_out_mean']):.6e}")

    # ── 2. spread check ──────────────────────────────────────────────────────
    if has_last:
        Jl = sum(r["people"] * (r["last_in"] + r["last_out"]) for r in rows)
        print(f"\n2. spread: last member vs ensemble mean, corrected weights: {100 * (Jl / J - 1):+.2f}%")
        wrun = os.path.join(run, "receptor_w.f32")
        if os.path.exists(wrun) and "J_mean" in m:
            w = load(wrun)[0]
            Jl_run = float(np.dot(w.ravel().astype(np.float64), thl.ravel().astype(np.float64))) / n_omega
            print(f"   same check with the run's own weights: {100 * (Jl_run / float(m['J_mean']) - 1):+.2f}%"
                  f"  (similar values => the corrected spread is similar)")

    # ── 3. who is exposed ────────────────────────────────────────────────────
    o = np.argsort(-p); cumN = np.cumsum(N[o]) / N.sum(); dose = N[o] * p[o] / J
    k = np.searchsorted(cumN, 0.10); prev = cumN[k - 1] if k else 0.0
    top10 = dose[:k].sum() + dose[k] * (0.10 - prev) / (cumN[k] - prev)
    a = np.argsort(p); L = np.concatenate([[0], np.cumsum(N[a] * p[a]) / J]); w_ = N[a] / N.sum()
    gini = 1 - np.sum(w_ * (L[1:] + L[:-1]))
    pw = lambda sel: np.sum(N[sel] * p[sel]) / np.sum(N[sel])
    print(f"\n3. per-person dose across buildings (population-weighted mean {J / N.sum():.4g})")
    print(f"   most / least exposed building: {p.max() / p.min():.1f}x   median {np.median(p):.4g}")
    print(f"   most exposed 10% of people receive {100 * top10:.1f}% of the dose   Gini {gini:.3f}")
    for u in sorted({r['usage'] for r in rows}, key=str):
        sel = np.array([r["usage"] == u for r in rows])
        print(f"   {u:9s}: {sel.sum():3d} buildings, {N[sel].sum():7.0f} people, mean per-person dose {pw(sel):.4g}")
    h = np.array([r["height_m"] for r in rows]); dw = np.array([r["downwind_m"] for r in rows])
    for lab, x in (("building height", h), ("downwind position", dw)):
        rho, pv = spearman(x, p)
        ptxt = f"p = {pv:.3f}" if pv >= 0.001 else "p < 0.001"
        print(f"   Spearman per-person dose vs {lab:17s}: rho = {rho:+.2f}, permutation {ptxt}  (n = {len(p)})")

    # ── 4. timing (street pathway, from the z_ped frames of the last member) ──
    ts_path, frames = os.path.join(run, "exposure_timeseries.csv"), sorted(glob.glob(os.path.join(run, "frames", "conc_*.f32")))
    if os.path.exists(ts_path) and frames:
        # Θ accumulates C once per solver step (CINT += C), so integrate over steps.
        rows_ts = {int(r["frame"]): r for r in csv.DictReader(open(ts_path))}
        t, st, rate = [], [], []
        for fp in frames:
            fi = int(os.path.basename(fp)[5:9])
            if fi not in rows_ts: continue
            fnx, fny = (int(v) for v in np.fromfile(fp, dtype=np.int32, count=2))
            c = np.fromfile(fp, dtype=np.float32, offset=8).reshape(fny, fnx)
            t.append(float(rows_ts[fi]["t_s"])); st.append(float(rows_ts[fi]["phaseB_step"]))
            rate.append(float(np.sum(w_out2d * c)) / n_omega)
        t, st, rate = np.array(t), np.array(st), np.array(rate)
        cum = np.concatenate([[0], np.cumsum(0.5 * (rate[1:] + rate[:-1]) * np.diff(st))])
        print(f"\n4. street-exposure timing (last member, {len(t)} frames, every {np.median(np.diff(t)):.1f} s)")
        print(f"   peak sampled rate at {t[np.argmax(rate)]:.0f} s (end of run {t[-1]:.0f} s)")
        when = lambda c, q: float(np.interp(q, c, t))
        if has_last:
            Jo_last = sum(r["people"] * r["last_out"] for r in rows)
            frac = cum[-1] / Jo_last
            print(f"   frames capture {100 * frac:.0f}% of the exact outdoor dose (theta_last); the rest falls "
                  f"between snapshots, mostly in the first {t[1] - t[0]:.0f} s, which contain the release pulse")
            early = np.concatenate([[0], (1 - frac) + cum[1:] / Jo_last]) if frac < 1 else cum / Jo_last
            late = cum / cum[-1]
            for q in (0.5, 0.9, 0.99):
                print(f"   {int(100 * q):2d}% of outdoor dose by {when(early, q):5.0f} s if the missing dose is in the first "
                      f"interval;  {when(late, q):5.0f} s uncorrected (latest)")
        else:
            late = cum / cum[-1]
            print("   (no theta_last.f32: cannot check sampling)  " +
                  ", ".join(f"{int(100 * q)}% by {when(late, q):.0f} s" for q in (0.5, 0.9, 0.99)))
    else:
        print("\n4. timing: no frames/ or exposure_timeseries.csv in this run (SNAP_EVERY=0?)")

    out = os.path.join(run, "run_report_buildings.csv")
    keys = ["block", "usage", "height_m", "people", "x_m", "y_m", "downwind_m", "env_cells", "street_cells",
            "mean_env_dose", "mean_street_dose", "dose_per_person", "dose_per_person_in", "dose_per_person_out", "share_of_J"]
    with open(out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore"); wr.writeheader(); wr.writerows(rows)
    print(f"\nper-building table: {out}")

if __name__ == "__main__":
    main()
