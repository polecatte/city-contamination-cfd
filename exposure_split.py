#!/usr/bin/env python3
# exposure_split.py — re-score a stored forward_city run: J, J_in, J_out.
#
# J = <w, Θ> / |Ω| is linear in the receptor weights w, so the stored ensemble-mean
# dose field Θ (theta_mean.f32) can be re-scored with any weights without re-running
# the LBM. Use it to apply the building-anchored weights (EXPOSURE_METRIC.md eqs 2-3)
# to a run made before forward_city wrote receptor_w_in/out.f32: regenerate the
# weights for the SAME city with
#   DUMP_GEOM_ONLY=1 OUT_DIR=<weights_dir> bash run_forward_city.sh
# (same CELL and builder knobs as the run), then
#   python3 exposure_split.py <run_dir> [<weights_dir>]
#
# Self-check: the run's own receptor_w.f32 contracted with theta_mean.f32 must
# reproduce meta.txt's J_mean (mean of per-member J = J of the mean Θ, by linearity).
import sys, os, csv
import numpy as np

def load(path):
    """Read a forward_city binary: 5 int32 [nx, ny, nz, dx*1000, ncomp] + payload."""
    hdr = np.fromfile(path, dtype=np.int32, count=5)
    nx, ny, nz, _, nc = (int(v) for v in hdr)
    dt = np.uint8 if path.endswith(".u8") else np.float32
    a = np.fromfile(path, dtype=dt, offset=20)
    if a.size != nx * ny * nz * nc:
        sys.exit(f"{path}: payload {a.size} != {nx}x{ny}x{nz}x{nc}")
    return a, (nx, ny, nz)

def meta(run):
    m = {}
    for line in open(os.path.join(run, "meta.txt")):
        if line.startswith("#"): continue
        k, _, v = line.strip().partition(" ")
        m[k] = v
    return m

def dot(w, th):
    return float(np.dot(w.astype(np.float64), th.astype(np.float64)))

def main():
    if len(sys.argv) < 2: sys.exit(__doc__ or "usage: exposure_split.py RUN_DIR [WEIGHTS_DIR]")
    run = sys.argv[1]; wdir = sys.argv[2] if len(sys.argv) > 2 else run
    m = meta(run)
    n_omega = int(m["omega_source_cells"])
    tfile = "theta_mean.f32" if os.path.exists(os.path.join(run, "theta_mean.f32")) else "theta_last.f32"
    th, grid = load(os.path.join(run, tfile))
    print(f"run {run}: {tfile}, grid {grid[0]}x{grid[1]}x{grid[2]}, N_ens {m.get('N_ens', '?')}, |Omega| {n_omega}")

    # self-check against the run's own weights
    w, g = load(os.path.join(run, "receptor_w.f32"))
    if g != grid: sys.exit(f"receptor_w.f32 grid {g} != dose grid {grid}")
    j_run = dot(w, th) / n_omega
    j_meta = float(m.get("J_mean", "nan"))
    print(f"check: <receptor_w, Θ>/|Ω| = {j_run:.6e}   meta J_mean = {j_meta:.6e}   "
          f"(rel diff {abs(j_run - j_meta) / max(abs(j_meta), 1e-300):.1e})")
    print(f"       run's weight sum = {w.sum(dtype=np.float64):.4g}")

    # building-anchored split
    w_in, gi = load(os.path.join(wdir, "receptor_w_in.f32"))
    w_out, go = load(os.path.join(wdir, "receptor_w_out.f32"))
    if gi != grid or go != grid: sys.exit(f"weights grid {gi}/{go} != dose grid {grid} (same CELL and knobs?)")
    j_in, j_out = dot(w_in, th) / n_omega, dot(w_out, th) / n_omega
    j = j_in + j_out
    print(f"building-anchored weights from {wdir}: sum in {w_in.sum(dtype=np.float64):.4g}, out {w_out.sum(dtype=np.float64):.4g}")
    print(f"  J_in  (indoor)  = {j_in:.6e}  ({100 * j_in / j:.1f}%)")
    print(f"  J_out (outdoor) = {j_out:.6e}  ({100 * j_out / j:.1f}%)")
    print(f"  J     (total)   = {j:.6e}")

    # exposure window T: end of Phase B in the snapshot member's time series
    ts = os.path.join(run, "exposure_timeseries.csv")
    if os.path.exists(ts):
        rows = list(csv.DictReader(open(ts)))
        if rows: print(f"exposure window T (Phase B, last member) = {float(rows[-1]['t_s']):.1f} s")
    for k in ("mass_emitted", "mass_deposited", "mass_airborne", "mass_outflow"):
        if k in m: print(f"{k:15s} {float(m[k]):.4e}")

if __name__ == "__main__":
    main()
