# Lab box runbook — OpenLB port (gates + production city)

Everything is driven by `lab_openlb.sh` at the repo root. It is resumable, logs every step, and
records PASS/FAIL per gate in `~/olb_lab/summary.txt`. What each gate means and the results so
far (4-core cloud box): `OPENLB_PHASE5_6_GATES.md`.

## 0. Prerequisites

- g++ with C++20 (13.x tested), `make`, `curl`, `tar`, `python3` with `numpy` + `matplotlib`
- ~15 GB free under `~/olb_lab` (OpenLB tree + builds ~1 GB; the city runs' fields ~2 GB each)
- Internet for the OpenLB 1.8.1 tarball (GitLab, falling back to Zenodo). No internet: download
  `release-1.8.1.tar.gz` elsewhere, unpack it on the box and pass `OLB_ROOT=/path/release-1.8.1`.
- CPU runs: OpenMP. GPU runs (§7): an NVIDIA driver; the CUDA toolkit is fetched if needed.
  Not MPI: the per-step operators assume one cuboid per process.

## 1. Get the code

```bash
git clone https://github.com/polecatte/city-contamination-cfd.git && cd city-contamination-cfd
git checkout claude/gracious-lovelace-3wlllj
```

## 2. Run

```bash
mkdir -p ~/olb_lab
tmux new -s olb                                    # it takes hours; survive an SSH drop
./lab_openlb.sh all 2>&1 | tee -a ~/olb_lab/lab.log
# (typed INSIDE the tmux shell, so the window stays open if the script stops)
# detach: Ctrl-b d      reattach: tmux attach -t olb      progress: ./lab_openlb.sh status
```

Or stage by stage — `setup`, `gates`, `city`, `package`. A finished step is skipped on the next
call, so after a crash or a `git pull` just rerun the same command. To redo one step, delete its
marker: `rm ~/olb_lab/done/gate6b_dx2`. `THREADS=N` overrides the thread count (default: all).

## 3. What runs, and rough cost on 32 cores

Costs scale from the measured 4-core rate (14 MLUPS with WALE active); OpenMP scaling to 32
threads is untested, so treat these as ±2×.

| step | what | cells × steps | est. |
|---|---|---|---|
| setup | fetch + build OpenLB (`cpu-mt`), app, generators; Stage-A geometries | — | 10 min |
| gate5 | OpenLB's own voxel counts = Stage A, production map | 7.3 M × 1 | 1 min |
| gate6a | ABL drift < 10 % + peak \|u\| < 0.1 (empty 177×40×89 fetch) | 0.6 M × 22 k | 5 min |
| gate6b_dx4 | cube Xr/H, H/dx = 10 | 1.4 M × 26 k | 15 min |
| gate6b_dx2 | **cube Xr/H, H/dx = 20 — the test that decides 6b** | 11 M × 105 k | 3–4 h |
| gate7a/7c/7b | scalar budget, deposition, linearity (40³ box) | tiny | 3 min |
| city_s1000/2000 | production city 490×167×89, 3 FT spin-up + burst to 99 % | 7.3 M × ~85 k | 2–3 h each |
| gate8, wake | seed spread of J < 5 %; wake closed before the outlet | — | 1 min |

## 4. What to send back

`./lab_openlb.sh package` writes `~/olb_lab/olb_lab_<timestamp>.tar.gz` (summary, all logs,
time series, metadata, figures — not the large fields). That file is all I need.

## 5. Reading the summary

- **gate5, gate6a, gate7a/b/c** passed here at the same operating point; a FAIL on the lab box
  is an environment difference worth chasing before anything else.
- **gate6b_dx4** is expected to FAIL (Xr/H ≈ 2.6 here). **gate6b_dx2** is the open question:
  Xr/H moving into 1.4–1.8 (or clearly toward it) confirms resolution as the cause.
- **wake** checks the new 15 H downstream buffer: < 2 % reversed near-ground flow over the last
  15 % of the domain. On the old 70 m buffer this measured 68.9 %.
- **gate8** compares the two seeds' J at full clearance; the capped 4-core runs gave 1.27 %.

## 6. If something breaks

- `setup` fails in `olb_external` or the app build: `~/olb_lab/logs/build_app.log`; usually the
  compiler lacks C++20.
- A run aborts with `[DIVERGED]`: the log names the materials holding non-finite cells.
- Want to watch a run: `tail -f ~/olb_lab/out_6b_dx2.log` (timer lines give MLUPS and ETA).

## 7. GPU runs (Phase 8)

The same gates and city runs on the CUDA build. Everything goes to `~/olb_lab/gpu/` (own OpenLB
tree, logs, markers, summary); the geometries are shared with the CPU runs. It can run while a
CPU run is going (it needs one CPU core), but not two GPU jobs at once.

```bash
cd city-contamination-cfd && git pull
./lab_openlb.sh setup                    # CPU tree: needed as the parity reference (skips what's done)
GPU=1 ./lab_openlb.sh setup              # ~10 min; installs CUDA 12.6 into ~/olb_lab/cuda126 if needed
GPU=1 ./lab_openlb.sh parity             # ~15 min: GPU vs CPU on four short cases — run this FIRST
GPU=1 ./lab_openlb.sh gates              # then the gates, and
GPU=1 ./lab_openlb.sh city               # the production city, two seeds
./lab_openlb.sh compare                  # CPU and GPU numbers side by side
./lab_openlb.sh package                  # one tarball with both (send me this)
```

- **nvcc.** OpenLB 1.8 does not compile with CUDA 12.0 (Ubuntu 24.04's `nvidia-cuda-toolkit`):
  `fields.h(36): call to consteval function std::source_location::current…`. Setup uses `$NVCC`,
  else `nvcc` on PATH if ≥ 12.4, else installs CUDA 12.6 from conda-forge with micromamba (no
  root, ~1 GB). The build is double precision, `CUDA_ARCH` from `nvidia-smi` (A4000 = 86).
- **parity** must say `[parity] ALL PASS` before the gate numbers mean anything. It runs each
  case with the CPU binary and the GPU binary and compares every output. `box7` (frozen wind,
  linear) must agree to 1e-9; the three flow cases to 1e-4 (GPU rounding differs and the flow
  amplifies it). A FAIL names the output and the size of the difference: send the package.
- **Expected speed:** the GPU logs' `MLUPs :` lines. The A4000 is shared with the desktop
  (Xorg, Houdini); close Houdini for the long runs if it holds GPU memory — the city needs ~5 GB.
- Errors: `no CUDA device was found` = driver/`nvidia-smi` problem, not the build;
  `CUDA driver version is insufficient` = driver older than CUDA 12's minimum (525).
