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
git checkout claude/elegant-goodall-x5243h     # since 2026-10-29 (was claude/gracious-lovelace-3wlllj)
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

- **gate5, gate7a/b/c** passed here at the same operating point; a FAIL on the lab box is an
  environment difference worth chasing before anything else.
- **gate6a** decides at the production city face, 500 m downstream of the inlet (the guideline
  domain's 5 H_REF buffer; `CITY_FACE_M` overrides), where the lab runs drift ~4–5 %. The compact
  domain's 40 m face is still printed, with a warning when it is over 10 % (10.9 % on the lab):
  runs made with `DOMAIN=compact` meet a distorted inflow.
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
  root, ~1 GB). The build is double precision (`PRECISION=single`: a float build in its own
  tree, §10), `CUDA_ARCH` from `nvidia-smi` (A4000 = 86).
- **parity** must say `[parity] ALL PASS` before the gate numbers mean anything. It runs each
  case with the CPU binary and the GPU binary and compares every output. `box7` (frozen wind,
  linear) must agree to 1e-9; the three flow cases to 1e-4 (GPU rounding differs and the flow
  amplifies it). A FAIL names the output and the size of the difference: send the package.
- **Expected speed:** the GPU logs' `MLUPs :` lines. The A4000 is shared with the desktop
  (Xorg, Houdini); close Houdini for the long runs if it holds GPU memory — the city needs ~5 GB.
- Errors: `no CUDA device was found` = driver/`nvidia-smi` problem, not the build;
  `CUDA driver version is insufficient` = driver older than CUDA 12's minimum (525).

## 8. Showcase, 6b follow-ups, H100

```bash
GPU=1 ./lab_openlb.sh showcase    # small city end to end, ~1 h on the A4000, minutes on an H100
GPU=1 ./lab_openlb.sh cube_sens   # three dx=4 cube variants, ~10 min each on the A4000
GPU=1 ./lab_openlb.sh cube_dx1    # H100 only (>= 60 GB): the dx=1 m cube, 89 M cells
```

- **showcase**: a 400 m city (`gen_openlb_geom CITY_M=400 POP=8000`, 4.6 M cells: the fixed
  15 H wake buffer is most of the domain) run through the whole pipeline: live flow, burst to
  99 % clearance, Θ, deposition, J, and the figures (`showcase/figs/`). `showcase/SHOWCASE.txt`
  is a scorecard of six checks: flow stable (Mach), wake closed before the outlet, mass budget
  closed, clearance reached, J finite and positive, negative-Θ mass small. It runs on the CPU
  build too (drop `GPU=1`), about 6× slower.
- **cube_sens**: which knob moves the cube's Xr/H — floor model (`GROUND_MODEL=0`), inflow
  turbulence ×1.2 (`ABL_TI_SCALE`), WALE constant 0.20 (`LES_CONST`). Compare with
  gate6b_dx4's 2.63.
- **cube_dx1**: the third grid level (dx 4 → 2 → 1 m gave / will give Xr/H 2.63 → 2.31 → ?),
  which says whether the grid alone can reach 1.4–1.8. ~40 GB device memory and as much host
  RAM, double precision.

**On a different machine (the H100):** the GPU mode needs the CPU tree as the parity reference
and for the geometries, so there:
```bash
git clone -b claude/elegant-goodall-x5243h https://github.com/polecatte/city-contamination-cfd.git && cd city-contamination-cfd
./lab_openlb.sh setup                        # CPU build + geometries (~15 min)
GPU=1 ./lab_openlb.sh setup                  # CUDA_ARCH 90 read from nvidia-smi
GPU=1 ./lab_openlb.sh parity                 # must say ALL PASS on this GPU too
GPU=1 ./lab_openlb.sh cube_dx1
GPU=1 ./lab_openlb.sh showcase
./lab_openlb.sh package
```
On a Slurm cluster, wrap each line in a batch job (`sbatch --gres=gpu:1 --wrap "…"`); the steps
are resumable, so a job that hits its time limit just resumes on resubmission.

## 9. Phase 9: HRR, guideline domain, ranking study

Background and design: `OPENLB_PHASE9_FIDELITY.md`. `git pull` first; every command is resumable.

```bash
cd ~/cfd_gpu && git pull
# HRR (+ corrected WALE) build, on the lab GPU and/or the H100
MODEL=hrr GPU=1 ~/cfd_gpu/lab_openlb.sh setup
MODEL=hrr GPU=1 ~/cfd_gpu/lab_openlb.sh gates        # 6a, 6b dx4/dx2 etc. at tau 0.5001
# ranking study: the old model (lab A4000), then the corrected model (H100: dx 4; else dx 8)
GPU=1 ~/cfd_gpu/lab_openlb.sh setup                   # rebuilds the default app with the new code
GPU=1 ~/cfd_gpu/lab_openlb.sh rank
MODEL=hrr GPU=1 ~/cfd_gpu/lab_openlb.sh rank
~/cfd_gpu/lab_openlb.sh rank_report                   # -> ~/olb_lab/rank_report.txt
```

What to send back: `rank_report.txt`, `hrr/summary.txt` (or `gpu/hrr/summary.txt`) and the
`logs/` of any FAIL. The rank runs on the two machines write to their own `~/olb_lab`; copy the
H100's `rank_manifest.txt` plus the `rank_*` run directories' `theta.f32`/`meta_flow.txt` and the
`rank_geom_*` `receptor_w.f32` back if you want one combined report, or send both reports.
The guideline domain at dx 4 (32.5 M cells) needs ≈ 20 GB of GPU memory: H100 only.

### 9.1 Optimisation (after the ranking study)

```bash
~/cfd_gpu/lab_openlb.sh opt_setup                      # CPU torch + BoTorch venv, two self-tests (~4 min)
MODEL=hrr GPU=1 OPT_BUDGET=150 ~/cfd_gpu/lab_openlb.sh optimize   # resumable; rerun to continue
~/cfd_gpu/lab_openlb.sh opt_report                     # best designs so far
MODEL=hrr GPU=1 OPT_TOP=3 ~/cfd_gpu/lab_openlb.sh opt_verify      # top 3 at all 16 directions
```
Send back `opt_*/report.json`, `opt_*/verified.json` and `opt_*.log`. Method: OPTIMIZER.md.

### 9.2 Poster run (results by the next morning)

```bash
cd ~/cfd_gpu && git pull
GPU=1 ~/cfd_gpu/lab_openlb.sh setup          # regular GPU app (~10 min)
MODEL=hrr ~/cfd_gpu/lab_openlb.sh setup      # CPU HRR app with the latest code (~5 min)

# GPU queue, ~13-15 h unattended: 5 designs (1 seed) -> 3 wind directions -> second seeds
tmux new -s gpu 'RANK_CFG=quick GPU=1 ~/cfd_gpu/lab_openlb.sh rank; \
  RANK_CFG=quick GPU=1 ~/cfd_gpu/lab_openlb.sh winddirs; \
  RANK_CFG=quick RANK_SEEDS="1000 2000" GPU=1 ~/cfd_gpu/lab_openlb.sh rank; read -p done'

# CPU queue (second terminal): pathway split (minutes), HRR cube at dx 4 (~1 h), dx 2 overnight
tmux new -s cpu '~/cfd_gpu/lab_openlb.sh pathways; \
  MODEL=hrr GATES="gate6a gate6b_dx4" ~/cfd_gpu/lab_openlb.sh gates; \
  MODEL=hrr GATES="gate6b_dx2" ~/cfd_gpu/lab_openlb.sh gates; read -p done'

# next morning
~/cfd_gpu/lab_openlb.sh send
```
The ranking runs use the current BGK model (MODEL default) because HRR is not validated yet;
the CPU queue's cube runs say whether HRR fixes the wake. Detach from tmux with Ctrl-b d,
reattach with `tmux attach -t gpu`. Progress: `ls ~/olb_lab/gpu/done | grep -c rank_quick`.

## 10. GPU speed-up, float build, plume and MUST (2026-10-29)

What changed and why: `OPENLB_PHASE8_GPU.md` §6. The lab repo has to move to the branch that
now carries everything (it holds the OpenLB pipeline merged with the forward_city work):

```bash
cd ~/cfd_gpu && git fetch origin && git checkout claude/elegant-goodall-x5243h && git pull
```

Then, in this order (each line resumable; `send` at any point collects what has run):

```bash
L=~/cfd_gpu/lab_openlb.sh
$L setup                                   # rebuilds the CPU app (OpenLB core kept), ~5 min
$L parity_cpu                              # CPU operators vs host loops: must be bitwise (~10 min)
GPU=1 $L setup                             # rebuilds the GPU app (double)
GPU=1 $L parity                            # GPU vs CPU: must say [parity] ALL PASS (~15 min)
GPU=1 $L speed                             # double: MLUPS, burst ms/step, stage split (~10 min)
GPU=1 PRECISION=single $L setup            # float tree under ~/olb_lab/gpu_sp, ~20 min (OpenLB core in float)
GPU=1 PRECISION=single $L speed            # float: the same numbers
GPU=1 PRECISION=single $L gates            # 5, 6a, 6b dx4/dx2, 7a/b/c, plume, in float
GPU=1 $L precision                         # float vs double: 3 designs x (2 double seeds + 1 float), dx 8: a few hours
GPU=1 $L plume                             # the plume test on the double build too (minutes)
$L send
```

- **A compile error** in either GPU build shows in `send` under `build_app` (the code was not
  compiled in the cloud container; this is its first compile). Send it as is.
- **parity_cpu / parity** first: the WALE stencil is now chosen at start-up rather than every
  step (must be bitwise on CPU), and WALE's powers go through sqrt (round-off; parity
  tolerances unchanged).
- **speed** prints `spin-up: … MLUPS` and `burst: … ms/step` and, per stage, ms/step and share.
  The double build is expected near the ~100 MLUPS of October; the float build several times
  that if the step was FP64-bound as analysed. The stage split says what to optimise next.
- **precision** prints a table per run (iF, budget closure, stop, ms/step) and the verdict:
  each design's float−double difference against the double build's own seed-to-seed noise,
  the mean offset, the ranking, the wall-time speed-up. Use the float build for production
  only if it says `PRECISION PASS` *and* the float gates match the double ones (6a/6b within
  their seed spread, 7a/b/c and plume pass).
- **plume** (also inside `gates`): a one-cell puff in a frozen 4 m/s wind against the exact
  point-source solution. `mass` is 1.000 if transport conserves mass; `sigma_y/z` within 5 %
  says the lattice applies the diffusivity it is told; `centre` is worst next to the source.
  The production-D case uses wider bands (cell Péclet 13).
- **FRESH=1** reruns gates already marked done, e.g. `FRESH=1 GPU=1 $L gates` after a code
  change; the long city and ranking steps keep their markers (delete those by hand).
- **MODEL=hrr** works with all of the above (`MODEL=hrr GPU=1 PRECISION=single $L setup`, …).

**MUST** (Mock Urban Setting Test, the COST 732 container-array case) is ready to run once the
data is on the box: `MUST_DIR=~/must GPU=1 $L must`. The folder needs `containers.txt` (one
container per line: `xc yc length width height angle_deg`, metres), `must.env` (the case:
`OBST_ROT` = minus the direction the wind blows toward in the data's frame, `DX`, the inlet
`U_INLET ABL_ZREF ABL_Z0 ABL_LTURB`, the source `SRC_X SRC_Y SRC_Z`, and the normalisations
`MUST_UREF_POINT MUST_UREF_OBS MUST_CREF_U`) and `obs_flow.csv` (`x,y,z,u,v[,w]`) and/or
`obs_conc.csv` (`x,y,z,c` as C* = cUH²/Q). Scored by `tests/must_score.py`: velocity hit rate
(q ≥ 0.66, COST 732 / VDI 3783-9), concentration FB, NMSE, FAC2, NAD (Hanna & Chang 2012 urban
acceptance). DX 0.5 m gives 5 cells per container height (~10 M cells, fine on the A4000 in
float); 0.25 m (10 cells, the guideline) needs a large GPU.
