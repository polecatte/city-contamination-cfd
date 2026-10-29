#!/usr/bin/env bash
# lab_openlb.sh — run the OpenLB port's gates and production city on the lab box.
#
#   ./lab_openlb.sh setup     # fetch + build OpenLB 1.8.1 (OpenMP), build the app and tools
#   ./lab_openlb.sh gates     # 5, 6a, 6b (dx=4 and dx=2), 7a/7b/7c  — ~3-4 h on 32 cores
#   ./lab_openlb.sh city      # production city, two inlet seeds, to 99 % clearance
#   ./lab_openlb.sh package   # summary.txt + tarball of logs and small outputs
#   ./lab_openlb.sh all       # the four above, in order
#   ./lab_openlb.sh status    # what has run, what passed
#
#   GPU=1 ./lab_openlb.sh setup|gates|city|all|status   # the same, on the GPU build
#   GPU=1 ./lab_openlb.sh parity    # GPU vs CPU build on short cases (tests/device_parity.sh)
#   ./lab_openlb.sh compare         # CPU / GPU / HRR gate numbers side by side
#   ./lab_openlb.sh send            # everything to send back: one small text file + a logs tarball
#   [CONFIRM=1] ./lab_openlb.sh clean   # disk full? shows what fills it; removes only regenerable files
#   [GPU=1] ./lab_openlb.sh frames     # production city (seed 1000) again with snapshots -> GIFs + stills
#   [GPU=1] ./lab_openlb.sh showcase   # small city end to end: flow, burst, J, figures, scorecard
#   [GPU=1] ./lab_openlb.sh cube_sens  # 6b sensitivity at dx=4: floor model, inflow turbulence, C_w
#   GPU=1   ./lab_openlb.sh cube_dx1   # 6b at dx=1 m (H/dx=40, 89 M cells): needs >= 60 GB GPU (H100)
#
#   [MODEL=hrr] [GPU=1] ./lab_openlb.sh rank   # ranking study: 5 designs x 2 seeds (see do_rank)
#   ./lab_openlb.sh rank_report                 # its table: noise, spread, ANOVA, Spearman old vs new
#   RANK_CFG=quick [MODEL=..] GPU=1 ./lab_openlb.sh rank      # quick ranking: dx 8, guideline domain, tail stop, 1 seed
#   RANK_CFG=quick [MODEL=..] GPU=1 ./lab_openlb.sh winddirs  # one design at WIND_PHIS (default 0 22.5 45 deg)
#   ./lab_openlb.sh pathways                    # dose split indoor / park / street, from the finished city runs
#   GATES="gate6a gate6b_dx4" ./lab_openlb.sh gates          # only the gates named
#   ./lab_openlb.sh opt_setup                   # optimizer venv (CPU torch + BoTorch) + smoke test
#   [MODEL=hrr] GPU=1 ./lab_openlb.sh optimize | opt_report | opt_verify   # the optimisation (OPTIMIZER.md)
#   MODEL=hrr [GPU=1] ./lab_openlb.sh setup|gates|...  # the same with HRR collision + corrected
#             WALE (COLLISION_MODEL=3) at TAU=0.5001: its own app dir (urban_hrr) and outputs
#             under $WORK/hrr (GPU: $WORK/gpu/hrr); geometries shared with the default model
#
# Resumable: every step writes $WORK/done/<step> when it finishes and is skipped next time
# (delete the marker to rerun one). Run it under tmux or nohup — it takes hours:
#   tmux new -s olb './lab_openlb.sh all 2>&1 | tee -a ~/olb_lab/lab.log; read -p "[done - Enter to close]"'
#
# Env:
#   WORK      working directory (default ~/olb_lab): OpenLB tree, geometries, outputs, logs
#   OLB_ROOT  an existing OpenLB 1.8.1 tree to use instead of downloading one
#   THREADS   OpenMP threads (default: nproc)
#   MODEL     wale (default: BGK + WALE, tau 0.505) or hrr (HRR + corrected WALE, tau 0.5001)
#   GPU=1     use the CUDA build: its own OpenLB tree and outputs under $WORK/gpu, the
#             geometries shared with the CPU runs. Needs an NVIDIA driver; nvcc >= 12.4 is
#             taken from PATH or $NVCC, else CUDA 12.6 is installed from conda-forge into
#             $WORK/cuda126 (no root). CUDA_ARCH defaults to nvidia-smi's compute capability.
#
# Every per-step operator (inlet, WALE gradient, rough wall, top stress, time mean, Step-4
# flux/inject/deposit/theta/settle) exists twice: host loops (the CPU reference, default on
# CPU) and OpenLB operators in urban_ops.h (default on GPU, HOST_OPS=0 on CPU). The two agree
# bitwise on CPU (tests/device_parity.sh STRICT=1). Not MPI: one cuboid per process.
# Results and what each gate means: OPENLB_PHASE5_6_GATES.md.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${WORK:-$HOME/olb_lab}"
THREADS="${THREADS:-$(nproc)}"
OLB_URL_GITLAB="https://gitlab.com/openlb/release/-/archive/1.8.1/release-1.8.1.tar.gz"
OLB_URL_ZENODO="https://zenodo.org/records/15440776/files/release-1.8.1.tar.gz?download=1"
export OMP_NUM_THREADS="$THREADS"

WORK_CPU="$WORK"
# Compiler temporaries: nvcc and g++ write several GB of intermediates to $TMPDIR (default /tmp).
# On a shared machine /tmp can be small or full ("No space left on device" mid-build), so
# everything here uses a directory under the work dir, unless LAB_TMPDIR names another.
export TMPDIR="${LAB_TMPDIR:-$WORK_CPU/tmp}"
mkdir -p "$TMPDIR"
CPU_BIN="$WORK_CPU/release-1.8.1/examples/urban/urban_flow/urban_flow"
if [ "${GPU:-0}" = 1 ]; then
  WORK="$WORK_CPU/gpu"
  OLB_ROOT="${OLB_ROOT_GPU:-$WORK/release-1.8.1}"
else
  OLB_ROOT="${OLB_ROOT:-$WORK/release-1.8.1}"
fi
APP="$OLB_ROOT/examples/urban/urban_flow"
MODEL="${MODEL:-wale}"
case "$MODEL" in
  wale) ;;
  hrr)  # HRR (Jacob et al. 2018) + corrected WALE: near-inviscid lattice, tau 0.5001
        WORK="$WORK/hrr"; APP="$OLB_ROOT/examples/urban/urban_hrr"
        CPU_BIN="$WORK_CPU/release-1.8.1/examples/urban/urban_hrr/urban_flow"
        export TAU="${TAU:-0.5001}" ;;
  *) echo "MODEL must be wale or hrr"; exit 2 ;;
esac
mkdir -p "$WORK/done" "$WORK/logs"
BIN="$APP/urban_flow"
SUMMARY="$WORK/summary.txt"

say()  { printf '[lab %s] %s\n' "$(date +%H:%M:%S)" "$*"; }
done_() { [ -f "$WORK/done/$1" ]; }
mark() { date > "$WORK/done/$1"; }
note() { printf '%s\n' "$*" >> "$SUMMARY"; }

# step NAME CMD... : run once, log to logs/NAME.log, mark done on success
step() {
  local name="$1"; shift
  if done_ "$name"; then say "skip $name (done)"; return 0; fi
  say "run  $name  (log: $WORK/logs/$name.log)"
  if ( cd "$WORK" && "$@" ) > "$WORK/logs/$name.log" 2>&1; then
    mark "$name"; say "ok   $name"
  else
    say "FAIL $name — last lines of $WORK/logs/$name.log:"
    tail -n 25 "$WORK/logs/$name.log" | sed 's/^/     /'; return 1
  fi
}

# build NAME CMD... : compile, and on failure show the first errors instead of just a path
build() {
  local name="$1"; shift
  say "build $name  (log: $WORK/logs/build_$name.log)"
  if ! "$@" > "$WORK/logs/build_$name.log" 2>&1; then
    say "FAIL build $name. First errors:"
    { grep -m 20 -E 'error|Error|fatal|undefined reference|cannot find' "$WORK/logs/build_$name.log" || tail -n 30 "$WORK/logs/build_$name.log"; } | sed 's/^/     /'
    say "compiler: $(g++ --version | head -1)${NVCC_USED:+; nvcc: $("$NVCC_USED" --version | tail -2 | head -1)}"
    exit 1
  fi
}

# gate NAME CMD... : like step, but a non-zero exit is a recorded gate FAIL, not an abort
gate() {
  local name="$1"; shift
  # GATES="gate6a gate6b_dx4" runs only the gates named
  if [ -n "${GATES:-}" ] && [[ " $GATES " != *" $name "* ]]; then return 0; fi
  if done_ "$name"; then say "skip $name (done)"; return 0; fi
  say "gate $name"
  local rc=0
  ( cd "$WORK" && "$@" ) > "$WORK/logs/$name.log" 2>&1 || rc=$?
  if [ $rc -eq 0 ]; then note "PASS $name"; else note "FAIL $name (exit $rc)"; fi
  grep -E '^\[(6a|6b|7a|7c|8|J)\]|GATE|PASS|FAIL' "$WORK/logs/$name.log" | tail -8 | sed 's/^/     /' || true
  mark "$name"
}

# ─────────────────────────────────────────────────────────────────────────────
# nvcc for the GPU build. OpenLB 1.8 needs CUDA >= 12.4 or so: 12.0 rejects its consteval
# std::source_location (src/core/fields.h). Order: $NVCC, nvcc on PATH if new enough, a
# conda-forge CUDA 12.6 in $WORK_CPU/cuda126 (installed here with micromamba if missing).
cuda_toolchain() {
  local smi
  if ! command -v nvidia-smi >/dev/null; then
    say "WARNING: nvidia-smi not found — the GPU build will compile (with CUDA_ARCH set) but cannot run here"
  elif ! smi=$(nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total,memory.used --format=csv,noheader 2>&1) \
       || [[ "$smi" == *Failed* || "$smi" == *mismatch* ]]; then
    say "nvidia-smi fails: $(printf '%s' "$smi" | head -2 | tr '\n' ' ')"
    if [[ "$smi" == *mismatch* ]]; then
      say "  'Driver/library version mismatch' = the NVIDIA driver was updated but the old kernel module is"
      say "  still loaded. Reboot the machine (or unload/reload the nvidia kernel modules). No CUDA program,"
      say "  this one included, can run until then."
    fi
    [ -n "${CUDA_ARCH:-}" ] || { say "stopping (set CUDA_ARCH=86 / 90 to compile anyway)"; exit 1; }
    say "CUDA_ARCH=$CUDA_ARCH given: compiling anyway"
  else
    printf '%s\n' "$smi" | sed 's/^/     GPU: /'
  fi
  local v
  if [ -n "${NVCC:-}" ]; then NVCC_USED="$NVCC"
  elif command -v nvcc >/dev/null && v=$(nvcc --version | sed -n 's/.*release \([0-9]*\)\.\([0-9]*\).*/\1 \2/p') \
       && [ "${v% *}" -gt 12 -o \( "${v% *}" -eq 12 -a "${v#* }" -ge 4 \) ]; then NVCC_USED="$(command -v nvcc)"
  else
    local C="$WORK_CPU/cuda126"
    if [ ! -x "$C/bin/nvcc" ]; then
      say "no nvcc >= 12.4 (found: $(command -v nvcc >/dev/null && nvcc --version | tail -2 | head -1 || echo none)); installing CUDA 12.6 from conda-forge into $C"
      [ -x "$WORK_CPU/micromamba" ] || curl -fL -o "$WORK_CPU/micromamba" \
        https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-linux-64
      chmod +x "$WORK_CPU/micromamba"
      MAMBA_ROOT_PREFIX="$WORK_CPU/mamba" "$WORK_CPU/micromamba" create -y -q -p "$C" -c conda-forge \
        "cuda-nvcc=12.6" "cuda-cudart-dev=12.6" "cuda-driver-dev=12.6" > "$WORK/logs/cuda_install.log" 2>&1 \
        || { say "FAIL installing CUDA (log: $WORK/logs/cuda_install.log)"; exit 1; }
    fi
    NVCC_USED="$C/bin/nvcc"; CUDA_LIBDIR="$C/lib"
  fi
  # CUDA 12.x accepts gcc <= 13 as host compiler
  local gv; gv=$(g++ -dumpversion | cut -d. -f1)
  if [ -z "${CUDA_HOST_CXX:-}" ] && [ "$gv" -gt 13 ]; then
    for c in g++-13 g++-12; do command -v $c >/dev/null && { CUDA_HOST_CXX=$c; break; }; done
    [ -n "${CUDA_HOST_CXX:-}" ] || say "WARNING: g++ $gv is newer than CUDA 12 supports and no g++-13/12 found"
  fi
  say "nvcc: $NVCC_USED ($("$NVCC_USED" --version | tail -2 | head -1))${CUDA_HOST_CXX:+, host compiler $CUDA_HOST_CXX}"
}

# GPU build vs CPU build on four short cases that execute every operator (see the script).
do_parity() {
  [ "${GPU:-0}" = 1 ] || { echo "parity compares the GPU build against the CPU one: run it as GPU=1 $0 parity"; exit 2; }
  [ -x "$BIN" ] || { echo "run 'GPU=1 $0 setup' first"; exit 1; }
  [ -x "$CPU_BIN" ] || { echo "no CPU build at $CPU_BIN: run '$0 setup' first"; exit 1; }
  gate parity env REF="$CPU_BIN" TEST="$BIN" "$REPO/tests/device_parity.sh" "$WORK_CPU" "$WORK/parity"
  grep -E '^\[parity\]|rel.diff|worst' "$WORK/logs/parity.log" | sed 's/^/     /'
}

# A small city through the whole pipeline (geometry -> live flow -> burst to 99 % clearance ->
# Theta, deposition, J -> figures), with a PASS/FAIL scorecard (tests/showcase_report.py).
# 400 m of city; the domain is still ~1.8 km long, since the 15 H wake buffer is fixed.
do_showcase() {
  [ -x "$BIN" ] || { echo "run '$0 setup' first"; exit 1; }
  step geom_showcase sh -c "DOMAIN=compact CITY_M=${SHOW_CITY_M:-400} POP=${SHOW_POP:-8000} OUT_DIR=geom_showcase ./gen_openlb_geom"
  step showcase_run sh -c "STEP4=1 GEOM_DIR=geom_showcase OUT_DIR=showcase CHECK_EVERY=2000 TS_EVERY=500 '$BIN' > showcase.log 2>&1"
  gate showcase python3 "$REPO/tests/showcase_report.py" geom_showcase showcase showcase.log
  step showcase_viz python3 "$REPO/visualize_forward.py" showcase
  say "scorecard: $WORK/showcase/SHOWCASE.txt   figures: $WORK/showcase/figs/"
  cat "$WORK/showcase/SHOWCASE.txt"
}

# The production city once more with street-level snapshots every FRAME_EVERY burst steps
# (urban_flow FRAMES=1), then visualize_forward.py: airflow.gif, concentration.gif, three-time
# ground maps, deposition maps, the 3-D cloud at half clearance. ~130 MB of frames at 500.
do_frames() {
  [ -x "$BIN" ] || { echo "run '$0 setup' first"; exit 1; }
  python3 -c 'import matplotlib' 2>/dev/null || { echo "needs matplotlib: pip install --user matplotlib"; exit 1; }
  step city_frames sh -c "STEP4=1 ABL_SEED=1000 FRAMES=1 TS_EVERY=${FRAME_EVERY:-500} GEOM_DIR=geom_prod OUT_DIR=city_frames CHECK_EVERY=2000 '$BIN' > city_frames.log 2>&1"
  step frames_viz python3 "$REPO/visualize_forward.py" city_frames
  say "figures: $WORK/city_frames/figs/"; ls -1 "$WORK/city_frames/figs/" | sed 's/^/     /'
}

# Gate 6b sensitivity on the dx=4 cube (each run = gate6b_dx4's cost): which knob moves Xr/H?
#   gm0   plain bounce-back floor (top stress kept): the rough-wall model inside the bubble
#   ti12  inflow turbulence x1.2 (I_u at roof height measured ~8 % under the neutral target)
#   cw20  WALE constant 0.20 instead of 0.325 (less subgrid dissipation)
do_cube_sens() {
  [ -x "$BIN" ] || { echo "run '$0 setup' first"; exit 1; }
  local P="$REPO/tests" name envs
  for v in "gm0|GROUND_MODEL=0 TOP_STRESS=1" "ti12|ABL_TI_SCALE=1.2" "cw20|LES_CONST=0.20"; do
    name="${v%%|*}"; envs="${v#*|}"
    gate "cube_$name" sh -c "env $envs GEOM_DIR=geom_cube OUT_DIR=out_6b_$name SPINUP_FT=4 AVG_FT=2 CHECK_EVERY=2000 '$BIN' > out_6b_$name.log 2>&1;
      python3 '$P/gate6_analyze.py' cube geom_cube out_6b_$name"
  done
  grep -h 'Xr/H =' "$WORK"/logs/gate6b_dx4.log "$WORK"/logs/cube_*.log 2>/dev/null | sed 's/^/     /'
}

# Gate 6b at dx = 1 m (H/dx = 40): the third point of the grid study (dx 4 m: 2.63, dx 2 m: 2.31).
# 840x440x241 = 89 M cells: ~40 GB of device memory in double, and the same again in host RAM.
do_cube_dx1() {
  [ "${GPU:-0}" = 1 ] || { echo "cube_dx1 is a GPU run: GPU=1 $0 cube_dx1"; exit 2; }
  [ -x "$BIN" ] || { echo "run 'GPU=1 $0 setup' first"; exit 1; }
  local mem; mem=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 0)
  [ "${mem:-0}" -ge 60000 ] || { echo "cube_dx1 needs >= 60 GB of GPU memory (this GPU: ${mem:-?} MiB)"; exit 1; }
  step geom_cube_dx1 sh -c "CASE=cube DX=1 CUBE_H=40 OUT_DIR=geom_cube_dx1 ./gen_gate6_geom"
  gate gate6b_dx1 sh -c "GEOM_DIR=geom_cube_dx1 OUT_DIR=out_6b_dx1 SPINUP_FT=4 AVG_FT=2 CHECK_EVERY=10000 '$BIN' > out_6b_dx1.log 2>&1;
    python3 '$REPO/tests/gate6_analyze.py' cube geom_cube_dx1 out_6b_dx1"
}

# Ranking study: does a spread of designs give a spread of exposure beyond the inlet-seed noise,
# and does the old model rank the designs the way the corrected one does? Five contrasting
# designs from gen_openlb_geom (builder rev 9: same floor area, <= 100 m), each with
# RANK_SEEDS inlet seeds (common to all designs). Configuration from MODEL:
#   wale: the old model: BGK+WALE tau 0.505, compact domain, legacy release set, constant D,
#         dx 4 (7.3 M cells: fits the A4000)
#   hrr:  the corrected model: HRR + corrected WALE tau 0.5001, COST 732 domain, release over
#         city + environs ring, per-cell D (D_LOCAL=1); dx RANK_DX, default 4 on a >= 40 GB
#         GPU (32.5 M cells, H100), else 8 (4.1 M cells)
# Every run appends "config design seed geom out" to $WORK/rank_manifest.txt; the report
# (tests/rank_report.py) reads every manifest under $WORK_CPU, so both configurations land in
# one table with Spearman/Kendall between them.
RANK_DESIGNS=(
  "base|"
  "core|HEIGHT_CONC=1 EMP_CENTERS=1 EMP_CONC=0.9"
  "flat|HEIGHT_CONC=0 ROUGH=0 STREET_W=28 PARK_FRAC=0.25"
  "fine|BLOCK_W=32 BLOCK_D=32 STREET_W=12"
  "coarse|BLOCK_W=100 BLOCK_D=100 STREET_W=30"
)
do_rank() {
  [ -x "$BIN" ] || { echo "run '$0 setup' first"; exit 1; }
  local dx="${RANK_DX:-}" cfg genv runv gen n e s mem seeds="${RANK_SEEDS:-1000 2000}"
  if [ "${RANK_CFG:-}" = quick ]; then
    # quick: the production setup at dx 8 (COST 732 domain, release over city + environs,
    # per-cell D, tail stop), physics from MODEL, one seed unless RANK_SEEDS says otherwise
    dx="${dx:-8}"; cfg="quick_${MODEL}_dx$dx"; genv=""; runv="D_LOCAL=1 TAIL=1"; seeds="${RANK_SEEDS:-1000}"
  elif [ "$MODEL" = hrr ]; then
    if [ -z "$dx" ]; then
      mem=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 0)
      dx=8; [ "${GPU:-0}" = 1 ] && [ "${mem:-0}" -ge 40000 ] && dx=4
    fi
    cfg="new_dx$dx"; genv=""; runv="D_LOCAL=1"
  else
    dx="${dx:-4}"; cfg="old_dx$dx"; genv="DOMAIN=compact"; runv=""
  fi
  case "$dx" in 4) gen=./gen_openlb_geom ;; 8) gen=./gen_openlb_geom8 ;; *) echo "RANK_DX must be 4 or 8"; exit 2 ;; esac
  say "ranking study: configuration $cfg ($MODEL), designs: ${RANK_DESIGNS[*]%%|*}"
  for v in "${RANK_DESIGNS[@]}"; do
    n="${v%%|*}"; e="${v#*|}"
    step "rank_geom_${cfg}_$n" sh -c "env $genv $e OUT_DIR=rank_geom_${cfg}_$n $gen"
    for s in $seeds; do
      need_space "$WORK" 1 "a ranking run"
      step "rank_${cfg}_${n}_s$s" sh -c "env $runv STEP4=1 ABL_SEED=$s GEOM_DIR=rank_geom_${cfg}_$n OUT_DIR=rank_${cfg}_${n}_s$s CHECK_EVERY=2000 TS_EVERY=200 '$BIN' > rank_${cfg}_${n}_s$s.log 2>&1"
      # the report needs Theta + meta only; the flow snapshots are ~0.1-0.4 GB per run
      rm -f "$WORK/rank_${cfg}_${n}_s$s/umean_full.f32" "$WORK/rank_${cfg}_${n}_s$s/uavg.f32"
      local line="$cfg $n $s $WORK/rank_geom_${cfg}_$n $WORK/rank_${cfg}_${n}_s$s"
      grep -qxF "$line" "$WORK/rank_manifest.txt" 2>/dev/null || echo "$line" >> "$WORK/rank_manifest.txt"
    done
  done
  do_rank_report
}
# Wind directions for one design (default base) at the quick-ranking settings: WIND_PHIS lists
# directions in degrees (toward, counter-clockwise from the city's +x; tools/wind_rose.py maps each
# to ORIENT + inflow angle). 0 reuses the quick ranking's run of the same design if it exists.
do_winddirs() {
  [ -x "$BIN" ] || { echo "run '$0 setup' first"; exit 1; }
  local dx="${RANK_DX:-8}" design="${WIND_DESIGN:-base}" s="${WIND_SEED:-1000}" e="" v k th n gen cfg rcfg line
  case "$dx" in 4) gen=./gen_openlb_geom ;; 8) gen=./gen_openlb_geom8 ;; *) echo "RANK_DX must be 4 or 8"; exit 2 ;; esac
  cfg="wind_${MODEL}_dx$dx"; rcfg="quick_${MODEL}_dx$dx"
  for v in "${RANK_DESIGNS[@]}"; do [ "${v%%|*}" = "$design" ] && e="${v#*|}"; done
  say "wind directions for design $design: ${WIND_PHIS:-0 22.5 45} deg ($MODEL, dx $dx)"
  for phi in ${WIND_PHIS:-0 22.5 45}; do
    read -r k th < <(python3 "$REPO/tools/wind_rose.py" --case "$phi")
    n="phi$phi"
    if [ "$k" = 0 ] && [ "$th" = 0 ] && [ -f "$WORK/rank_${rcfg}_${design}_s$s/theta.f32" ]; then
      line="$cfg $n $s $WORK/rank_geom_${rcfg}_$design $WORK/rank_${rcfg}_${design}_s$s"
      say "phi 0: reusing the quick ranking's run of $design"
    else
      step "wind_geom_${cfg}_$n" sh -c "env $e ORIENT=$k WIND_DEG=$th OUT_DIR=wind_geom_${cfg}_$n $gen"
      need_space "$WORK" 1 "a wind-direction run"
      step "wind_${cfg}_$n" sh -c "env D_LOCAL=1 TAIL=1 STEP4=1 ABL_SEED=$s GEOM_DIR=wind_geom_${cfg}_$n OUT_DIR=wind_${cfg}_${n} CHECK_EVERY=2000 TS_EVERY=200 '$BIN' > wind_${cfg}_${n}.log 2>&1"
      rm -f "$WORK/wind_${cfg}_${n}/umean_full.f32" "$WORK/wind_${cfg}_${n}/uavg.f32"
      line="$cfg $n $s $WORK/wind_geom_${cfg}_$n $WORK/wind_${cfg}_${n}"
    fi
    grep -qxF "$line" "$WORK/wind_manifest.txt" 2>/dev/null || echo "$line" >> "$WORK/wind_manifest.txt"
  done
  python3 "$REPO/tests/rank_report.py" "$WORK/wind_manifest.txt" | tee "$WORK_CPU/wind_report.txt"
}

# Who breathes the dose: re-score the finished production-city runs (geom_prod) by pathway with
# the corrected receptor weights (tests/pathways.py). No CFD: the geometry is re-generated with
# the legacy builder (byte-identical, checked) plus the per-pathway weights.
do_pathways() {
  local g="$WORK_CPU/geom_prod" rx="$WORK_CPU/geom_prod_rx" gen="$TMPDIR/gen_rx" runs=() d
  [ -f "$g/material_map.dat" ] || { echo "no $g (run '$0 setup' first)"; exit 1; }
  g++ -O2 -std=c++17 -I"$REPO" -DCELL_SIZE_M=4.0 "$REPO/gen_openlb_geom.cpp" -o "$gen" || exit 1
  ( cd "$WORK_CPU" && BUILDER=legacy DOMAIN=compact RECEPTOR_PARTS=1 OUT_DIR="$rx" "$gen" > "$WORK_CPU/logs/pathways_geom.log" 2>&1 )
  cmp -s "$rx/material_map.dat" "$g/material_map.dat" \
    || { say "the re-generated geometry differs from $g: not re-scoring (see logs/pathways_geom.log)"; exit 1; }
  rm -f "$rx"/material_map.dat "$rx"/dep_vel.f32 "$rx"/geom_type.u8 "$rx"/source_mask.u8 "$rx"/release_zone.u8
  for d in city_s1000 city_s2000 gpu/city_s1000 gpu/city_s2000 gpu/city_frames; do
    [ -f "$WORK_CPU/$d/theta.f32" ] && runs+=("$WORK_CPU/$d"); done
  [ ${#runs[@]} -gt 0 ] || { echo "no finished city runs with theta.f32"; exit 1; }
  python3 "$REPO/tests/pathways.py" "$rx" "${runs[@]}" --legacy "$g" | tee "$WORK_CPU/pathways.txt"
}

do_rank_report() {
  local m; m=$(find "$WORK_CPU" -maxdepth 3 -name rank_manifest.txt | sort)
  [ -n "$m" ] || { echo "no rank_manifest.txt under $WORK_CPU yet"; exit 1; }
  # shellcheck disable=SC2086
  python3 "$REPO/tests/rank_report.py" $m | tee "$WORK_CPU/rank_report.txt"
}

# Optimizer environment: a venv at $WORK_CPU/optenv with CPU torch + BoTorch (requirements-opt.txt),
# then the wind-surrogate smoke test. BoTorch 0.18 needs Python >= 3.11.
do_opt_setup() {
  local E="$WORK_CPU/optenv"
  python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' \
    || { echo "BoTorch 0.18 needs Python >= 3.11 (found $(python3 --version)); install a newer python3 or use conda"; exit 1; }
  [ -x "$E/bin/python" ] || python3 -m venv "$E"
  step opt_torch sh -c "'$E/bin/pip' install -q torch --index-url https://download.pytorch.org/whl/cpu || '$E/bin/pip' install -q torch"
  step opt_reqs  "$E/bin/pip" install -q -r "$REPO/requirements-opt.txt"
  "$E/bin/python" -W ignore "$REPO/tests/test_wind_surrogate.py"
  "$E/bin/python" -W ignore "$REPO/tests/test_driver_synth.py" "$WORK_CPU/opt_synth" | grep '^\[synth\]' 
  say "optimizer env ready: $E/bin/python"
}

# Optimisation (opt/driver.py, OPTIMIZER.md): BoTorch over the 11 design knobs, one run per
# (design, wind direction), resumable. MODEL decides the physics (hrr = production: tau 0.5001,
# per-cell D); OPT_DX the resolution (default 4 on a >= 40 GB GPU, else 8); OPT_BUDGET the runs.
#   [MODEL=hrr] GPU=1 ./lab_openlb.sh optimize      # run (or continue) to OPT_BUDGET runs
#   ./lab_openlb.sh opt_report                      # best designs, surrogate F +- sd, directions
#   [MODEL=hrr] GPU=1 ./lab_openlb.sh opt_verify    # top OPT_TOP designs at all 16 directions
opt_common() {
  [ -x "$BIN" ] || { echo "run '$0 setup' first"; exit 1; }
  [ -x "$WORK_CPU/optenv/bin/python" ] || { echo "run '$0 opt_setup' first"; exit 1; }
  local dx="${OPT_DX:-}" mem
  if [ -z "$dx" ]; then
    mem=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 0)
    dx=8; [ "${GPU:-0}" = 1 ] && [ "${mem:-0}" -ge 40000 ] && dx=4
  fi
  case "$dx" in 4) OPT_GEN="$WORK/gen_openlb_geom" ;; 8) OPT_GEN="$WORK/gen_openlb_geom8" ;; *) echo "OPT_DX must be 4 or 8"; exit 2 ;; esac
  OPT_DIR="$WORK/opt_${MODEL}_dx$dx"
  OPT_RUNENV=""; [ "$MODEL" = hrr ] && OPT_RUNENV="TAU=${TAU:-0.5001} D_LOCAL=1"
  say "optimiser: $OPT_DIR (model $MODEL, dx $dx)"
}
do_optimize() {
  opt_common
  ( cd "$REPO" && "$WORK_CPU/optenv/bin/python" -W ignore -m opt.driver run --workdir "$OPT_DIR" --gen "$OPT_GEN" --solver "$BIN" \
      --budget "${OPT_BUDGET:-150}" --n-init "${OPT_INIT:-24}" --run-env "$OPT_RUNENV" ) 2>&1 | tee -a "$OPT_DIR.log"
}
do_opt_report() { opt_common; ( cd "$REPO" && "$WORK_CPU/optenv/bin/python" -W ignore -m opt.driver report --workdir "$OPT_DIR" ); }
do_opt_verify() {
  opt_common
  ( cd "$REPO" && "$WORK_CPU/optenv/bin/python" -W ignore -m opt.driver verify --workdir "$OPT_DIR" --gen "$OPT_GEN" --solver "$BIN" \
      --top "${OPT_TOP:-3}" --run-env "$OPT_RUNENV" ) 2>&1 | tee -a "$OPT_DIR.log"
}

# send: everything worth sending back, in one small text file to paste into the chat (and a
# small tarball of the logs, no fields, to upload if the text is not enough). Covers the CPU, GPU,
# HRR and GPU-HRR work dirs, whatever has run: setup/build errors, gate and parity verdicts,
# throughput, ranking and optimiser reports, and the last lines of every failed step.
do_send() {
  local ts out tarf d lab f; ts="$(date +%Y%m%d-%H%M%S)"
  out="$WORK_CPU/send_$ts.txt"; tarf="$WORK_CPU/send_$ts.tar.gz"
  # collecting must not stop at a missing piece: no -e / pipefail inside (grep finding nothing is fine)
  ( set +e +o pipefail
  {
    echo "== lab results $ts on $(hostname); repo commit $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null)"
    command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>&1 | head -2 | sed 's/^/GPU: /'
    df -Ph /tmp "$TMPDIR" "$WORK_CPU" 2>/dev/null | awk 'NR>1{print "disk: " $6 " " $4 " free of " $2}' | sort -u
    for d in "$WORK_CPU" "$WORK_CPU/gpu" "$WORK_CPU/hrr" "$WORK_CPU/gpu/hrr"; do
      [ -d "$d/logs" ] || continue
      lab="${d#$WORK_CPU}"; lab="${lab#/}"; lab="${lab:-cpu}"
      echo; echo "======== $lab ($d)"
      echo "done: $(ls -1 "$d/done" 2>/dev/null | tr '\n' ' ')"
      [ -f "$d/summary.txt" ] && sed 's/^/summary: /' "$d/summary.txt"
      for f in "$d"/logs/build_*.log; do
        [ -f "$f" ] && grep -q -E 'error:|fatal|Error [0-9]+|undefined reference' "$f" && \
          { echo "-- $(basename "$f") (errors)"; grep -m 8 -E 'error:|fatal|Error [0-9]+|undefined reference' "$f"; }
      done
      for f in "$d"/logs/*.log; do
        [ -f "$f" ] || continue
        case "$(basename "$f")" in build_*|olb_core_*|olb_external*|olb_config*|cuda_install*) continue ;; esac
        lines=$(grep -E '^\[(6a|6b|7a|7b|7c|8|J|wake|parity|rank|tail|orient|synth|wind|opt|verify)\]|GATE|PASS|FAIL|Xr/H|drift|closure|DIVERG|error:|fatal' "$f" | tail -8)
        [ -n "$lines" ] && { echo "-- $(basename "$f" .log)"; echo "$lines"; }
      done
      # a step that failed: its log's last lines (the step's own FAIL report)
      grep -l -E 'FAIL|DIVERG|error:|fatal|Error [0-9]+' "$d"/logs/*.log 2>/dev/null | while read -r f; do
        case "$(basename "$f")" in build_*) continue ;; esac
        echo "-- tail of $(basename "$f")"; tail -n 12 "$f"; done
      grep -h -E 'MLUPs :' "$d"/out_*.log "$d"/city_*.log "$d"/rank_*.log 2>/dev/null | sed 's/^/throughput /' | head -12
    done
    for f in rank_report.txt wind_report.txt pathways.txt; do
      [ -f "$WORK_CPU/$f" ] && { echo; echo "======== $f"; cat "$WORK_CPU/$f"; }
    done
    for f in "$WORK_CPU"/opt_*/report.json "$WORK_CPU"/*/opt_*/report.json "$WORK_CPU"/opt_*/verified.json "$WORK_CPU"/*/opt_*/verified.json; do
      [ -f "$f" ] && { echo; echo "======== $f"; head -c 4000 "$f"; echo; }
    done
    echo; echo "== end; GPU status now: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>&1 | head -1)"
  } > "$out" 2>&1
  exit 0 )
  ( cd "$WORK_CPU" && shopt -s nullglob && \
    files=( "$(basename "$out")" logs gpu/logs hrr/logs gpu/hrr/logs */summary.txt summary.txt rank_report.txt \
            */rank_manifest.txt rank_manifest.txt */*/rank_*/meta_flow.txt */rank_*/meta_flow.txt rank_*/meta_flow.txt \
            */*/rank_*/exposure_timeseries.csv */rank_*/exposure_timeseries.csv rank_*/exposure_timeseries.csv \
            opt_*/report.json opt_*/verified.json opt_*/history.jsonl */opt_*/report.json */opt_*/verified.json */opt_*/history.jsonl ) && \
    present=() && for f in "${files[@]}"; do if [ -e "$f" ]; then present+=("$f"); fi; done && \
    tar czf "$tarf" --exclude='*.f32' --exclude='*.gif' --exclude='*.bin' "${present[@]}" )
  say "text to paste: $out ($(wc -c < "$out") bytes)"
  say "logs to upload if asked: $tarf ($(du -h "$tarf" | cut -f1))"
  say "show it with:  cat $out"
}

# clean: what fills the disk, and removal of what can be regenerated. Dry run unless CONFIRM=1.
#   always offered : micromamba's download cache (CUDA is already installed), compiler temporaries,
#                    pip's cache, and the flow snapshots of finished runs (umean_full.f32,
#                    uavg.f32: their figures and gate numbers are already made; Theta is kept)
#   only on request: CLEAN_PACKAGES=1 the olb_lab_*.tar.gz backups and older send_*.tar.gz,
#                    CLEAN_FRAMES=1 the raw animation frames (the GIFs/PNGs in figs/ are kept)
do_clean() {
  ( set +e +o pipefail; shopt -s nullglob
  say "disk: $(df -Ph "$WORK_CPU" | awk 'NR==2{print $4 " free of " $2 " on " $6}')"
  say "largest items in $HOME:"
  du -sh "$HOME"/* "$HOME"/.[!.]* 2>/dev/null | sort -h | tail -12 | sed 's/^/     /'
  say "largest items in $WORK_CPU:"
  du -sh "$WORK_CPU"/* "$WORK_CPU"/gpu/* "$WORK_CPU"/hrr/* "$WORK_CPU"/gpu/hrr/* 2>/dev/null | sort -h | tail -12 | sed 's/^/     /'
  local list=( "$WORK_CPU"/mamba/pkgs "$TMPDIR"/* "$HOME"/.cache/pip )
  while IFS= read -r f; do list+=("$f"); done < <(find "$WORK_CPU" -type f \( -name umean_full.f32 -o -name uavg.f32 \) \
                                                  -not -path '*/release-1.8.1/*' 2>/dev/null)
  if [ "${CLEAN_PACKAGES:-0}" = 1 ]; then
    list+=( "$WORK_CPU"/olb_lab_*.tar.gz )
    local sends=( "$WORK_CPU"/send_*.tar.gz ); [ ${#sends[@]} -gt 1 ] && list+=( "${sends[@]:0:${#sends[@]}-1}" )
  fi
  [ "${CLEAN_FRAMES:-0}" = 1 ] && list+=( "$WORK_CPU"/city_frames/frames "$WORK_CPU"/gpu/city_frames/frames )
  local present=(); for f in "${list[@]}"; do [ -e "$f" ] && present+=("$f"); done
  if [ ${#present[@]} -eq 0 ]; then say "nothing regenerable to remove"; exit 0; fi
  say "regenerable: ${#present[@]} items, $(du -shc "${present[@]}" 2>/dev/null | tail -1 | cut -f1) in total"
  du -sh "${present[@]}" 2>/dev/null | sort -h | tail -8 | sed 's/^/     /'
  local pk=( "$WORK_CPU"/olb_lab_*.tar.gz )
  [ "${CLEAN_PACKAGES:-0}" != 1 ] && [ ${#pk[@]} -gt 0 ] && \
    say "not included: ${#pk[@]} backup package(s) olb_lab_*.tar.gz, $(du -shc "${pk[@]}" | tail -1 | cut -f1) (CLEAN_PACKAGES=1 adds them)"
  if [ "${CONFIRM:-0}" = 1 ]; then
    rm -rf "${present[@]}"; say "removed. disk: $(df -Ph "$WORK_CPU" | awk 'NR==2{print $4 " free"}')"
  else
    say "dry run: nothing removed. CONFIRM=1 $0 clean removes the items above"
  fi
  exit 0 )
}

# the gates' deciding numbers, CPU next to GPU
do_compare() {
  for g in gate6a gate6b_dx4 gate6b_dx2 gate6b_dx1 cube_gm0 cube_ti12 cube_cw20 gate7a gate7c gate7b gate8 wake showcase parity; do
    for side in cpu gpu hrr gpu/hrr; do
      local L="$WORK_CPU/$side/logs/$g.log"; [ $side = cpu ] && L="$WORK_CPU/logs/$g.log"
      [ -f "$L" ] || continue
      grep -E '^\[(6a|6b|7a|7c|7b|8|J|wake)\]|drift|Xr/H|closure|PASS|FAIL' "$L" | tail -6 | sed "s/^/$g $side: /"
    done
  done
  grep -H -E 'MLUPs :' "$WORK_CPU"/out_*.log "$WORK_CPU"/gpu/out_*.log "$WORK_CPU"/hrr/out_*.log "$WORK_CPU"/gpu/hrr/out_*.log 2>/dev/null \
    | sed "s|$WORK_CPU/||" | head -24 || true
}

# ─────────────────────────────────────────────────────────────────────────────
# need_space DIR GB WHAT: stop early, with the reason, if DIR's filesystem has less than GB free
need_space() {
  local free; free=$(df -Pk "$1" 2>/dev/null | awk 'NR==2{print int($4/1048576)}')
  [ "${free:-0}" -ge "$2" ] || { say "only ${free:-?} GB free on the filesystem of $1; $3 needs >= $2 GB."
                                 say "free some space there (or point LAB_TMPDIR at a disk with room) and rerun"; exit 1; }
}

do_setup() {
  command -v g++ >/dev/null || { echo "need g++ (C++20)"; exit 1; }
  # measured: an nvcc build of urban_flow peaks at ~0.3 GB of temporaries (+0.07 GB output)
  need_space "$TMPDIR" 1 "compiling (temporaries in $TMPDIR)"
  need_space "$WORK" 2 "the builds and the gate outputs"
  rm -rf "${TMPDIR:?}"/tmpxft_* "${TMPDIR:?}"/cc* 2>/dev/null || true    # leftovers of interrupted builds
  if [ ! -d "$OLB_ROOT/src" ]; then
    say "fetching OpenLB 1.8.1"
    if [ -f "$WORK_CPU/olb.tar.gz" ] && [ "$WORK" != "$WORK_CPU" ]; then
      ( cd "$(dirname "$OLB_ROOT")" && tar xzf "$WORK_CPU/olb.tar.gz" )        # the GPU tree: same tarball
    else
      ( cd "$(dirname "$OLB_ROOT")" && { curl -fL -o olb.tar.gz "$OLB_URL_GITLAB" || curl -fL -o olb.tar.gz "$OLB_URL_ZENODO"; } \
        && tar xzf olb.tar.gz )
    fi
    [ -d "$OLB_ROOT/src" ] || { echo "no src/ under $OLB_ROOT after extracting — check the tarball's top directory"; exit 1; }
  fi
  if [ "${GPU:-0}" = 1 ]; then
    cuda_toolchain
    # Always re-run (cheap, idempotent: config.mk is rebuilt from the template). A marker left
    # by an older or failed attempt must not keep a stale config.mk, e.g. one carrying
    # nvidia-smi's error text as CUDA_ARCH; and the GPU tree is shared by MODEL=wale and hrr.
    rm -f "$WORK/done/olb_config"
    step olb_config env OLB_ROOT="$OLB_ROOT" NVCC="$NVCC_USED" CUDA_LIBDIR="${CUDA_LIBDIR:-}" \
         CUDA_HOST_CXX="${CUDA_HOST_CXX:-}" CUDA_ARCH="${CUDA_ARCH:-}" "$REPO/olbconfig.sh" gpu
    grep -E '^(CXX|CUDA_ARCH|FLOATING_POINT_TYPE) ' "$OLB_ROOT/config.mk" | sed 's/^/     /'
    grep -qE '^CUDA_ARCH[[:space:]]*:?=[[:space:]]*[0-9]{2,3}[[:space:]]*$' "$OLB_ROOT/config.mk" \
      || { say "config.mk has no numeric CUDA_ARCH (see above): fix the driver or set CUDA_ARCH=86/90"; exit 1; }
  else
    step olb_config env OLB_ROOT="$OLB_ROOT" "$REPO/olbconfig.sh" cpu-mt
  fi
  step olb_external  make -C "$OLB_ROOT/external"
  # OpenLB's core library and any app objects must match the configured mode. A tree built
  # earlier in another mode (serial for Phases 1-4) keeps objects that no longer link
  # ("undefined reference to omp_..."). The marker is keyed on config.mk, so switching mode
  # rebuilds the core once; the app objects are always rebuilt (cheap next to any run).
  local cfg; cfg="$(md5sum "$OLB_ROOT/config.mk" | cut -c1-12)"
  # logs of core builds for other (older) configs are history only; drop them so `send` does
  # not report an old failure as current
  for f in "$WORK"/logs/olb_core_*.log; do [ "$f" = "$WORK/logs/olb_core_$cfg.log" ] || rm -f "$f"; done
  # The marker means "core built for this config.mk"; trust it only while the library is still
  # there: a failed rebuild for another config (it runs clean-core first) deletes it from the tree.
  [ -f "$OLB_ROOT/build/lib/libolbcore.a" ] || rm -f "$WORK/done/olb_core_$cfg"
  step "olb_core_$cfg" sh -c "make -C '$OLB_ROOT' clean-core && make -C '$OLB_ROOT' -j$THREADS core"
  mkdir -p "$APP"
  for f in urban_flow.cpp geometry_loader.h abl_inlet_olb.h abl_inlet.h urban_ops.h urban_les.h tail_fit.h; do ln -sf "$REPO/$f" "$APP/$f"; done
  printf 'EXAMPLE = urban_flow\nOLB_ROOT := ../../..\ninclude $(OLB_ROOT)/default.mk\n' > "$APP/Makefile"
  [ "$MODEL" = hrr ] && echo 'CXXFLAGS += -DCOLLISION_MODEL=3' >> "$APP/Makefile"
  # the app is always rebuilt: it is cheap next to a run and the sources may have been pulled
  rm -f "${APP:?}"/*.o "${APP:?}"/*.d "${APP:?}/urban_flow"
  build app       make -C "$APP"
  if [ "${GPU:-0}" = 1 ] && command -v cuobjdump >/dev/null; then
    cuobjdump --list-elf "$BIN" 2>/dev/null | grep -q "sm_$(grep -E '^CUDA_ARCH' "$OLB_ROOT/config.mk" | awk '{print $3}')" \
      && say "app holds sm_$(grep -E '^CUDA_ARCH' "$OLB_ROOT/config.mk" | awk '{print $3}') kernels" \
      || say "WARNING: no kernels for the configured CUDA_ARCH in $BIN (cuobjdump --list-elf)"
  fi
  build gen_openlb_geom g++ -O3 -std=c++17 -I"$REPO" -DCELL_SIZE_M=4.0 "$REPO/gen_openlb_geom.cpp" -o "$WORK/gen_openlb_geom"
  build gen_openlb_geom8 g++ -O3 -std=c++17 -I"$REPO" -DCELL_SIZE_M=8.0 "$REPO/gen_openlb_geom.cpp" -o "$WORK/gen_openlb_geom8"
  build gen_gate6_geom  g++ -O2 -std=c++17 -I"$REPO" "$REPO/gen_gate6_geom.cpp" -o "$WORK/gen_gate6_geom"
  build linearity_guard g++ -O2 -std=c++17 -I"$REPO" "$REPO/tests/linearity_guard.cpp" -o "$WORK/linearity_guard"
  python3 -c 'import numpy, matplotlib' 2>/dev/null || say "WARNING: python3 numpy/matplotlib missing — analysers need numpy"
  # GPU and HRR runs share the default CPU geometries
  if [ "$WORK" != "$WORK_CPU" ] && [ -f "$WORK_CPU/done/geometry" ]; then
    for g in geom_prod geom_abl geom_cube geom_cube_dx2 geom_box; do ln -sfn "$WORK_CPU/$g" "$WORK/$g"; done
    mark geometry; say "geometry: shared with the CPU runs ($WORK_CPU)"
  fi
  step geometry sh -c '
    DOMAIN=compact OUT_DIR=geom_prod ./gen_openlb_geom &&
    CASE=abl  OUT_DIR=geom_abl  ./gen_gate6_geom &&
    CASE=cube OUT_DIR=geom_cube ./gen_gate6_geom &&
    CASE=cube DX=2 CUBE_H=20 OUT_DIR=geom_cube_dx2 ./gen_gate6_geom &&
    CASE=box  OUT_DIR=geom_box  ./gen_gate6_geom'
  { grep -E 'Voxel grid|Omega|GATE' "$WORK/logs/geometry.log" 2>/dev/null || true; } | head -4 | sed 's/^/     /'
}

do_gates() {
  [ -x "$BIN" ] || { echo "run '$0 setup' first"; exit 1; }
  need_space "$WORK" 1 "the gate runs (~0.6 GB of fields)"
  local P="$REPO/tests"
  # Gate 5 on the production map: OpenLB's own voxel counts vs the Stage-A histogram
  gate gate5 sh -c "GEOM_DIR=geom_prod OUT_DIR=out_gate5 MAX_STEPS=1 '$BIN' > out_gate5.log 2>&1;
    grep GATE5 out_gate5.log | awk '{print \$4, \$5}' | sed 's/olb=//' > g5_olb.txt;
    python3 '$P/gate5_expect.py' geom_prod | awk '/^ +[0-9] /{print \$1, \$4}' > g5_exp.txt;
    cat g5_olb.txt; diff g5_olb.txt g5_exp.txt && echo 'GATE5 PASS: counts identical'"
  # 6a: ABL drift over the empty fetch; also the peak |u| < 0.1 condition (6a')
  gate gate6a sh -c "GEOM_DIR=geom_abl OUT_DIR=out_6a SPINUP_FT=4 AVG_FT=2 CHECK_EVERY=2000 '$BIN' > out_6a.log 2>&1;
    grep 'peak lattice' out_6a.log; awk '/peak lattice/{print (\$8<0.1)?\"[6a] peak |u| < 0.1: PASS\":\"[6a] peak |u| < 0.1: FAIL\"}' out_6a.log;
    python3 '$P/gate6_analyze.py' abl geom_abl out_6a"
  # 6b: cube wake at dx=4 (H/dx=10) and the deciding resolution test at dx=2 (H/dx=20)
  gate gate6b_dx4 sh -c "GEOM_DIR=geom_cube OUT_DIR=out_6b SPINUP_FT=4 AVG_FT=2 CHECK_EVERY=2000 '$BIN' > out_6b.log 2>&1;
    python3 '$P/gate6_analyze.py' cube geom_cube out_6b"
  gate gate6b_dx2 sh -c "GEOM_DIR=geom_cube_dx2 OUT_DIR=out_6b_dx2 SPINUP_FT=4 AVG_FT=2 CHECK_EVERY=5000 '$BIN' > out_6b_dx2.log 2>&1;
    python3 '$P/gate6_analyze.py' cube geom_cube_dx2 out_6b_dx2"
  # 7a/7c/7b on the 40^3 box with a frozen 4 m/s wind
  local B="STEP4=1 STEP4_UNIFORM_U=4 GEOM_DIR=geom_box TS_EVERY=100000"
  gate gate7a sh -c "env $B CLEAR_FRAC=0.001 MAX_BURST_STEPS=9000 OUT_DIR=out_7a '$BIN' > out_7a.log 2>&1;
    python3 '$P/gate7_check.py' budget out_7a && python3 '$P/gate7_check.py' dep out_7a geom_box"
  gate gate7c sh -c "env $B CLEAR_FRAC=0.001 MAX_BURST_STEPS=9000 VD_SCALE=0 OUT_DIR=out_7c '$BIN' > out_7c.log 2>&1;
    python3 '$P/gate7_check.py' nodep out_7c"
  gate gate7b sh -c "env $B CLEAR_FRAC=0 MAX_BURST_STEPS=2400 VD_SCALE=10 ./linearity_guard --run '$BIN' geom_box lin7b"
}

do_city() {
  [ -x "$BIN" ] || { echo "run '$0 setup' first"; exit 1; }
  local P="$REPO/tests"
  # Gate 8: production city (490x167x89, 15H wake buffer), 3 flow-through spin-up, burst to
  # 99 % clearance (CLEAR_FRAC 0.01; the default cap is 20 flow-throughs). Two inlet seeds.
  for s in 1000 2000; do
    step "city_s$s" sh -c "STEP4=1 ABL_SEED=$s GEOM_DIR=geom_prod OUT_DIR=city_s$s CHECK_EVERY=2000 TS_EVERY=1000 '$BIN' > city_s$s.log 2>&1"
  done
  gate gate8 python3 "$P/stage_c_J.py" geom_prod city_s1000 city_s2000
  # the wake must close before the outlet: near-ground centreline u_x over the last 15 % of x
  gate wake python3 - <<'EOF'
import numpy as np
b=open('city_s1000/umean_full.f32','rb').read(); h=np.frombuffer(b[:20],np.int32)
nx,ny,nz=h[:3]; u=np.frombuffer(b[20:],np.float32)[:nx*ny*nz].reshape(nz,ny,nx)
tail=u[1,5:-5,int(0.85*nx):-2]
f=float((tail<0).mean())
print(f"[wake] near-ground reversed-flow fraction over the last 15% of the domain: {100*f:.1f}%")
raise SystemExit(0 if f<0.02 else 1)
EOF
  step viz sh -c "python3 '$REPO/visualize_forward.py' city_s1000"
}

do_package() {
  local ts; ts="$(date +%Y%m%d-%H%M%S)"
  {
    echo "== OpenLB lab run $ts on $(hostname), $THREADS threads =="
    echo "repo commit: $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)"
    cat "$SUMMARY" 2>/dev/null || true
    grep -h -E 'MLUPs :' "$WORK"/out_*.log "$WORK"/city_s*.log 2>/dev/null | sed 's/^/throughput /' || true
    if [ -f "$WORK/gpu/summary.txt" ]; then echo "== GPU build =="; cat "$WORK/gpu/summary.txt"
      grep -h -E 'MLUPs :' "$WORK"/gpu/out_*.log "$WORK"/gpu/city_s*.log 2>/dev/null | sed 's/^/gpu throughput /' || true; fi
  } > "$WORK/summary_$ts.txt"
  # Only what exists: a pattern that matches nothing (e.g. no run logs yet) is dropped rather
  # than handed to tar as a literal name; plain names (no wildcard) are kept only if they exist.
  ( cd "$WORK" && shopt -s nullglob && \
    files=( "summary_$ts.txt" logs done *.log g5_*.txt out_*/meta*.txt out_*/*.png out_*/*.csv \
            city_s*/meta*.txt city_s*/*.csv city_s*/figs lin7b/*/meta_flow.txt \
            gpu/summary.txt gpu/logs gpu/done gpu/*.log gpu/out_*/meta*.txt gpu/out_*/*.png gpu/out_*/*.csv \
            gpu/city_s*/meta*.txt gpu/city_s*/*.csv gpu/city_s*/figs gpu/parity/*.log \
            showcase/SHOWCASE.txt showcase/meta*.txt showcase/*.csv showcase/figs \
            gpu/showcase/SHOWCASE.txt gpu/showcase/meta*.txt gpu/showcase/*.csv gpu/showcase/figs gpu/out_6b_*/*.png ) && \
    present=() && for f in "${files[@]}"; do if [ -e "$f" ]; then present+=("$f"); fi; done && \
    tar czf "olb_lab_$ts.tar.gz" --exclude='*.gif' --exclude='*.f32' "${present[@]}" )
  say "packaged $WORK/olb_lab_$ts.tar.gz ($(du -h "$WORK/olb_lab_$ts.tar.gz" | cut -f1); animations and fields left out)"
  cat "$WORK/summary_$ts.txt"
}

case "${1:-status}" in
  setup)   do_setup ;;
  gates)   do_gates ;;
  city)    do_city ;;
  package) [ "${GPU:-0}" = 1 ] && { WORK="$WORK_CPU"; SUMMARY="$WORK/summary.txt"; }; do_package ;;
  parity)  do_parity ;;
  showcase) do_showcase ;;
  frames)  do_frames ;;
  cube_sens) do_cube_sens ;;
  cube_dx1) do_cube_dx1 ;;
  compare) do_compare ;;
  send)    do_send ;;
  clean)   do_clean ;;
  rank)    do_rank ;;
  rank_report) do_rank_report ;;
  winddirs) do_winddirs ;;
  pathways) do_pathways ;;
  opt_setup) do_opt_setup ;;
  optimize)  do_optimize ;;
  opt_report) do_opt_report ;;
  opt_verify) do_opt_verify ;;
  all)     do_setup; do_gates; do_city; do_package ;;
  status)  ls -1 "$WORK/done" 2>/dev/null | sed 's/^/done: /'; cat "$SUMMARY" 2>/dev/null || true ;;
  *) sed -n '2,20p' "$0"; exit 2 ;;
esac
