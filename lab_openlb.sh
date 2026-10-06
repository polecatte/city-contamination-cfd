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
#   ./lab_openlb.sh compare         # CPU and GPU gate numbers side by side
#   [GPU=1] ./lab_openlb.sh frames     # production city (seed 1000) again with snapshots -> GIFs + stills
#   [GPU=1] ./lab_openlb.sh showcase   # small city end to end: flow, burst, J, figures, scorecard
#   [GPU=1] ./lab_openlb.sh cube_sens  # 6b sensitivity at dx=4: floor model, inflow turbulence, C_w
#   GPU=1   ./lab_openlb.sh cube_dx1   # 6b at dx=1 m (H/dx=40, 89 M cells): needs >= 60 GB GPU (H100)
#
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
  command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total,memory.used --format=csv,noheader \
    | sed 's/^/     GPU: /' || say "WARNING: nvidia-smi not found — the GPU build will compile but cannot run here"
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

# the gates' deciding numbers, CPU next to GPU
do_compare() {
  for g in gate6a gate6b_dx4 gate6b_dx2 gate6b_dx1 cube_gm0 cube_ti12 cube_cw20 gate7a gate7c gate7b gate8 wake showcase; do
    for side in cpu gpu; do
      local L="$WORK_CPU/logs/$g.log"; [ $side = gpu ] && L="$WORK_CPU/gpu/logs/$g.log"
      [ -f "$L" ] || continue
      grep -E '^\[(6a|6b|7a|7c|7b|8|J|wake)\]|drift|Xr/H|closure|PASS|FAIL' "$L" | tail -6 | sed "s/^/$g $side: /"
    done
  done
  grep -h -E 'MLUPs :' "$WORK_CPU"/out_*.log "$WORK_CPU"/gpu/out_*.log 2>/dev/null | head -20 || true
}

# ─────────────────────────────────────────────────────────────────────────────
do_setup() {
  command -v g++ >/dev/null || { echo "need g++ (C++20)"; exit 1; }
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
    step olb_config env OLB_ROOT="$OLB_ROOT" NVCC="$NVCC_USED" CUDA_LIBDIR="${CUDA_LIBDIR:-}" \
         CUDA_HOST_CXX="${CUDA_HOST_CXX:-}" CUDA_ARCH="${CUDA_ARCH:-}" "$REPO/olbconfig.sh" gpu
    grep -E '^(CXX|CUDA_ARCH|FLOATING_POINT_TYPE) ' "$OLB_ROOT/config.mk" | sed 's/^/     /'
  else
    step olb_config env OLB_ROOT="$OLB_ROOT" "$REPO/olbconfig.sh" cpu-mt
  fi
  step olb_external  make -C "$OLB_ROOT/external"
  # OpenLB's core library and any app objects must match the configured mode. A tree built
  # earlier in another mode (serial for Phases 1-4) keeps objects that no longer link
  # ("undefined reference to omp_..."). The marker is keyed on config.mk, so switching mode
  # rebuilds the core once; the app objects are always rebuilt (cheap next to any run).
  local cfg; cfg="$(md5sum "$OLB_ROOT/config.mk" | cut -c1-12)"
  step "olb_core_$cfg" sh -c "make -C '$OLB_ROOT' clean-core && make -C '$OLB_ROOT' -j$THREADS core"
  mkdir -p "$APP"
  for f in urban_flow.cpp geometry_loader.h abl_inlet_olb.h abl_inlet.h urban_ops.h urban_les.h; do ln -sf "$REPO/$f" "$APP/$f"; done
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
  all)     do_setup; do_gates; do_city; do_package ;;
  status)  ls -1 "$WORK/done" 2>/dev/null | sed 's/^/done: /'; cat "$SUMMARY" 2>/dev/null || true ;;
  *) sed -n '2,20p' "$0"; exit 2 ;;
esac
