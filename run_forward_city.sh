#!/usr/bin/env bash
# run_forward_city.sh — PRODUCTION forward airflow + scalar (contaminant) solve on a
# mid-sized DENSITY-builder city, launched DETACHED so it survives logout/SSH close.
#
#   density city builder -> voxelize (solid buildings, RELAXED buffers) ->
#   HRR mean flow -> simultaneous burst over Ω (every street+park ground cell) ->
#   live QUICK transport to 99% clearance -> J = <w, Θ> / |Ω|.
#
# Driver: forward_city.cpp (live-flow forward regime, FORWARD_LIVE.md). Builds
# CPU/OpenMP and FAILS LOUDLY on a compiler error (never silently no-ops). All run
# datasets are stored under OUT_DIR (self-describing binaries; meta.txt lists them)
# so a failed visualizer can reload from stored values. Divergent (non-finite)
# states abort the run and are reported IMMEDIATELY (startup watch + STATUS file).
#
# USAGE
#   [CELL=2.0] [N_ENS=1] [U_INLET=4.0] [WIND_DEG=0] [PEAK_H=45] [CITY_M=600] \
#   [OUT_DIR=forward_city_out] bash run_forward_city.sh
#   DUMP_GEOM_ONLY=1 bash run_forward_city.sh   # store geometry/plan + exit (domain image)
#
#   CELL       grid resolution (m) — COMPILE-TIME -DCELL_SIZE_M. Coarse=2.0.
#   GPU        auto (default: GPU/CUDA if nvcc present, ELSE abort unless CPU is
#              explicitly allowed) | 1 (require GPU) | 0 (force CPU). A GPU build
#              failure ABORTS by default — it never silently runs on CPU. Set
#              ALLOW_CPU_FALLBACK=1 to permit a CPU fallback.
#   RELAXED buffers (this case, below COST 732): BUF_UP=40 BUF_DOWN=70 BUF_LAT=35
#              HEADROOM_H=3  (override any to restore fuller margins).
#   N_ENS      turbulent realisations to average (source is already the full Ω).
#   WATCH_SECS startup window (default 30 s) during which divergence is reported live.
#   Other forward_city knobs (BLOCK_W, STREET_W, EMP_*, RES_GRAD, JH_MIX, FINF,
#   F_IN, F_OUT, Z_PED, SPINUP_FT, CLEAR_FRAC, MAX_WARMUP, STORE_FULL, FLAT_W...)
#   pass straight through — see forward_city.cpp.
#
# MANAGE
#   tail -f logs/forward_city-<ts>.log     # follow;  kill $(cat logs/forward_city.pid) to stop
#   cat <OUT_DIR>/STATUS                    # RUNNING | COMPLETE | DIVERGED | FAIL_*
set -uo pipefail
cd "$(dirname "$0")"

CELL=${CELL:-${1:-2.0}}
CXX=${CXX:-g++}
OMP=${OMP:--fopenmp}
BIN=./forward_city
OBJ=".obj_cell${CELL}"
export OUT_DIR=${OUT_DIR:-forward_city_out}
mkdir -p logs "$OBJ" "$OUT_DIR"

# ── Backend selection: GPU/CUDA when nvcc is available (unless GPU=0), else CPU. ──
#   ARCH      target GPU arch (A4000 = sm_86; H100 = sm_90). Default sm_86.
#   CCBIN     host compiler for nvcc (nvcc needs gcc <= its supported max; A4000
#             boxes usually pin g++-10). Auto-detects g++-12/11/10 if unset.
#   CUDA_LIB  dir containing libcudart. Auto-detected if unset.
ARCH=${ARCH:-sm_86}
# Prefer an OLDER host gcc for nvcc. Thrust pulls in <functional>, and older CUDA
# toolkits cannot parse GCC 11+ standard headers (the classic
# "std_function.h: parameter packs not expanded" error). GCC 9/10 are broadly
# compatible with both old and new CUDA, so try them first; override with CCBIN.
pick_ccbin(){ for c in "${CCBIN:-}" g++-10 g++-9 g++-11 g++-12 g++; do [ -n "$c" ] && command -v "$c" >/dev/null 2>&1 && { echo "$c"; return; }; done; echo ""; }
find_cudalib(){ [ -n "${CUDA_LIB:-}" ] && { echo "$CUDA_LIB"; return; }; dirname "$(find /usr/local/cuda* /usr/lib -name 'libcudart.so*' 2>/dev/null | head -1)" 2>/dev/null; }

build_gpu(){
  local host; host="$(pick_ccbin)"; local cl; cl="$(find_cudalib)"
  echo "[forward_city] building GPU/CUDA at dx=${CELL} m  (arch=$ARCH, host=${host:-default}, cudart=$cl)"
  nvcc -O3 -std=c++17 -arch="$ARCH" --extended-lambda ${host:+-ccbin $host} -DCELL_SIZE_M="$CELL" -c lbm_kernels.cu -o "$OBJ/kernels.o" || return 1
  $CXX -O3 -std=c++17 $OMP -DCELL_SIZE_M="$CELL" -c lbm_solver.cpp -o "$OBJ/so.o" || return 1
  $CXX -O3 -std=c++17 $OMP -DCELL_SIZE_M="$CELL" forward_city.cpp "$OBJ/kernels.o" "$OBJ/so.o" -L"$cl" -lcudart -o "$BIN" || return 1
}
build_cpu(){
  echo "[forward_city] building CPU/OpenMP at dx=${CELL} m (-DCELL_SIZE_M=${CELL})"
  $CXX -O3 -std=c++17 $OMP -DCELL_SIZE_M="$CELL" -c lbm_kernels_cpu.cpp -o "$OBJ/kc.o" || return 1
  $CXX -O3 -std=c++17 $OMP -DCELL_SIZE_M="$CELL" -c lbm_solver.cpp      -o "$OBJ/so.o" || return 1
  $CXX -O3 -std=c++17 $OMP -DCELL_SIZE_M="$CELL" forward_city.cpp "$OBJ/kc.o" "$OBJ/so.o" -o "$BIN" || return 1
}
GPU=${GPU:-auto}
have_nvcc=0; command -v nvcc >/dev/null 2>&1 && have_nvcc=1
# Desired backend: GPU unless GPU=0, or nvcc missing.
want_gpu=0; [ "$GPU" != 0 ] && [ "$have_nvcc" = 1 ] && want_gpu=1

# Build with the REQUIRED backend. A GPU build failure is FATAL by default (never a
# silent CPU fall-through) so you are certain the kernels run on the GPU.
build(){
  if [ "$GPU" = 0 ]; then build_cpu; return; fi
  if [ "$have_nvcc" = 1 ]; then
    if build_gpu; then return 0; fi
    if [ "${ALLOW_CPU_FALLBACK:-0}" = 1 ]; then
      echo "[forward_city] WARNING: GPU build FAILED — ALLOW_CPU_FALLBACK=1, building CPU/OpenMP instead."
      build_cpu; return
    fi
    echo "[forward_city] *** GPU build FAILED and GPU is required. ABORTING. ***"
    echo "[forward_city] HINT: a 'std_function.h: parameter packs not expanded' error means your"
    echo "[forward_city]   CUDA is too old for this host GCC (Thrust pulls in <functional>). Fix with an"
    echo "[forward_city]   OLDER host compiler:  sudo apt-get install -y g++-10  &&  CCBIN=g++-10 bash $0"
    echo "[forward_city]   (try g++-9 if 10 is unavailable), or upgrade CUDA to >= 11.4. Or GPU=0 for CPU."
    nvcc --version 2>/dev/null | tail -2 | sed 's/^/[forward_city]   nvcc: /'
    return 1
  fi
  if [ "$GPU" = 1 ]; then
    echo "[forward_city] *** GPU=1 requested but nvcc is not on PATH. Install CUDA or set GPU=0. ABORTING. ***"; return 1
  fi
  echo "[forward_city] NOTE: nvcc not found — building CPU/OpenMP (set GPU=1 to REQUIRE a GPU build)."
  build_cpu
}

backend_of(){ ldd "$1" 2>/dev/null | grep -q cudart && echo GPU || echo CPU; }
need_build=0
[ -x "$BIN" ] || need_build=1
[ "${REBUILD:-0}" = 1 ] && need_build=1
# Rebuild if the existing binary's backend differs from what we want now.
if [ -x "$BIN" ]; then
  cur="$(backend_of "$BIN")"; [ "$want_gpu" = 1 ] && wb=GPU || wb=CPU
  [ "$cur" != "$wb" ] && { echo "[forward_city] backend change ($cur -> $wb) — rebuilding."; need_build=1; }
fi
for src in forward_city.cpp lbm_solver.cpp lbm_kernels_cpu.cpp lbm_kernels.cu lbm_solver.h lbm_gpu.h \
           city_zoning.h city_builder7.h voxelize.h; do
  [ -f "$src" ] && [ "$src" -nt "$BIN" ] && need_build=1
done
if [ "$need_build" = 1 ]; then
  build || { echo "[forward_city] BUILD FAILED (see errors above)"; exit 1; }
  echo "[forward_city] built $BIN"
fi

# Confirm and announce the ACTUAL backend of the binary we will run.
BACKEND="$(backend_of "$BIN")"
if [ "$BACKEND" = GPU ]; then echo "[forward_city] backend: GPU/CUDA (arch=$ARCH, links libcudart) — kernels run on the GPU."
else echo "[forward_city] backend: CPU/OpenMP (no cudart link)."; fi
if [ "$want_gpu" = 1 ] && [ "$BACKEND" != GPU ]; then
  echo "[forward_city] *** expected a GPU binary but it is not linked against cudart. ABORTING. ***"; exit 1
fi

export COLLISION=${COLLISION:-hrr} SCALAR=${SCALAR:-quick} HRR_SIGMA=${HRR_SIGMA:-0.98}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$(nproc 2>/dev/null || echo 4)}

# ── DUMP_GEOM_ONLY runs in the FOREGROUND (fast, no LBM) so the caller sees it now ─
if [ "${DUMP_GEOM_ONLY:-0}" = 1 ]; then
  echo "[forward_city] geometry-only: storing domain geometry + plan rasters in $OUT_DIR/"
  "$BIN"; rc=$?
  echo "[forward_city] geometry-only exit=$rc  (STATUS: $(cat "$OUT_DIR/STATUS" 2>/dev/null))"
  exit $rc
fi

# ── MANDATORY PRE-FLIGHT (foreground): verify EVERY parameter is correct before we
#    commit the machine to the long single run. Builds geometry, resolves all derived
#    LBM quantities (dt, tau_f, Mach), runs the solver's memory pre-flight, and checks
#    the operator + Omega + receptor. The heavy run launches ONLY if this passes. ─
if [ "${SKIP_PREFLIGHT:-0}" != 1 ]; then
  echo "[forward_city] ===== pre-flight: validating all parameters (no LBM run) ====="
  PREFLIGHT=1 "$BIN"; prc=$?
  if [ $prc -ne 0 ]; then
    echo "[forward_city] *** PRE-FLIGHT FAILED (exit $prc, STATUS=$(cat "$OUT_DIR/STATUS" 2>/dev/null)) — NOT launching the run. Fix the flagged parameter(s) and retry. ***"
    exit $prc
  fi
  echo "[forward_city] pre-flight PASSED — parameters correct; launching the production run."
fi

ts="$(date +%Y%m%d-%H%M%S)"
log="logs/forward_city-${ts}.log"; pidf="logs/forward_city.pid"
: > "$OUT_DIR/STATUS"

# Line-buffer via stdbuf if available, so `tail -f` shows live output even for a
# binary built before the driver's own setvbuf() line-buffering was added.
SB=""; command -v stdbuf >/dev/null 2>&1 && SB="stdbuf -oL -eL"

# Wrap the job so its exit status is captured and a clear PASS/FAIL line hits the log.
runner="logs/forward_city-${ts}.runner.sh"
cat > "$runner" <<EOF
#!/usr/bin/env bash
$SB "$BIN"; rc=\$?
if [ \$rc -ne 0 ]; then
  echo ""
  echo "*** RUN FAILED (exit \$rc) — STATUS=\$(cat '$OUT_DIR/STATUS' 2>/dev/null). See FATAL/divergence above. ***"
else
  echo "*** RUN OK (exit 0) — datasets in $OUT_DIR/ ***"
fi
exit \$rc
EOF
chmod +x "$runner"

# Detach fully: new session (setsid), ignore SIGHUP (nohup), no stdin (</dev/null).
setsid nohup "$runner" </dev/null >"$log" 2>&1 &
pid=$!
echo "$pid" > "$pidf"

echo "[forward_city] config: backend=$BACKEND CELL=${CELL} N_ENS=${N_ENS:-1} U_INLET=${U_INLET:-4.0} WIND_DEG=${WIND_DEG:-0} "\
"PEAK_H=${PEAK_H:-45} CITY_M=${CITY_M:-600} buffers(up/down/lat)=${BUF_UP:-40}/${BUF_DOWN:-70}/${BUF_LAT:-35} OUT_DIR=$OUT_DIR"
echo "launched : $BIN (detached, backend=$BACKEND)  pid=$pid  log=$log"

# ── Immediate divergence watch: report at once if it blows up early ───────────────
WATCH=${WATCH_SECS:-30}
echo "[forward_city] watching first ${WATCH}s for divergence/early failure..."
for ((t=0; t<WATCH; t++)); do
  sleep 1
  st="$(cat "$OUT_DIR/STATUS" 2>/dev/null || true)"
  if [ "$st" = "DIVERGED" ] || [[ "$st" == FAIL_* ]]; then
    echo ""
    echo ">>> FAILURE DETECTED: STATUS=$st  <<<"
    grep -E "FATAL|DIVERG|BUILD FAIL" "$log" | tail -8 || true
    echo "    full log: $log"
    exit 1
  fi
  if ! kill -0 "$pid" 2>/dev/null; then
    # process ended within the watch window — inspect exit
    if grep -q "RUN FAILED" "$log"; then
      echo ""; echo ">>> RUN FAILED during startup <<<"; grep -E "FATAL|DIVERG|RUN FAILED" "$log" | tail -8
      exit 1
    fi
    echo "[forward_city] run finished quickly (small case) — STATUS=$(cat "$OUT_DIR/STATUS" 2>/dev/null)"
    break
  fi
done

echo "[forward_city] healthy so far — running detached."
echo "  follow : tail -f $log"
echo "  status : cat $OUT_DIR/STATUS"
echo "  stop   : kill $pid   # or: kill \$(cat $pidf)"
echo "Safe to close this session now — the run continues in the background."
