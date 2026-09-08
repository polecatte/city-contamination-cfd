#!/usr/bin/env bash
# run_oblique_diagnosis.sh — GPU battery for the T_mass oblique-wind FATAL divergence.
# See OBLIQUE_DIVERGENCE_DIAGNOSIS.md for the full write-up.
#
# WHAT DIVERGED (recap): `airflow_validation all` aborted on the SECOND 336x176x96
# solve. That grid is T_mass's cube domain, and T_mass sweeps wind {0,15,30,45}deg.
# 0deg is stable; the first OBLIQUE angle is what NaN'd. Hypothesis: the single
# x-normal inlet imposes a cross-stream uInY=u_lb*sin(theta) for theta!=0, but the
# +/-y boundaries are SPECULAR SYMMETRY planes (zero cross-stream flux) — the two
# are inconsistent for any oblique angle, so injected y-momentum piles up against
# +y / the inlet corner and blows up. The seven experiments below each rule a
# hypothesis IN or OUT rather than merely re-observing the crash.
#
# Each case runs in its OWN process (the solver hard-exits(2) on divergence, which
# is exactly what killed the overnight loop). This script captures the exit code
# (0=finite, 2=diverged) and the patched solver's [DIAG:...] line (argmax|u| +
# first-NaN cell, boundary/corner-labelled — that IS experiment #5, on every run).
#
# USAGE (A4000 defaults; full T_mass scale H=16 => 336x176x96):
#   ARCH=-arch=sm_86 CCBIN=g++-10 bash run_oblique_diagnosis.sh
# Knobs (env): ARCH CCBIN CUDA_LIB   H(=16)  SPIN(=3000)  COLLISION(=reg)
#              NU_FLOOR(=5e-3) WALL_MODEL(=1)   WITH_CPU_MIRROR(=0)
set -uo pipefail
cd "$(dirname "$0")"

# ── toolchain (mirrors build_gpu.sh / run_overnight.sh) ──────────────────────
NVCC="${NVCC:-nvcc}"
ARCH="${ARCH:--arch=sm_86}"                 # A4000=sm_86 A100=sm_80 H100=sm_90 4090=sm_89
CCBIN="${CCBIN:-g++-10}"                     # host g++ for nvcc (g++-11+ too new)
CUDA_LIB="${CUDA_LIB:-/usr/local/cuda/lib64}"
HOSTCXX="${CCBIN:-g++}"; FLAGS="-O3 -std=c++17 -fopenmp"
CELLDEF="${CELL:+-DCELL_SIZE_M=$CELL}"

# ── physics / run knobs ──────────────────────────────────────────────────────
H="${H:-16}"                                 # cube height in cells (T_mass uses 16)
SPIN="${SPIN:-3000}"                          # Phase-A1 steps (divergence caught here; real one NaN'd by 2500)
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$(nproc)}"
# reg is the production collision mode AND is stable at wind=0, so any oblique
# divergence under reg is attributable to the boundary, not the collision.
export COLLISION="${COLLISION:-reg}" WALL_MODEL="${WALL_MODEL:-1}"
export NU_FLOOR="${NU_FLOOR:-5e-3}" D_FLOOR="${D_FLOOR:-1e-4}"
WITH_CPU_MIRROR="${WITH_CPU_MIRROR:-0}"       # experiment #7 CPU side (off by default)

LOG=oblique_diagnosis.log; : > "$LOG"
CSV=oblique_diagnosis.csv; echo "experiment,backend,exit,outcome,last_diag" > "$CSV"
say(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# ── build (patched sources) ──────────────────────────────────────────────────
command -v "$NVCC" >/dev/null 2>&1 || { say "FATAL: nvcc not found — run on the GPU box (set CUDA_LIB/CCBIN)"; exit 1; }
say "build: CUDA kernels ($ARCH host=$CCBIN)  +  solver  +  driver"
$NVCC -O3 $ARCH ${CCBIN:+-ccbin $CCBIN} --extended-lambda -c lbm_kernels.cu -o kernels.o   >>"$LOG" 2>&1 || { say "FATAL: nvcc kernels.o failed — see $LOG"; exit 1; }
$HOSTCXX $FLAGS $CELLDEF -c lbm_solver.cpp -o solver.o                                      >>"$LOG" 2>&1 || { say "FATAL: solver.o failed — see $LOG"; exit 1; }
$HOSTCXX $FLAGS $CELLDEF oblique_divergence_test.cpp kernels.o solver.o -L"$CUDA_LIB" -lcudart -o oblique_test >>"$LOG" 2>&1 || { say "FATAL: driver link failed — see $LOG"; exit 1; }
say "  built ./oblique_test  (GPU)"

if [ "$WITH_CPU_MIRROR" = 1 ]; then
  say "build: CPU mirror (experiment #7)"
  g++ $FLAGS $CELLDEF -c lbm_kernels_cpu.cpp -o kernels_cpu.o                               >>"$LOG" 2>&1 \
   && g++ $FLAGS $CELLDEF -c lbm_solver.cpp -o solver_cpu.o                                 >>"$LOG" 2>&1 \
   && g++ $FLAGS $CELLDEF oblique_divergence_test.cpp kernels_cpu.o solver_cpu.o -o oblique_test_cpu >>"$LOG" 2>&1 \
   && say "  built ./oblique_test_cpu" || say "  WARN: CPU-mirror build failed (skipping exp #7 CPU side)"
fi

# ── run one case: run <label> <extra-env> <driver-args...> ───────────────────
run(){
  local label="$1"; shift; local envp="$1"; shift
  local bin="${FORCE_BIN:-./oblique_test}"
  say "── $label   [$bin $*  ${envp:+| env: $envp}]"
  local out ec; out=$(env $envp "$bin" "$@" --H "$H" --spin "$SPIN" 2>&1); ec=$?
  echo "$out" >>"$LOG"
  local diag; diag=$(echo "$out" | grep -E "DIAG:(DIVERGED|spinup)" | tail -1 | sed 's/^\[//;s/\]$//')
  local outcome="DIVERGED"; [ "$ec" = 0 ] && outcome="FINITE"; [ "$ec" = 124 ] && outcome="BOUNDED(timeout)"
  say "   -> exit=$ec  $outcome";  [ -n "$diag" ] && say "     $diag"
  echo "\"$label\",\"$bin\",$ec,\"$outcome\",\"$diag\"" >>"$CSV"
}

say "=================================================================="
say "OBLIQUE-WIND DIVERGENCE DIAGNOSIS"
say "  grid=$((21*H))x$((11*H))x$((6*H))  H=$H  SPIN=$SPIN  COLLISION=$COLLISION  NU_FLOOR=$NU_FLOOR  WALL_MODEL=$WALL_MODEL"
say "=================================================================="

say ""; say "### EXP 1 — angle onset  (predict: 0deg FINITE; growth/divergence rising in sin theta)"
for d in 0 5 10 15 45; do run "1:angle deg=$d" "" --deg "$d"; done

say ""; say "### EXP 2 — nu_floor control  (predict: 45deg still fails => log's 'raise nu_floor' hint refuted)"
for nf in 5e-3 1e-2 2e-2; do run "2:nu_floor=$nf deg=45" "NU_FLOOR=$nf" --deg 45; done

say ""; say "### EXP 3 — turbulence off  (predict: persists => MEAN cross-stream is the driver, not RFG)"
run "3:rfg-off deg=45" "" --deg 45 --rfg0

say ""; say "### EXP 4 — empty domain  (predict: persists => boundary, not the cube wake)"
run "4:empty deg=45" "" --deg 45 --empty

say ""; say "### EXP 5 — blow-up location  (annotated on every run above via [DIAG:...]; expect +/-y plane & inlet corner)"

say ""; say "### EXP 6 — lateral BC swap  (predict: OPEN stays bounded/decays => specular reflection was the cause)"
run "6:45deg symmetry (default)" "" --deg 45
run "6:45deg OPEN +/-y"          "" --deg 45 --open

say ""; say "### EXP 7 — CPU/GPU mirror  (predict: same divergence => shared BC logic, not a CUDA/float artifact)"
FORCE_BIN=./oblique_test     run "7:45deg symmetry (GPU)" "" --deg 45
if [ "$WITH_CPU_MIRROR" = 1 ] && [ -x ./oblique_test_cpu ]; then
  FORCE_BIN=./oblique_test_cpu run "7:45deg symmetry (CPU)" "" --deg 45
else
  say "   (CPU side skipped; set WITH_CPU_MIRROR=1 to build+run it)"
fi

say ""; say "=================================================================="
say "SUMMARY  (also $CSV)"; column -s, -t "$CSV" 2>/dev/null | sed 's/^/    /' | tee -a "$LOG"
say ""
say "READING IT: exit=0 FINITE=stable; exit=2=hard NaN (its DIAG names the first-NaN cell);"
say "exit=124=survived the window—watch the max|u| trend across [DIAG] lines (rising at +/-y = pileup)."
say "==== done ===="
