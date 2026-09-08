#!/usr/bin/env bash
# run_overnight.sh — comprehensive validation battery, detachable.
#   1. forward contaminant analytical (Gaussian puff/plume, conservation, stability,
#      QUICK overshoot characterisation) — CPU, fast.
#   2. airflow battery, full, at default floor — GPU.
#   3. the definitive low-ν HRR cube ladder (cube/cubebench/wale at 5e-3 then 3e-3,
#      WALL_MODEL on, SPINUP_FT=3) — GPU, the long part.
# Everything logged; builds fail LOUDLY. Runs the collision-mode assertion inside
# each driver, so a wrong-operator run aborts rather than wasting the night.
#
# Detach:  bash launch.sh -- bash run_overnight.sh
# Tune:    ARCH=sm_90 (H100), CXX=g++-12, FLOORS="5e-3 3e-3 1e-3"
set -u
TS=$(date +%Y%m%d-%H%M%S); mkdir -p logs
LOG=logs/overnight-$TS.log
ARCH=${ARCH:-sm_86}; CXX=${CXX:-g++-10}; CUDA_LIB=${CUDA_LIB:-/usr/local/cuda/lib64}
FLOORS=${FLOORS:-"5e-3 3e-3"}
say(){ echo "[overnight $(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "===== COMPREHENSIVE BATTERY start (arch=$ARCH host=$CXX) ====="

# ── build the GPU airflow binary (needed by run_labverify) ───────────────────
if [ ! -x ./airflow_validation ]; then
  say "building airflow_validation (GPU)"
  if command -v nvcc >/dev/null 2>&1; then
    { nvcc -O3 -arch="$ARCH" -ccbin "$CXX" --extended-lambda -c lbm_kernels.cu -o kernels.o &&
      "$CXX" -O3 -std=c++17 -fopenmp -c lbm_solver.cpp -o solver.o &&
      "$CXX" -O3 -std=c++17 -fopenmp -c airflow_validation.cpp -o av.o &&
      "$CXX" kernels.o solver.o av.o -fopenmp -L"$CUDA_LIB" -lcudart -o airflow_validation
    } >>"$LOG" 2>&1 || { say "FATAL: airflow_validation GPU build failed (see log)"; exit 1; }
  else say "FATAL: nvcc not found — cannot build GPU airflow binary"; exit 1; fi
  say "built airflow_validation"
fi

# ── 1. forward contaminant analytical battery (CPU) ──────────────────────────
say "----- (1/3) forward analytical + QUICK overshoot -----"
bash run_forward_validation.sh >>"$LOG" 2>&1 && say "forward analytical: PASS" \
  || say "forward analytical: FAILURES (see log)"

# ── 2. full airflow battery at default floor (GPU) ───────────────────────────
say "----- (2/3) airflow full battery (default floor) -----"
COLL=hrr HRR_SIGMA=0.98 WALL_MODEL=1 MAX_WARMUP=60000 SPINUP_FT=3 \
  bash run_labverify.sh >>"$LOG" 2>&1 && say "airflow full: done" || say "airflow full: nonzero exit"
bash package_labverify.sh >>"$LOG" 2>&1; mv -f labverify_results_*.tar.gz "airflow_full_$TS.tar.gz" 2>/dev/null

# ── 3. low-ν HRR cube ladder (the physics test) ──────────────────────────────
for NF in $FLOORS; do
  say "----- (3/3) HRR cube ladder NU_FLOOR=$NF -----"
  rm -f av_*.csv *.bin 2>/dev/null                 # clean so the archive is not stale-mixed
  COLL=hrr HRR_SIGMA=0.98 WALL_MODEL=1 MAX_WARMUP=60000 SPINUP_FT=3 NU_FLOOR=$NF \
    bash run_labverify.sh cube cubebench wale >>"$LOG" 2>&1 && say "cube ladder $NF: done" || say "cube ladder $NF: nonzero exit"
  bash package_labverify.sh >>"$LOG" 2>&1; mv -f labverify_results_*.tar.gz "cube_nu${NF}_$TS.tar.gz" 2>/dev/null
done

say "===== BATTERY complete. Archives: airflow_full_$TS.tar.gz, cube_nu*_$TS.tar.gz ====="
say "grep 'Collision:' $LOG  # confirm HRR ran (not reg)"
