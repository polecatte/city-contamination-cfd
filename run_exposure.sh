#!/usr/bin/env bash
# run_exposure.sh - one command for a production exposure solve:
#   geometry -> HRR mean flow -> freeze -> reverse_steady -> Omega reduction -> J
# Builds exposure_solve if missing (fails LOUDLY on a compiler error rather than
# silently no-opping), then runs it detached via launch.sh (survives logout).
#
# Usage:  [ARCH=sm_86] [CXX=g++-10] [COLL=hrr] [HRR_SIGMA=0.98] [MAX_WARMUP=80000] \
#         bash run_exposure.sh [trial=2681829] [dx=0.5] [cap=300000] [z_ped=2.0] [w.csv]
set -u
ARCH=${ARCH:-sm_86}                 # A4000 = sm_86
CXX=${CXX:-g++-10}                  # nvcc host compiler (system g++-11+ breaks cudafe++)
CUDA_LIB=${CUDA_LIB:-/usr/local/cuda/lib64}
COLL=${COLL:-hrr}; HRR_SIGMA=${HRR_SIGMA:-0.98}
BIN=./exposure_solve

build_gpu(){
  echo "[exposure] building $BIN (GPU arch=$ARCH, host=$CXX)"
  nvcc -O3 -arch="$ARCH" -ccbin "$CXX" --extended-lambda -c lbm_kernels.cu -o kernels.o || return 1
  "$CXX" -O3 -std=c++17 -fopenmp -c lbm_solver.cpp -o solver.o || return 1
  "$CXX" -O3 -std=c++17 -fopenmp exposure_solve.cpp kernels.o solver.o \
        -L"$CUDA_LIB" -lcudart -o "$BIN" || return 1
}
build_cpu(){
  echo "[exposure] building $BIN (CPU/OpenMP fallback)"
  g++ -O3 -std=c++17 -fopenmp -c lbm_kernels_cpu.cpp -o kc.o || return 1
  g++ -O3 -std=c++17 -fopenmp -c lbm_solver.cpp -o so.o || return 1
  g++ -O3 -std=c++17 -fopenmp exposure_solve.cpp kc.o so.o -o "$BIN" || return 1
}

if [ ! -x "$BIN" ]; then
  if command -v nvcc >/dev/null 2>&1; then build_gpu || { echo "[exposure] GPU build FAILED (see nvcc errors above)"; exit 1; }
  else build_cpu || { echo "[exposure] CPU build FAILED"; exit 1; }; fi
  echo "[exposure] built $BIN"
fi

CMD="$BIN ${1:-2681829} ${2:-0.5} ${3:-300000} ${4:-2.0} ${5:-}"
echo "[exposure] launching: COLLISION=$COLL HRR_SIGMA=$HRR_SIGMA $CMD"
if [ -x ./launch.sh ] || [ -f ./launch.sh ]; then
  COLLISION="$COLL" HRR_SIGMA="$HRR_SIGMA" bash launch.sh -- $CMD
else
  echo "[exposure] launch.sh not found; running in foreground"
  COLLISION="$COLL" HRR_SIGMA="$HRR_SIGMA" $CMD
fi
