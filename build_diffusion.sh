#!/usr/bin/env bash
# build_diffusion.sh — build the transport / robustness diagnostic tools:
#   diffusion_compare     : linear-upwind numerical diffusion vs van-Leer (penalty)
#   ranking_stability     : reverse-objective vs production-truth design ranking
#   robustness_overnight  : intensive 2x2 (scheme x Reynolds-floor) ranking + integrity
#
# Default (no arg): self-contained, fast builds (no LBM) — bench + selftests.
# 'city' arg: also builds the production (LBM) paths, linking the kernels.o +
#   solver.o produced by build.sh (CPU) OR build_gpu.sh (GPU). The kernel backend
#   is AUTO-DETECTED: a GPU kernels.o (nvcc) references the CUDA runtime, so it must
#   be linked with -lcudart via the host compiler — exactly as build_gpu.sh links
#   lab_test. A CPU kernels.o links with a plain g++ command. Mixing a g++ link with
#   a GPU object is what produced the "undefined reference to cudaMalloc/
#   cudaLaunchKernel/__cudaPushCallConfiguration" wall of linker errors.
set -e
MODE="${1:-bench}"; CXX=${CXX:-g++}; FLAGS="-O3 -std=c++17 -fopenmp"
# SWEEP_CELL: cell size (m) for ranking/robustness domains. Default 8 m so the
# COST-buffered sweep domains fit a 16 GB GPU (4 m ⇒ 200 M+ cells ⇒ OOM). Ranking
# tests want many designs at consistent fidelity, not production resolution; the
# top-k designs are re-confirmed at 4 m separately. Set SWEEP_CELL=4 to force 4 m.
SWEEP_CELL="${SWEEP_CELL:-8.0}"
CELLFLAG="-DCELL_SIZE_M=${SWEEP_CELL}"
TOOLS="diffusion_compare ranking_stability robustness_overnight"

if [ "$MODE" = "city" ] || [ "$MODE" = "gpu" ]; then
    [ -f kernels.o ] && [ -f solver.o ] || { echo "needs kernels.o + solver.o (run build.sh or build_gpu.sh first)"; exit 1; }
    # GPU object iff kernels.o references the CUDA runtime at all (unanchored — nm
    # spacing varies by platform). 'gpu' forces the cudart link without detection.
    if [ "$MODE" = "gpu" ] || nm kernels.o 2>/dev/null | grep -q "cudaLaunchKernel"; then
        CUDA_LIB=${CUDA_LIB:-/usr/local/cuda/lib64}
        HOSTCXX=${CCBIN:-$CXX}        # same host compiler convention as build_gpu.sh
        [ -d "$CUDA_LIB" ] || echo "[build_diffusion] WARNING: CUDA_LIB=$CUDA_LIB not found — pass CUDA_LIB=/path/to/cuda/lib64"
        echo "[build_diffusion] GPU link: $HOSTCXX + -lcudart ($CUDA_LIB)  [sweep cell=${SWEEP_CELL} m]"
        for t in $TOOLS; do
            echo "[build_diffusion]   $t…"
            $HOSTCXX $FLAGS $CELLFLAG -DWITH_LBM "$t.cpp" kernels.o solver.o -L"$CUDA_LIB" -lcudart -o "$t"
        done
    else
        echo "[build_diffusion] CPU link: $CXX  [sweep cell=${SWEEP_CELL} m]"
        for t in $TOOLS; do
            echo "[build_diffusion]   $t…"
            $CXX $FLAGS $CELLFLAG -DWITH_LBM "$t.cpp" kernels.o solver.o -o "$t"
        done
    fi
else
    echo "[build_diffusion] building self-contained tools (no LBM)…"
    for t in $TOOLS; do $CXX $FLAGS "$t.cpp" -o "$t"; done
fi
echo "[build_diffusion] done."
echo
echo "Fast checks (no LBM):  ./ranking_stability selftest ; ./robustness_overnight selftest"
if [ "$MODE" = "city" ]; then
echo "Ranking test:          ./ranking_stability run 6 512 60 4000"
echo "Overnight (detached):  nohup bash -c 'OMP_NUM_THREADS=\$(nproc) ./robustness_overnight run 48 768 120 10000 > robust.log 2>&1' >/dev/null 2>&1 & disown"
fi
