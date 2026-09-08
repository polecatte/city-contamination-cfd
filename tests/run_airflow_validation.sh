#!/usr/bin/env bash
# run_airflow_validation.sh — build (if needed) and run the airflow validation
# suite overnight, detached. Safe to close the SSH session after launch.
#
#   bash run_airflow_validation.sh [arch]      # arch e.g. sm_86 (A4000), default sm_86
#
# Progress + per-test pass/fail stream to av_run.log; CSVs are av_*.csv.
set -euo pipefail
ARCH="${1:-sm_86}"
CUDA_LIB="${CUDA_LIB:-/usr/local/cuda/lib64}"
HOSTCXX="${HOSTCXX:-g++-10}"
CCBIN="${CCBIN:-g++-10}"        # host compiler for nvcc (system g++-11+ breaks cudafe++ on <functional>)

echo "[build] arch=$ARCH"
if [ ! -f kernels.o ] || [ lbm_kernels.cu -nt kernels.o ]; then
    nvcc -O3 -arch="$ARCH" ${CCBIN:+-ccbin $CCBIN} --extended-lambda -c lbm_kernels.cu -o kernels.o
fi
if [ ! -f solver.o ] || [ lbm_solver.cpp -nt solver.o ]; then
    "$HOSTCXX" -O3 -std=c++17 -fopenmp -c lbm_solver.cpp -o solver.o
fi
"$HOSTCXX" -O3 -std=c++17 -fopenmp -c airflow_validation.cpp -o av.o
"$HOSTCXX" kernels.o solver.o av.o -fopenmp -L"$CUDA_LIB" -lcudart -o airflow_validation
echo "[build] ok → ./airflow_validation"

# also build the analytical Poiseuille check (A1) if present
if [ -f test_poiseuille.cpp ]; then
    "$HOSTCXX" -O3 -std=c++17 -fopenmp -c test_poiseuille.cpp -o tp.o
    "$HOSTCXX" kernels.o solver.o tp.o -fopenmp -L"$CUDA_LIB" -lcudart -o test_poiseuille || true
fi

# Run detached; survives logout. Cheap tests first so failures surface early.
echo "[run] launching full suite (nohup+disown) → av_run.log"
nohup bash -c '
  echo "==== A1 Poiseuille (analytical)  $(date) ===="
  [ -x ./test_poiseuille ] && ./test_poiseuille || echo "  (test_poiseuille not built)"
  # cheap-first: inflow is solver-free and surfaces inflow-statistics problems in seconds.
  # gridconv is the heaviest (three grids incl. H=36 ≈ 22.9M cells) so it runs last.
  for t in inflow smoke mass abl wale cube lateral reynolds cp shell blasius \
           resolution massbudget diffusion stability determinism \
           cubebench canyon gridconv; do
    echo "==== $t  $(date) ===="
    ./airflow_validation "$t" || echo "  ($t exited nonzero)"
  done
  echo "==== analysis  $(date) ===="
  python3 airflow_validation.py || true
  python3 airflow_validation_plots.py || true
  python3 viz_airflow.py || true
  echo "==== ALL DONE  $(date) ===="
' > av_run.log 2>&1 &
disown
echo "[run] PID $! detached. Watch:  tail -f av_run.log"
echo "[run] when finished:  python3 airflow_validation.py"
