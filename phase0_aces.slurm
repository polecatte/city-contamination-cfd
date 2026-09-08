#!/bin/bash
# phase0_aces.slurm - Phase 0 on TAMU HPRC ACES (NVIDIA H100 nodes).
# Submit from the project dir:  sbatch phase0_aces.slurm
#
# IMPORTANT: keep every #SBATCH line BARE (option only). A trailing "# comment"
# or any non-ASCII character on a directive line can make sbatch drop the
# request, which the ACES job_submit plugin then reports as
#   "wall time (--time) is not specified".
# All explanatory notes therefore live on their own comment lines, never on the
# directive lines themselves.
#
# SITE VALUES TO VERIFY before the first submit (they differ per cluster/account):
#   partition + gres : sinfo -o "%P %G %m"   (confirm the H100 partition + gres spelling)
#   account          : your ACCESS/ACES allocation
#   module versions  : module avail CUDA ; module avail GCC ; module avail Python
#   time cap         : sinfo -o "%P %l"      (--time must be under the partition/QOS max)
#
#SBATCH --job-name=lbm_phase0
#SBATCH --output=phase0_%j.out
#SBATCH --error=phase0_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
##SBATCH --account=CHANGE_ME
#
# Notes on the directives above (kept off the directive lines on purpose):
#   --cpus-per-task : OpenMP inlet fill (fill_inlet_plane) is host-side.
#   --mem           : HOST RAM for geometry vectors; GPU VRAM is separate.
#   --partition     : VERIFY the ACES H100 partition name.
#   --gres          : VERIFY the exact one-H100 gres string.
#   --account       : set your allocation, then activate by deleting ONE leading '#'.

set -euo pipefail

# Modules: H100 = Hopper (compute 9.0 => sm_90); CUDA 12 pairs with GCC 12/13.
# (NOT g++-10 - that was only the CUDA-11 + GCC-11 workaround on the A4000 box.)
# VERIFY exact versions with: module avail CUDA ; module avail GCC ; module avail Python
module purge
module load GCC/12.3.0
module load CUDA/12.4.0
# (no Python module: analyze_phase0.py is stdlib-only and runs on system python3.
#  the module name Python/3.11.3-GCCcore-12.3.0 does not exist on ACES and kills the job.)

# H100 is Hopper => sm_90 (an A30 co-tenant would be sm_80). CCBIN is the loaded GCC.
export ARCH=sm_90
export CCBIN=g++
export HOSTCXX=g++
# EasyBuild/LMOD exposes the CUDA root as EBROOTCUDA; fall back to CUDA_HOME.
export CUDA_LIB="${EBROOTCUDA:-${CUDA_HOME:-/usr/local/cuda}}/lib64"

cd "$SLURM_SUBMIT_DIR"
echo "[aces] node=$(hostname)"; nvcc --version | tail -2; g++ --version | head -1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Build fresh (the tree ships without .o; nvcc runs with the loaded GCC).
rm -f kernels.o solver.o av.o airflow_validation
nvcc -O3 -arch=$ARCH -ccbin $CCBIN --extended-lambda -c lbm_kernels.cu -o kernels.o
$HOSTCXX -O3 -std=c++17 -fopenmp -c lbm_solver.cpp         -o solver.o
$HOSTCXX -O3 -std=c++17 -fopenmp -c airflow_validation.cpp -o av.o
$HOSTCXX -O3 -std=c++17 -fopenmp kernels.o solver.o av.o -L"$CUDA_LIB" -lcudart -o airflow_validation
echo "[aces] build OK -> ./airflow_validation"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

# Phase 0: steady-inflow stability + convergence.
# H100 80 GB fits the COST domain to H=54 (~63 GB); the solver's cudaMemGetInfo
# pre-flight aborts cleanly if a grid won't fit, so oversizing is caught, not silent.
# Tunables (env; sensible defaults): FLOORS, H_SWEEP, H_CONV, PLATEAU,
#   MAX_WARMUP (raise so fine grids reach stationarity), SPINUP_FT (spin-up
#   flow-throughs, raise to flush a slow wake), FINE_CHECK/H_FINE/SPOT_WARMUP
#   (H=54 stability spot-check before Phase 1).
# COLLISION OPERATOR — the point of this run.
# run_phase0.sh defaults to COLL=reg; job 1943267 therefore re-ran reg and
# reproduced the known reg verdict instead of testing HRR. Set it explicitly.
# Override at submit time with:  sbatch --export=ALL,COLL=reg ... for an A/B.
COLL=${COLL:-hrr}
HRR_SIGMA=${HRR_SIGMA:-0.98}
echo "[aces] phase0 collision=$COLL sigma=$HRR_SIGMA"
COLL="$COLL" HRR_SIGMA="$HRR_SIGMA" bash run_phase0.sh
echo "[aces] Phase 0 complete - see phase0_summary.txt, fig_phase0.png"
