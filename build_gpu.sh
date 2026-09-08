#!/usr/bin/env bash
# build_gpu.sh — CUDA build of the lab test driver.
# ARCH:  A4000/RTX30xx=sm_86  A100=sm_80  H100=sm_90  RTX4090=sm_89
# CCBIN: host g++ for nvcc if the default is too new (e.g. g++-10).
# CELL:  cell size in metres (default 4). Set CELL=8 to cut memory ~8x.
set -e
NVCC=${NVCC:-nvcc}
CXX=${CXX:-g++}
ARCH=${ARCH:--arch=native}
CUDA_LIB=${CUDA_LIB:-/usr/local/cuda/lib64}
CCBIN_FLAG=${CCBIN:+-ccbin $CCBIN}
HOSTCXX=${CCBIN:-$CXX}
CELLDEF=${CELL:+-DCELL_SIZE_M=$CELL}     # empty unless CELL is set

echo "[gpu-build] CUDA kernels ($ARCH, host=${CCBIN:-default}, cell=${CELL:-4}m)…"
$NVCC -O3 $ARCH $CCBIN_FLAG --extended-lambda -c lbm_kernels.cu -o kernels.o
echo "[gpu-build] solver…";   $HOSTCXX -O3 -std=c++17 -fopenmp $CELLDEF -c lbm_solver.cpp -o solver.o
echo "[gpu-build] lab_test…"; $HOSTCXX -O3 -std=c++17 -fopenmp $CELLDEF -c lab_test.cpp   -o lab_test.o
echo "[gpu-build] link…";     $HOSTCXX kernels.o solver.o lab_test.o -fopenmp -L$CUDA_LIB -lcudart -o lab_test
echo "[gpu-build] done -> ./lab_test"
