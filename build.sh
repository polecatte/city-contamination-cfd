#!/usr/bin/env bash
# build.sh — compile the lab test driver and the unit tests.
# CPU/OpenMP build (default). For the GPU build, compile lbm_kernels.cu with
# nvcc (see header of test_poiseuille.cpp) and link it in place of kernels.o.
set -e
CXX=${CXX:-g++}
OMP=${OMP:--fopenmp}
echo "[build] kernels (CPU)…"; $CXX -O3 -std=c++17 $OMP -c lbm_kernels_cpu.cpp -o kernels.o
echo "[build] solver…";        $CXX -O3 -std=c++17 -c lbm_solver.cpp -o solver.o
echo "[build] lab_test…";      $CXX -O3 -std=c++17 -c lab_test.cpp   -o lab_test.o
$CXX kernels.o solver.o lab_test.o $OMP -o lab_test
echo "[build] main_cpu (single-size driver)…"; $CXX -O3 -std=c++17 -c main_cpu.cpp -o main.o
$CXX kernels.o solver.o main.o $OMP -o solver
echo "[build] done -> ./lab_test   (then: python3 lab_visualize.py)"
