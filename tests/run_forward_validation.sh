#!/usr/bin/env bash
# run_forward_validation.sh — forward contaminant solver validation.
# Runs the self-contained analytical dispersion battery (fast) and smoke-checks
# that the live-flow forward driver builds. See TEST_SUITE.md section B.
set -u
echo "===== forward contaminant validation ====="
echo "[1/2] analytical dispersion (Gaussian puff + plume, conservation, stability)"
g++ -O2 -std=c++17 -I. plume_validation.cpp -o plume_validation || { echo "BUILD FAIL"; exit 1; }
./plume_validation; rc=$?
echo ""
echo "[2/3] QUICK unphysical-value characterisation"
g++ -O2 -std=c++17 -I. overshoot_test.cpp -o overshoot_test && ./overshoot_test; orc=$?
echo ""
echo "[3/3] live-flow forward driver build check"
g++ -O3 -std=c++17 -fopenmp -c lbm_kernels_cpu.cpp -o /tmp/kc.o 2>/dev/null \
 && g++ -O3 -std=c++17 -fopenmp -c lbm_solver.cpp -o /tmp/so.o 2>/dev/null \
 && g++ -O3 -std=c++17 -fopenmp forward_live.cpp /tmp/kc.o /tmp/so.o -o /tmp/fl 2>/dev/null \
 && echo "  forward_live builds+links OK" || echo "  forward_live BUILD FAIL"
rm -f /tmp/kc.o /tmp/so.o /tmp/fl
echo ""
[ $rc -eq 0 ] && echo "===== analytical battery PASSED =====" || echo "===== FAILURES (rc=$rc) ====="
exit $rc
