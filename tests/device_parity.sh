#!/usr/bin/env bash
# device_parity.sh — run the same short cases with a reference and a test urban_flow command
# and compare every output (tests/device_parity.py).
#
#   REF="cmd" TEST="cmd" tests/device_parity.sh GEOM_ROOT OUT_ROOT
#
# REF / TEST are command prefixes, e.g.
#   CPU host loops vs CPU operators (must agree to the last bit; single thread, see below):
#     REF="env OMP_NUM_THREADS=1 HOST_OPS=1 $CPU_BIN" TEST="env OMP_NUM_THREADS=1 HOST_OPS=0 $CPU_BIN" STRICT=1
#   CPU build vs GPU build:
#     REF="$CPU_BIN" TEST="$GPU_BIN"
# GEOM_ROOT holds geom_abl, geom_cube, geom_box (gen_gate6_geom). Outputs and logs go to
# OUT_ROOT/{ref,test}_<case>. Exit 0 only when every case passes.
#
# Cases (together they execute every operator in urban_ops.h):
#   abl     400 NSE steps on the empty fetch: inlet, WALE gradient (+ its one-shot check against
#           OpenLB's functor, which on a GPU tests the device gradient), rough wall, top stress
#   cube    600 NSE steps past the cube with a short time mean: the TMean operator + fetch
#   box7    frozen uniform wind, scalar only: flux, inject, deposit, theta (no NSE at all)
#   boxlive live NSE + coupling + settling (W_SETTLE) + the scalar operators
#
# Tolerances (max|test-ref| / max|ref| per output):
#   STRICT=1   1e-12 everywhere: the operator path must reproduce the host loops. Run it single
#              threaded: OpenLB's own InterpolatedPressure outlet post-processor is not
#              bit-reproducible under OpenMP (rounding-level; located by per-material checksums
#              to MAT_OUTLET), so two 4-thread runs of the SAME path can differ in the last bits.
#   default    TOL_LINEAR (1e-9) for box7, whose transport is linear and does not amplify
#              rounding; TOL_FLOW (1e-4) for the NSE cases. GPU and CPU round differently
#              (fused multiply-add, device libm), and the flow amplifies that. Measured with a
#              CPU build without FMA contraction vs the normal one: <= 3.6e-6 (flow cases),
#              9.5e-14 (box7), so ~30x / ~1e4x headroom (OPENLB_PHASE8_GPU.md §2).
set -uo pipefail
GEOM_ROOT="${1:?GEOM_ROOT}"; OUT_ROOT="${2:?OUT_ROOT}"
: "${REF:?set REF to the reference command}"; : "${TEST:?set TEST to the test command}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ "${STRICT:-0}" = 1 ]; then TL=1e-12; TF=1e-12; else TL="${TOL_LINEAR:-1e-9}"; TF="${TOL_FLOW:-1e-4}"; fi
mkdir -p "$OUT_ROOT"

CASES=(
  "abl|$TF|GEOM_DIR=$GEOM_ROOT/geom_abl MAX_STEPS=400 CHECK_EVERY=100 WALE_GRAD_CHECK=1"
  "cube|$TF|GEOM_DIR=$GEOM_ROOT/geom_cube MAX_STEPS=600 CHECK_EVERY=200 AVG_FT=0.1 AVG_EVERY=5"
  "box7|$TL|GEOM_DIR=$GEOM_ROOT/geom_box STEP4=1 STEP4_UNIFORM_U=4 CLEAR_FRAC=0 MAX_BURST_STEPS=1500 TS_EVERY=100"
  "boxlive|$TF|GEOM_DIR=$GEOM_ROOT/geom_box STEP4=1 MAX_STEPS=300 CHECK_EVERY=100 CLEAR_FRAC=0 MAX_BURST_STEPS=600 TS_EVERY=100 W_SETTLE=0.01"
)
[ -n "${ONLY:-}" ] && { sel=(); for c in "${CASES[@]}"; do [[ " $ONLY " == *" ${c%%|*} "* ]] && sel+=("$c"); done; CASES=("${sel[@]}"); }

fail=0
for c in "${CASES[@]}"; do
  name="${c%%|*}"; rest="${c#*|}"; tol="${rest%%|*}"; envs="${rest#*|}"
  for side in ref test; do
    cmd="$REF"; [ "$side" = test ] && cmd="$TEST"
    out="$OUT_ROOT/${side}_$name"; mkdir -p "$out"
    t0=$(date +%s)
    # shellcheck disable=SC2086
    env $envs OUT_DIR="$out" $cmd > "$out.log" 2>&1; rc=$?
    echo "[parity] $name $side: exit $rc, $(( $(date +%s)-t0 )) s  ($out.log)"
    [ $rc -eq 0 ] || { tail -5 "$out.log" | sed 's/^/    /'; }
  done
  echo "[parity] $name: tolerance $tol"
  if python3 "$HERE/device_parity.py" "$OUT_ROOT/ref_$name" "$OUT_ROOT/test_$name" "$tol" \
       --log-ref "$OUT_ROOT/ref_$name.log" --log-test "$OUT_ROOT/test_$name.log"; then
    echo "[parity] $name PASS"
  else
    echo "[parity] $name FAIL"; fail=1
  fi
done
[ $fail -eq 0 ] && echo "[parity] ALL PASS" || echo "[parity] SOME CASES FAILED"
exit $fail
