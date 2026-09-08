#!/usr/bin/env bash
# run_oblique_validation.sh — physics acceptance for the two-inlet oblique-wind fix.
# Runs the validation ladder (OBLIQUE_DIVERGENCE_DIAGNOSIS.md) on GPU:
#   RUNG 2 horizontal homogeneity   (per angle; empty-domain incident profile)
#   RUNG 3 incompressibility         (per angle; RMS|div u|)
#   RUNG 4 rotation invariance       (across angles; profile overlay must collapse)
# and, for contrast, shows WIND_BC=legacy still diverges at an oblique angle.
#
# USAGE (A4000, full scale):
#   ARCH=-arch=sm_86 CCBIN=g++-10 bash run_oblique_validation.sh
# Knobs: ARCH CCBIN CUDA_LIB  H(=16) SPIN(=4000) AVG(=3000)  ANGLES("0 15 30 45")
set -uo pipefail
cd "$(dirname "$0")"

NVCC="${NVCC:-nvcc}"; ARCH="${ARCH:--arch=sm_86}"; CCBIN="${CCBIN:-g++-10}"
CUDA_LIB="${CUDA_LIB:-/usr/local/cuda/lib64}"; HOSTCXX="${CCBIN:-g++}"
FLAGS="-O3 -std=c++17 -fopenmp"; CELLDEF="${CELL:+-DCELL_SIZE_M=$CELL}"
H="${H:-16}"; ANGLES="${ANGLES:-0 15 30 45}"
# Runtime is sized by the flow-through time n_ft = nx/u_lb (nx=21·H, u_lb≈0.05774,
# so n_ft ≈ 364·H steps). Turbulent ABL statistics need several flow-throughs of
# spin-up and many of averaging — a 1-flow-through run is pure transient. Defaults:
# spin = 3·n_ft, avg = 12·n_ft. Override SPIN/AVG directly, or tune SPIN_FT/AVG_FT.
NFT=$(awk "BEGIN{printf \"%d\", 21*$H/0.05774}")
SPIN_FT="${SPIN_FT:-3}"; AVG_FT="${AVG_FT:-12}"
SPIN="${SPIN:-$(awk "BEGIN{printf \"%d\", $SPIN_FT*$NFT}")}"
AVG="${AVG:-$(awk "BEGIN{printf \"%d\", $AVG_FT*$NFT}")}"
# Smoke pass: a fast H=8 convergence check (short but not trivially short) so a broken
# GPU path OR a wildly non-homogeneous field surfaces before the full sweep.
SMOKE="${SMOKE:-1}"; TIMEOUT="${TIMEOUT:-14400}"
SMOKE_H="${SMOKE_H:-8}"
SMOKE_NFT=$(awk "BEGIN{printf \"%d\", 21*$SMOKE_H/0.05774}")
SMOKE_SPIN="${SMOKE_SPIN:-$(awk "BEGIN{printf \"%d\", 2*$SMOKE_NFT}")}"
SMOKE_AVG="${SMOKE_AVG:-$(awk "BEGIN{printf \"%d\", 6*$SMOKE_NFT}")}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$(nproc)}"
export COLLISION="${COLLISION:-reg}" WALL_MODEL="${WALL_MODEL:-1}" NU_FLOOR="${NU_FLOOR:-5e-3}" D_FLOOR="${D_FLOOR:-1e-4}"

# Stream a command live to console AND append to the log, with an optional timeout,
# line-buffered so [LBM]/[DIAG] checkpoints appear in real time. Returns the child's
# exit status (not tee's). TO=<seconds> or empty.
run_streamed(){ local TO="$1"; shift
  if [ -n "$TO" ]; then stdbuf -oL -eL timeout "$TO" "$@" 2>&1 | tee -a "$LOG"
  else                  stdbuf -oL -eL "$@"                 2>&1 | tee -a "$LOG"; fi
  return "${PIPESTATUS[0]}"; }

LOG=oblique_validation.log; : > "$LOG"
say(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

command -v "$NVCC" >/dev/null 2>&1 || { say "FATAL: nvcc not found — run on the GPU box"; exit 1; }
say "build: kernels + solver + validation harness"
$NVCC -O3 $ARCH ${CCBIN:+-ccbin $CCBIN} --extended-lambda -c lbm_kernels.cu -o kernels.o >>"$LOG" 2>&1 || { say "FATAL nvcc"; exit 1; }
$HOSTCXX $FLAGS $CELLDEF -c lbm_solver.cpp -o solver.o                                    >>"$LOG" 2>&1 || { say "FATAL solver"; exit 1; }
$HOSTCXX $FLAGS $CELLDEF oblique_validation.cpp kernels.o solver.o -L"$CUDA_LIB" -lcudart -o oblique_val >>"$LOG" 2>&1 || { say "FATAL link"; exit 1; }
say "  built ./oblique_val"
rm -f profile_deg*.csv

# ── Smoke pass: fast single-angle H=8 run so a broken GPU path / hang surfaces in
#    a minute or two instead of after a multi-hour full sweep. Streams live. ──────
if [ "$SMOKE" = "1" ]; then
  say "=================================================================="
  say "SMOKE PASS  (H=$SMOKE_H spin=$SMOKE_SPIN avg=$SMOKE_AVG, 30deg, streamed live)"
  say "=================================================================="
  run_streamed "$TIMEOUT" ./oblique_val --deg 30 --H "$SMOKE_H" --spin "$SMOKE_SPIN" --avg "$SMOKE_AVG" --empty
  sec=$?
  if [ "$sec" = 124 ]; then
    say "SMOKE TIMEOUT after ${TIMEOUT}s — the fill/solve is too slow or hung."
    say "  Check the [DIAG] checkpoints above: crawling => raise TIMEOUT or check OMP_NUM_THREADS ($OMP_NUM_THREADS);"
    say "  no checkpoints at all => the run never entered stepping. Aborting before the full sweep."; exit 1
  fi
  if ! grep -q "^RESULT deg=30" "$LOG"; then
    say "SMOKE FAILED (no RESULT line, exit=$sec) — see streamed output above. Aborting."; exit 1
  fi
  say "smoke ok (exit=$sec) — proceeding to the full sweep."
  rm -f profile_deg*.csv
fi

say "=================================================================="
say "OBLIQUE-WIND VALIDATION   (per-face BC, H=$H COLLISION=$COLLISION)"
say "  flow-through n_ft≈$NFT steps | spin=$SPIN (${SPIN_FT}·n_ft) avg=$AVG (${AVG_FT}·n_ft)"
say "  per-angle timeout: ${TIMEOUT:-none}s   threads: $OMP_NUM_THREADS"
say "=================================================================="
declare -A PASS
for d in $ANGLES; do
  say ""; say "### angle ${d}deg (empty domain, per-face inlet) — streaming live"
  run_streamed "$TIMEOUT" ./oblique_val --deg "$d" --H "$H" --spin "$SPIN" --avg "$AVG" --empty
  ec=$?
  [ "$ec" = 124 ] && say "    !! angle ${d}deg TIMED OUT after ${TIMEOUT}s (no profile written)"
  PASS[$d]=$ec
done

# ── RUNG 4: rotation invariance — overlay the per-angle |U|/Uref profiles ────
say ""; say "### RUNG 4 — rotation invariance (overlay of |U|(z)/Uref across angles)"
files=(); for d in $ANGLES; do p=$(printf "profile_deg%02d.csv" "$d"); [ -f "$p" ] && files+=("$p"); done
if [ "${#files[@]}" -ge 2 ]; then
  # Column 2 is |U|/Uref. For each height gather it from every angle, take the STD
  # across angles (U_ref units), and report RMS over height. A 1-D ABL profile
  # cannot depend on θ, so this collapses to ~0 for a rotation-invariant inlet.
  paste -d, "${files[@]}" | awk -F, '
    NR==1{next}
    { sum=0; ss=0; cnt=0;
      for(i=2;i<=NF;i+=5){ v=$i+0; sum+=v; ss+=v*v; cnt++ }
      if(cnt>1){ m=sum/cnt; var=ss/cnt-m*m; if(var<0)var=0; acc+=var; nz++ } }
    END{ rms=(nz>0)?sqrt(acc/nz):0;
         printf "    cross-angle RMS spread = %.1f%% (U_ref units, RMS over z)  -> %s\n",
                100*rms, (rms<0.10?"PASS":"FAIL"); exit (rms<0.10?0:1) }' | tee -a "$LOG"
  R4=${PIPESTATUS[1]}
else
  say "    (need >=2 finite angles for the overlay; some diverged?)"; R4=1
fi

# ── contrast: legacy path is broken at an oblique angle ──────────────────────
# Run in a subdir so its profile CSV can't clobber the per-face profile_deg30.csv.
# At coarse resolution legacy may not NaN outright but goes "stable-but-garbage"
# (huge lateral spread / log-law error); either outcome demonstrates the fix.
say ""; say "### contrast — WIND_BC=legacy at 30deg (expect divergence OR a broken profile)"
mkdir -p _legacy_contrast
lo=$(cd _legacy_contrast && WIND_BC=legacy stdbuf -oL -eL ../oblique_val --deg 30 --H "$H" --spin "$SPIN" --avg 800 --empty 2>&1); lec=$?
echo "$lo" >>"$LOG"; echo "$lo" | grep -E "RESULT|FATAL|DIAG:DIVERGED|RUNG 2 hom" | sed 's/^/    /' | tee -a "$LOG"
lloglaw=$(echo "$lo" | sed -n 's/.*loglaw=\([0-9.]*\).*/\1/p' | tail -1)
legacy_broken=0
[ "$lec" = 2 ] && legacy_broken=1
awk "BEGIN{exit !(${lloglaw:-0} > 0.20)}" && legacy_broken=1
say "    legacy exit=$lec  loglaw=${lloglaw:-n/a}  -> $([ "$legacy_broken" = 1 ] && echo 'BROKEN as expected (fix is necessary)' || echo 'UNEXPECTEDLY OK')"
rm -rf _legacy_contrast

say ""; say "=================================================================="
say "SUMMARY"
for d in $ANGLES; do say "  ${d}deg (rungs 2-3): $([ "${PASS[$d]}" = 0 ] && echo PASS || echo "FAIL/exit${PASS[$d]}")"; done
say "  RUNG 4 (rotation invariance): $([ "${R4:-1}" = 0 ] && echo PASS || echo FAIL)"
say "  legacy contrast: $([ "${legacy_broken:-0}" = 1 ] && echo 'broken as expected (fix necessary)' || echo 'UNEXPECTEDLY OK')"
say "==== done ===="
