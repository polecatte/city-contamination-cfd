#!/usr/bin/env bash
# run_labverify.sh - lab-computer (A4000) comprehensive verification battery.
# Airflow (under HRR) + scalar-scheme numerical-diffusion certification, each
# checked against its existing pass band. NOT for publication - a whole-solver
# physical-sanity gate. Exports slice fields for the interactive visualizer.
#
# Usage:  COLL=hrr HRR_SIGMA=0.98 bash run_labverify.sh [mode...]
#   default runs the full battery; pass specific modes to run a subset.
# Each airflow mode sizes its own grid; keep them within the A4000's 16 GB.
#
# Env knobs (all recorded in the log banner so a run is self-documenting):
#   NOTE (ABL): the wall model alone will not fix the near-ground profile while
#   z0=0.70 m is resolved with 4 m cells (roughness < cell => first 3-4 points
#   sit inside the roughness sublayer; see av_fig_abl.png kink at z~15 m). The
#   ABL fix needs BOTH WALL_MODEL=1 AND finer near-ground cells (or a smaller z0).
#   The 07-14 run only executed cube/cubebench/wale, so the ABL fix is UNTESTED.
#   WALL_MODEL  1 = rough-wall log-law floor (DEFAULT). Required to sustain the
#               ABL profile: a z0-rough log-law inlet over a smooth no-slip floor
#               drains near-ground momentum (measured: outlet/inlet = 0.18 at the
#               first cell, +14% aloft => T2 drift 36.6%). 0 restores the old
#               smooth-floor behaviour for A/B comparison.
#   MAX_WARMUP  mean-flow warmup cap (DEFAULT 60000; was 30000, which left every
#               mean-flow case flagged "not stationary within cap").
#   SPINUP_FT   spin-up flow-throughs before averaging (DEFAULT 3).
#   NU_FLOOR    viscosity floor. UNSET => each mode's own default (1e-2, laminar
#               Re_eff~115). Set 5e-3 then 3e-3 for the real low-nu HRR test.
#               NOTE: never exported empty - lbm_solver does atof() with no guard,
#               so NU_FLOOR="" would mean a ZERO viscosity floor.
#
# The low-nu HRR physics test (the run that actually exercises HRR):
#   COLL=hrr HRR_SIGMA=0.98 NU_FLOOR=5e-3 MAX_WARMUP=60000 SPINUP_FT=3 \
#     bash launch.sh -- bash run_labverify.sh cube cubebench wale
set -u
BIN=${BIN:-./airflow_validation}
COLL=${COLL:-hrr}                 # collision operator for all airflow cases
HRR_SIGMA=${HRR_SIGMA:-0.98}
WALL_MODEL=${WALL_MODEL:-1}       # rough-wall log-law floor ON by default (ABL fix)
MAX_WARMUP=${MAX_WARMUP:-60000}   # raised from 30000 (means were non-stationary)
SPINUP_FT=${SPINUP_FT:-3}
NU_FLOOR=${NU_FLOOR:-}            # empty => per-mode default; NEVER export empty
TIMEOUT=${TIMEOUT:-14400}
OUT=${OUT:-labverify_out}; mkdir -p "$OUT"

# Certified airflow modes (reuse the existing validation battery) + the new
# scalar numerical-diffusion certification. Order: cheap gates first.
MODES=${*:-"numdiff mass abl wale inflow cube cubebench canyon reynolds cp lateral massbudget"}

[ -x "$BIN" ] || { echo "[labverify] build first: need $BIN"; exit 1; }
CFG="collision=$COLL sigma=$HRR_SIGMA wall_model=$WALL_MODEL max_warmup=$MAX_WARMUP spinup_ft=$SPINUP_FT nu_floor=${NU_FLOOR:-<mode default>}"
echo "[labverify] $CFG"
echo "[labverify] modes: $MODES"
echo "[labverify] results -> $OUT/labverify.log"
: > "$OUT/labverify.log"
# run-start marker: this run's outputs are written AFTER it; a previous run's
# leftovers are older. package_labverify.sh uses this to detect stale files
# reliably (comparing to the LOG mtime false-positives, since the log is
# finalized at the END of the run, after the early modes' CSVs are written).
touch "$OUT/.labverify_start"
# record the config IN the log so the archive is self-documenting (the previous
# run's config could only be inferred from a missing banner)
echo "[labverify] CONFIG $CFG" >> "$OUT/labverify.log"
echo "[labverify] CONFIG modes: $MODES" >> "$OUT/labverify.log"

pass=0; fail=0; declare -a failed
for md in $MODES; do
  echo "-- $md --" | tee -a "$OUT/labverify.log"
  ENVA=(COLLISION="$COLL" HRR_SIGMA="$HRR_SIGMA" WALL_MODEL="$WALL_MODEL"
        MAX_WARMUP="$MAX_WARMUP" SPINUP_FT="$SPINUP_FT")
  # only pass NU_FLOOR when set: an empty value would atof() to a ZERO floor
  [ -n "$NU_FLOOR" ] && ENVA+=(NU_FLOOR="$NU_FLOOR")
  if env "${ENVA[@]}" stdbuf -oL -eL timeout "$TIMEOUT" \
       "$BIN" "$md" >>"$OUT/labverify.log" 2>&1; then
    # a mode "passes" if it printed PASS and never printed FAIL
    if grep -qiE "\bPASS\b" "$OUT/labverify.log" && ! tail -40 "$OUT/labverify.log" | grep -qiE "\bFAIL\b"; then
      echo "   $md: PASS"; pass=$((pass+1))
    else
      echo "   $md: FAIL (no PASS / saw FAIL)"; fail=$((fail+1)); failed+=("$md")
    fi
  else
    echo "   $md: ERROR (nonzero exit / timeout)"; fail=$((fail+1)); failed+=("$md")
  fi
done

echo ""
echo "[labverify] ===== summary: $pass passed, $fail failed ====="
[ "$fail" -gt 0 ] && echo "[labverify] failed: ${failed[*]}"
echo "[labverify] visualize: airflow_validation.py reads the slice CSVs in the run dir"
echo "[labverify] full log: $OUT/labverify.log"
exit $([ "$fail" -eq 0 ] && echo 0 || echo 1)
