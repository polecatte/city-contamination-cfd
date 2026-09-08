#!/usr/bin/env bash
# package_labverify.sh - analyze + package the A4000 lab-verification results into
# one timestamped tarball for transfer off the lab machine.
#   1. grade the airflow modes against reference bands (airflow_validation.py)
#   2. render plots (airflow_validation_plots.py, viz_airflow.py) if matplotlib is present
#   3. score the MUST benchmark if observed data is present
#   4. tar the log, per-test CSVs, slice .bin fields, plots, summary -> archive
#
# Usage:  bash package_labverify.sh [OUTDIR=labverify_out] [MUST_OBS=observed.csv]
set -u
OUTDIR="${1:-labverify_out}"
MUST_OBS="${2:-}"
TS="$(date +%Y%m%d-%H%M%S)"
SUM="$OUTDIR/summary.txt"
mkdir -p "$OUTDIR"

echo "===== lab-verify analysis $TS =====" | tee "$SUM"

# 1. pass/fail tally straight from the run log
if [ -f "$OUTDIR/labverify.log" ]; then
  npass=$(grep -ciE "\bPASS\b" "$OUTDIR/labverify.log" || true)
  nfail=$(grep -ciE "\bFAIL\b" "$OUTDIR/labverify.log" || true)
  echo "run log: $npass PASS lines, $nfail FAIL lines" | tee -a "$SUM"
  grep -iE "VERDICT|FAIL|not stationary|diverg" "$OUTDIR/labverify.log" | head -40 >> "$SUM" || true
else
  echo "WARNING: no $OUTDIR/labverify.log (did the suite run?)" | tee -a "$SUM"
fi

# 2. graded summary against physical reference bands
if command -v python3 >/dev/null && [ -f airflow_validation.py ]; then
  echo "" | tee -a "$SUM"; echo "--- graded airflow summary ---" | tee -a "$SUM"
  # capture python's own exit (a bare `python … | tee` would mask it behind tee)
  if graded=$(python3 airflow_validation.py 2>/dev/null); then
    printf '%s\n' "$graded" | tee -a "$SUM"
  else
    echo "(airflow_validation.py failed / no av_*.csv?)" | tee -a "$SUM"
  fi
fi

# 3. plots (guarded: skip cleanly if matplotlib/headless issues)
for p in airflow_validation_plots.py viz_airflow.py; do
  [ -f "$p" ] && python3 "$p" >/dev/null 2>&1 && echo "plotted: $p" | tee -a "$SUM"
done

# 4. MUST scorecard if predictions + observations are both present
if [ -n "$MUST_OBS" ] && [ -f "$MUST_OBS" ] && [ -f must_predicted.csv ] && [ -x ./must_score ]; then
  echo "" | tee -a "$SUM"; echo "--- MUST scorecard ---" | tee -a "$SUM"
  ./must_score "$MUST_OBS" must_predicted.csv 2>&1 | tee -a "$SUM"
fi

# 5. package everything that exists (missing globs ignored)
ARCH="labverify_results_${TS}.tar.gz"
shopt -s nullglob
files=( "$OUTDIR" av_*.csv *.bin *.png "$SUM" )
# MUST outputs are optional (only present after a must_benchmark run); add them
# only if they exist so tar doesn't emit a suppressed "no such file" error.
[ -f must_predicted.csv ] && files+=( must_predicted.csv )
[ -f must_vel_z2.bin ]    && files+=( must_vel_z2.bin )
[ -d logs ] && files+=( logs/job-*.log logs/airflow_suite-*.log )

# STALE-FILE GUARD: the av_*.csv / *.bin are globbed from the working dir, so a
# partial run (e.g. only cube/cubebench/wale) will sweep up outputs left over
# from an earlier run and silently mix them into the archive (this is how a
# 07-13 av_abl.csv ended up in a 07-14 cube-only archive). Flag files written
# BEFORE this run started. Reference = the run-start marker that run_labverify.sh
# touches at launch; comparing to labverify.log instead false-positives, because
# the log is finalized at the END of the run (after the early modes' CSVs).
REF="$OUTDIR/.labverify_start"
[ -f "$REF" ] || REF=""      # no marker (old run_labverify / manual) => skip guard rather than mis-flag
if [ -n "$REF" ]; then
  stale=()
  for g in av_*.csv *.bin *.png; do
    [ -e "$g" ] || continue
    [ "$g" -ot "$REF" ] && stale+=("$g")
  done
  if [ "${#stale[@]}" -gt 0 ]; then
    echo "WARNING: ${#stale[@]} file(s) written BEFORE this run started — stale, from a previous run:" | tee -a "$SUM"
    printf '  STALE: %s\n' "${stale[@]}" | tee -a "$SUM"
    echo "  (still packaged; delete them before the run, or ignore them in analysis)" | tee -a "$SUM"
  fi
fi

tar czf "$ARCH" "${files[@]}" 2>/dev/null
shopt -u nullglob

echo ""
echo "packaged -> $ARCH ($(du -h "$ARCH" 2>/dev/null | cut -f1))"
echo "  contents: $(tar tzf "$ARCH" 2>/dev/null | wc -l) files"
echo "  summary : $SUM"
echo "copy off the lab box with:  scp <user>@<labhost>:$(pwd)/$ARCH ."
