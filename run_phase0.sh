#!/usr/bin/env bash
# run_phase0.sh - Phase 0: stability wall + grid convergence (steady) and the
# reg-vs-HRR REGIME decision (turbulent).
#
# PHYSICS SPLIT (from the first H100 run): steady inflow (ABL_RFG=0) gives a
# LAMINAR wake at reg-stable floors (nu_t~0), so it has NO Reynolds-independence
# plateau. We therefore use steady inflow ONLY for the stability wall and grid
# convergence (verification), and make the regime call on a TURBULENT (RFG-on)
# case at reg's floor limit, judged against the reattachment band [1.0, 2.3].
#
#   A  steady floor sweep (H_SWEEP) -> reg's STABILITY WALL; chosen floor = lowest
#      steady-stable floor (+MARGIN), i.e. the highest Re reg can hold.
#   B  RFG-on at the chosen floor, long average -> stable? WALE engaged? Xr in
#      band? = the reg-vs-HRR decision.
#   C  steady grid convergence at the chosen floor (H_CONV); C2 fine-grid (H_FINE)
#      stability spot-check.
#
# A summary is ALWAYS written (phase0_summary.txt), even on an early stop.
# Detach:  bash launch.sh -- bash run_phase0.sh
# Knobs: BIN FLOORS H_SWEEP H_CONV H_FINE COLL MARGIN MAX_WARMUP SPINUP_FT
#        SPOT_WARMUP FINE_CHECK TIMEOUT
set -u
BIN=${BIN:-./airflow_validation}
TIMEOUT=${TIMEOUT:-21600}
FLOORS=${FLOORS:-"1e-2 7e-3 5e-3 3e-3 2e-3 1.5e-3 1e-3"}
H_SWEEP=${H_SWEEP:-32}
H_CONV=${H_CONV:-"24 36"}
H_FINE=${H_FINE:-54}
COLL=${COLL:-reg}
MARGIN=${MARGIN:-0}               # chosen floor = MARGIN steps above the stability wall
MAX_WARMUP=${MAX_WARMUP:-60000}   # averaging cap; raise if the summary flags non-stationarity
SPINUP_FT=${SPINUP_FT:-3}         # spin-up flow-throughs (flush the wake transient before averaging)
HRR_SIGMA=${HRR_SIGMA:-0.98}      # HRR hybrid blend (only used when COLL=hrr; 1=pure RR)
SPOT_WARMUP=${SPOT_WARMUP:-4000}  # short cap for the fine-grid stability spot-check
FINE_CHECK=${FINE_CHECK:-1}
HDR="H,collision,Cw,rfg,nu_floor,tau,stable,maxu,Xr,Cd,nut_peak,nut_mean,n_avg"
SWEEPB=""; CONV=""; FINE=""; FLOOR=""

[ -x "$BIN" ] || { echo "[phase0] build first: need $BIN"; exit 1; }
command -v python3 >/dev/null || { echo "[phase0] need python3 on PATH for analysis"; exit 1; }

run_case(){  # H nf rfg [mw]
  local H=$1 nf=$2 rfg=$3 mw=${4:-$MAX_WARMUP} line
  line=$(env STAB_H="$H" COLLISION="$COLL" NU_FLOOR="$nf" ABL_RFG="$rfg" \
           MAX_WARMUP="$mw" SPINUP_FT="$SPINUP_FT" HRR_SIGMA="${HRR_SIGMA:-0.98}" \
           timeout "$TIMEOUT" stdbuf -oL -eL "$BIN" stabone 2>&1 \
         | grep -m1 '\[stabone\] RESULT' || true)
  if [ -n "$line" ]; then
    g(){ echo "$line" | sed -n "s/.*[ ]$1=\([0-9.eE+-]*\).*/\1/p"; }
    echo "$H,$COLL,$(g Cw),$rfg,$nf,$(g tau),$(g stable),$(g maxu),$(g Xr),$(g Cd),$(g nut_peak),$(g nut_mean),$(g n_avg)"
  else
    local tau; tau=$(awk "BEGIN{printf \"%.4f\",0.5+3*$nf}")
    echo "$H,$COLL,0.325,$rfg,$nf,$tau,0,nan,nan,nan,nan,nan,0"
  fi
}

finish(){  # ALWAYS produce a summary from whatever stages completed
  python3 analyze_phase0.py phase0_sweepA.csv \
    ${SWEEPB:+--sweepB phase0_sweepB.csv} ${CONV:+--conv phase0_conv.csv} \
    ${FINE:+--fine phase0_fine.csv} ${FLOOR:+--floor "$FLOOR"} \
    --margin "$MARGIN" --max-warmup "$MAX_WARMUP" | tee phase0_summary.txt
}

# -- Stage A: steady stability + convergence sweep -----------------------------
echo "$HDR" > phase0_sweepA.csv
echo "[phase0] Stage A - steady (ABL_RFG=0) $COLL, H=$H_SWEEP, floor descent (stability wall)"
for nf in $FLOORS; do
  printf "  [A] nu_floor=%-6s ... " "$nf"
  row=$(run_case "$H_SWEEP" "$nf" 0); echo "$row" >> phase0_sweepA.csv
  echo "$row" | awk -F, '{printf "stable=%s Cd=%s Xr=%s nut=%s\n",$7,$10,$9,$12}'
done

FLOOR=$(python3 analyze_phase0.py --pick phase0_sweepA.csv --margin "$MARGIN")
if [ -z "$FLOOR" ] || [ "$FLOOR" = "none" ]; then
  echo "[phase0] no stable floor in the swept range - stopping."; FLOOR=""; finish; exit 2
fi
echo "[phase0] chosen floor ($COLL stability limit +${MARGIN}): nu_floor=$FLOOR"

# -- Stage B: turbulent regime decision (RFG-on at chosen floor, long average) --
echo "$HDR" > phase0_sweepB.csv; SWEEPB=1
echo "[phase0] Stage B - RFG ON at nu_floor=$FLOOR, H=$H_SWEEP (regime decision; long average)"
row=$(run_case "$H_SWEEP" "$FLOOR" 1); echo "$row" >> phase0_sweepB.csv
echo "$row" | awk -F, '{printf "  -> stable=%s Xr=%s nut=%s\n",$7,$9,$12}'

# -- Stage C: steady grid convergence at the chosen floor ----------------------
echo "$HDR" > phase0_conv.csv; CONV=1
echo "[phase0] Stage C - steady grid convergence, nu_floor=$FLOOR, H in {$H_CONV}"
for H in $H_CONV; do
  printf "  [C] H=%-3s ... " "$H"
  row=$(run_case "$H" "$FLOOR" 0); echo "$row" >> phase0_conv.csv
  echo "$row" | awk -F, '{printf "Cd=%s Xr=%s\n",$10,$9}'
done

# -- Stage C2: fine-grid stability spot-check ----------------------------------
if [ "$FINE_CHECK" = "1" ]; then
  echo "$HDR" > phase0_fine.csv; FINE=1
  echo "[phase0] Stage C2 - fine-grid ($H_FINE) stability spot-check, steady, nu_floor=$FLOOR (short avg)"
  printf "  [C2] H=%-3s ... " "$H_FINE"
  row=$(run_case "$H_FINE" "$FLOOR" 0 "$SPOT_WARMUP"); echo "$row" >> phase0_fine.csv
  echo "$row" | awk -F, '{printf "stable=%s\n",$7}'
fi

finish
echo "[phase0] done -> phase0_sweepA.csv phase0_sweepB.csv phase0_conv.csv phase0_fine.csv phase0_summary.txt fig_phase0.png"
