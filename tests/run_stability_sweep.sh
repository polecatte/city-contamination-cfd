#!/usr/bin/env bash
# run_stability_sweep.sh — collision stability + WALE-engagement probe (WS1+WS2).
#
# For each resolution H, operator (mrt|reg) and viscosity floor, run one
# aligned-cube solve (stabone) with the PRODUCTION inflow (RFG on) and record:
# stability, reattachment Xr/H, and SGS activity (peak & mean ν_t/(U·H)). Two
# questions at once:
#   WS1  how far below tau=0.53 does each operator stay stable (Re headroom)?
#   WS2  as Re (lower floor) and resolution (higher H) rise, does WALE ENGAGE
#        (ν_t grows, wake turns turbulent) and does Xr/H fall toward ~1.6?
#
# NOTE: stabone now uses the COST-732/AIJ domain make_cube(H,5,15,5,5) — so a run
# is ~1386·H³ cells (H=32 ≈ 45M ≈ 13 GB, H=36 ≈ 65M ≈ 19 GB at 289 B/cell), NOT
# the old compact 490·H³. The RESULT line also now carries rfg= and Cd= fields;
# this script parses the fields it needs by name and ignores the rest. For
# FLOOR-SELECTION with steady inflow use run_phase0.sh; this stays the RFG-on
# resolution/engagement probe.
#
# Each case is a SUBPROCESS: the solver calls exit(2) on divergence, so a blow-up
# must not take the sweep down. "No RESULT line" (diverged/timeout) ⇒ stable=0.
#
# Usage:  bash run_stability_sweep.sh
#   BIN=./airflow_validation
#   HSET="16 24 32"            # resolutions; H=32 ≈ 45M cells/run on the COST domain (heavy) — trim if needed
#   FLOORS="1e-2 7e-3 5e-3 3e-3 2e-3 1.5e-3 1e-3"
#   CW=""                     # WALE constant; empty ⇒ solver default 0.325. e.g. CW=0.5
#   TIMEOUT=2400  OUT=stability_sweep.csv
#
# NOTE ON COST: a full HSET×{mrt,reg}×FLOORS grid is many solves; the mrt cases
# below 1e-2 diverge fast (cheap), but stable reg cases at H=32 are the expensive
# ones. Trim HSET or FLOORS for a quick look.
set -u
BIN=${BIN:-./airflow_validation}
OUT=${OUT:-stability_sweep.csv}
TIMEOUT=${TIMEOUT:-2400}
HSET=${HSET:-"16 24 32"}
FLOORS=${FLOORS:-"1e-2 7e-3 5e-3 3e-3 2e-3 1.5e-3 1e-3"}
CW=${CW:-}                     # empty ⇒ do not set env ⇒ solver default

if [ ! -x "$BIN" ]; then echo "[sweep] build first: need $BIN (see run_airflow_validation.sh)"; exit 1; fi
echo "H,collision,Cw,nu_floor,tau,stable,maxu,Xr,nut_peak,nut_mean,n_avg" > "$OUT"

for H in $HSET; do
 for col in mrt reg; do
  laststable=""
  for nf in $FLOORS; do
    printf "[sweep] H=%-2s %-3s nu_floor=%-6s Cw=%-5s ... " "$H" "$col" "$nf" "${CW:-def}"
    # build the env for this case; only export CW if the user set one
    line=$(env STAB_H="$H" COLLISION="$col" NU_FLOOR="$nf" ${CW:+CW="$CW"} \
             timeout "$TIMEOUT" stdbuf -oL -eL "$BIN" stabone 2>&1 \
           | grep -m1 '\[stabone\] RESULT' || true)
    if [ -n "$line" ]; then
        g(){ echo "$line" | sed -n "s/.*[ ]$1=\([0-9.eE+-]*\).*/\1/p"; }
        cw=$(g Cw); tau=$(g tau); stable=$(g stable); maxu=$(g maxu); Xr=$(g Xr)
        npk=$(g nut_peak); nmn=$(g nut_mean); navg=$(g n_avg)
        echo "$H,$col,$cw,$nf,$tau,$stable,$maxu,$Xr,$npk,$nmn,$navg" >> "$OUT"
        [ "$stable" = "1" ] && laststable=$nf
        echo "stable=$stable Xr/H=$Xr nut_peak=$npk"
    else
        tau=$(awk "BEGIN{printf \"%.4f\",0.5+3*$nf}")
        echo "$H,$col,${CW:-0.325},$nf,$tau,0,nan,nan,nan,nan,0" >> "$OUT"
        echo "DIVERGED/timeout"
    fi
  done
  echo "[sweep] H=$H $col: lowest stable nu_floor = ${laststable:-none}  (lower ⇒ higher Re)"
 done
done
echo "[sweep] wrote $OUT ; plot: python3 plot_stability.py"
