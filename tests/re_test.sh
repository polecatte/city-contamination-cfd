#!/usr/bin/env bash
# re_test.sh — Reynolds-independence sensitivity sweep (hardened v2).
# Deletes the output before each run and copies it back only if the run produced
# a FRESH file AND reached its mass budget — so a crash/OOM can never leave run N
# copying run N-1's stale field (the bug that made every floor look identical).
set -u
CITY=${CITY:-1024}; REL=${REL:-600}; WARM=${WARM:-20000}
FLOORS=${FLOORS:-"0.0017 0.005 0.010"}
echo "[re-test] city=${CITY} release=${REL}s  floors: $FLOORS"
for NF in $FLOORS; do
  TAU=$(python3 -c "print(f'{3*$NF+0.5:.3f}')")
  LOG="re_tau${TAU}.log"
  echo "=== nu_floor=$NF  (target tau=$TAU) ==="
  rm -f dispersion_z1.bin
  NU_FLOOR=$NF stdbuf -oL ./lab_test "$CITY" "$REL" 1 "$WARM" > "$LOG" 2>&1 || true
  grep -m1 "Effective floors" "$LOG" | sed 's/^/    /' || true
  if grep -q "Mass budget" "$LOG" && [ -f dispersion_z1.bin ]; then
     cp dispersion_z1.bin "disp_tau${TAU}.bin"; echo "    OK -> disp_tau${TAU}.bin"
  else
     echo "    FAILED at tau=$TAU — no fresh output. Tail of $LOG:"
     tail -n 3 "$LOG" | sed 's/^/      /'
     echo "    (free GPU memory: pkill -9 -f lab_test ; nvidia-smi)"
  fi
done
N=$(ls disp_tau*.bin 2>/dev/null | wc -l)
echo "[re-test] done: $N field(s) saved."
[ "$N" -ge 2 ] && echo "  compare: python3 re_compare.py disp_tau*.bin" || echo "  NOT ENOUGH runs — free GPU mem and rerun."
