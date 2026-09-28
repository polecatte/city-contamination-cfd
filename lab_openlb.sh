#!/usr/bin/env bash
# lab_openlb.sh — run the OpenLB port's gates and production city on the lab box.
#
#   ./lab_openlb.sh setup     # fetch + build OpenLB 1.8.1 (OpenMP), build the app and tools
#   ./lab_openlb.sh gates     # 5, 6a, 6b (dx=4 and dx=2), 7a/7b/7c  — ~3-4 h on 32 cores
#   ./lab_openlb.sh city      # production city, two inlet seeds, to 99 % clearance
#   ./lab_openlb.sh package   # summary.txt + tarball of logs and small outputs
#   ./lab_openlb.sh all       # the four above, in order
#   ./lab_openlb.sh status    # what has run, what passed
#
# Resumable: every step writes $WORK/done/<step> when it finishes and is skipped next time
# (delete the marker to rerun one). Run it under tmux or nohup — it takes hours:
#   tmux new -s olb './lab_openlb.sh all 2>&1 | tee -a ~/olb_lab/lab.log'
#
# Env:
#   WORK      working directory (default ~/olb_lab): OpenLB tree, geometries, outputs, logs
#   OLB_ROOT  an existing OpenLB 1.8.1 tree to use instead of downloading one
#   THREADS   OpenMP threads (default: nproc)
#
# CPU/OpenMP only. The host operators added in Phases 5-7 (WALE gradient refresh, rough wall,
# top stress, the Step-4 loops) are not MPI-reduced and not yet on-device, so neither
# PARALLEL_MODE=MPI nor the GPU build is valid for these runs (Phase 8).
# Results and what each gate means: OPENLB_PHASE5_6_GATES.md.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${WORK:-$HOME/olb_lab}"
THREADS="${THREADS:-$(nproc)}"
OLB_URL_GITLAB="https://gitlab.com/openlb/release/-/archive/1.8.1/release-1.8.1.tar.gz"
OLB_URL_ZENODO="https://zenodo.org/records/15440776/files/release-1.8.1.tar.gz?download=1"
export OMP_NUM_THREADS="$THREADS"

mkdir -p "$WORK/done" "$WORK/logs"
OLB_ROOT="${OLB_ROOT:-$WORK/release-1.8.1}"
APP="$OLB_ROOT/examples/urban/urban_flow"
BIN="$APP/urban_flow"
SUMMARY="$WORK/summary.txt"

say()  { printf '[lab %s] %s\n' "$(date +%H:%M:%S)" "$*"; }
done_() { [ -f "$WORK/done/$1" ]; }
mark() { date > "$WORK/done/$1"; }
note() { printf '%s\n' "$*" >> "$SUMMARY"; }

# step NAME CMD... : run once, log to logs/NAME.log, mark done on success
step() {
  local name="$1"; shift
  if done_ "$name"; then say "skip $name (done)"; return 0; fi
  say "run  $name  (log: $WORK/logs/$name.log)"
  if ( cd "$WORK" && "$@" ) > "$WORK/logs/$name.log" 2>&1; then
    mark "$name"; say "ok   $name"
  else
    say "FAIL $name — see $WORK/logs/$name.log"; return 1
  fi
}

# gate NAME CMD... : like step, but a non-zero exit is a recorded gate FAIL, not an abort
gate() {
  local name="$1"; shift
  if done_ "$name"; then say "skip $name (done)"; return 0; fi
  say "gate $name"
  local rc=0
  ( cd "$WORK" && "$@" ) > "$WORK/logs/$name.log" 2>&1 || rc=$?
  if [ $rc -eq 0 ]; then note "PASS $name"; else note "FAIL $name (exit $rc)"; fi
  grep -E '^\[(6a|6b|7a|7c|8|J)\]|GATE|PASS|FAIL' "$WORK/logs/$name.log" | tail -8 | sed 's/^/     /' || true
  mark "$name"
}

# ─────────────────────────────────────────────────────────────────────────────
do_setup() {
  command -v g++ >/dev/null || { echo "need g++ (C++20)"; exit 1; }
  if [ ! -d "$OLB_ROOT/src" ]; then
    say "fetching OpenLB 1.8.1"
    ( cd "$WORK" && { curl -fL -o olb.tar.gz "$OLB_URL_GITLAB" || curl -fL -o olb.tar.gz "$OLB_URL_ZENODO"; } \
      && tar xzf olb.tar.gz )
    [ -d "$OLB_ROOT/src" ] || { echo "no src/ under $OLB_ROOT after extracting — check the tarball's top directory"; exit 1; }
  fi
  step olb_config    env OLB_ROOT="$OLB_ROOT" "$REPO/olbconfig.sh" cpu-mt
  step olb_external  make -C "$OLB_ROOT/external"
  mkdir -p "$APP"
  for f in urban_flow.cpp geometry_loader.h abl_inlet_olb.h abl_inlet.h; do ln -sf "$REPO/$f" "$APP/$f"; done
  printf 'EXAMPLE = urban_flow\nOLB_ROOT := ../../..\ninclude $(OLB_ROOT)/default.mk\n' > "$APP/Makefile"
  # the app is always rebuilt: it is cheap next to a run and the sources may have been pulled
  say "build urban_flow"; make -C "$APP" > "$WORK/logs/build_app.log" 2>&1 \
    || { say "FAIL app build — $WORK/logs/build_app.log"; exit 1; }
  g++ -O3 -std=c++17 -I"$REPO" -DCELL_SIZE_M=4.0 "$REPO/gen_openlb_geom.cpp" -o "$WORK/gen_openlb_geom"
  g++ -O2 -std=c++17 -I"$REPO" "$REPO/gen_gate6_geom.cpp" -o "$WORK/gen_gate6_geom"
  g++ -O2 -std=c++17 -I"$REPO" "$REPO/tests/linearity_guard.cpp" -o "$WORK/linearity_guard"
  python3 -c 'import numpy, matplotlib' 2>/dev/null || say "WARNING: python3 numpy/matplotlib missing — analysers need numpy"
  step geometry sh -c '
    OUT_DIR=geom_prod ./gen_openlb_geom &&
    CASE=abl  OUT_DIR=geom_abl  ./gen_gate6_geom &&
    CASE=cube OUT_DIR=geom_cube ./gen_gate6_geom &&
    CASE=cube DX=2 CUBE_H=20 OUT_DIR=geom_cube_dx2 ./gen_gate6_geom &&
    CASE=box  OUT_DIR=geom_box  ./gen_gate6_geom'
  grep -E 'Voxel grid|Omega|GATE' "$WORK/logs/geometry.log" | head -4 | sed 's/^/     /'
}

do_gates() {
  [ -x "$BIN" ] || { echo "run '$0 setup' first"; exit 1; }
  local P="$REPO/tests"
  # Gate 5 on the production map: OpenLB's own voxel counts vs the Stage-A histogram
  gate gate5 sh -c "GEOM_DIR=geom_prod OUT_DIR=out_gate5 MAX_STEPS=1 '$BIN' > out_gate5.log 2>&1;
    grep GATE5 out_gate5.log | awk '{print \$4, \$5}' | sed 's/olb=//' > g5_olb.txt;
    python3 '$P/gate5_expect.py' geom_prod | awk '/^ +[0-9] /{print \$1, \$4}' > g5_exp.txt;
    cat g5_olb.txt; diff g5_olb.txt g5_exp.txt && echo 'GATE5 PASS: counts identical'"
  # 6a: ABL drift over the empty fetch; also the peak |u| < 0.1 condition (6a')
  gate gate6a sh -c "GEOM_DIR=geom_abl OUT_DIR=out_6a SPINUP_FT=4 AVG_FT=2 CHECK_EVERY=2000 '$BIN' > out_6a.log 2>&1;
    grep 'peak lattice' out_6a.log; awk '/peak lattice/{print (\$8<0.1)?\"[6a] peak |u| < 0.1: PASS\":\"[6a] peak |u| < 0.1: FAIL\"}' out_6a.log;
    python3 '$P/gate6_analyze.py' abl geom_abl out_6a"
  # 6b: cube wake at dx=4 (H/dx=10) and the deciding resolution test at dx=2 (H/dx=20)
  gate gate6b_dx4 sh -c "GEOM_DIR=geom_cube OUT_DIR=out_6b SPINUP_FT=4 AVG_FT=2 CHECK_EVERY=2000 '$BIN' > out_6b.log 2>&1;
    python3 '$P/gate6_analyze.py' cube geom_cube out_6b"
  gate gate6b_dx2 sh -c "GEOM_DIR=geom_cube_dx2 OUT_DIR=out_6b_dx2 SPINUP_FT=4 AVG_FT=2 CHECK_EVERY=5000 '$BIN' > out_6b_dx2.log 2>&1;
    python3 '$P/gate6_analyze.py' cube geom_cube_dx2 out_6b_dx2"
  # 7a/7c/7b on the 40^3 box with a frozen 4 m/s wind
  local B="STEP4=1 STEP4_UNIFORM_U=4 GEOM_DIR=geom_box TS_EVERY=100000"
  gate gate7a sh -c "env $B CLEAR_FRAC=0.001 MAX_BURST_STEPS=9000 OUT_DIR=out_7a '$BIN' > out_7a.log 2>&1;
    python3 '$P/gate7_check.py' budget out_7a && python3 '$P/gate7_check.py' dep out_7a geom_box"
  gate gate7c sh -c "env $B CLEAR_FRAC=0.001 MAX_BURST_STEPS=9000 VD_SCALE=0 OUT_DIR=out_7c '$BIN' > out_7c.log 2>&1;
    python3 '$P/gate7_check.py' nodep out_7c"
  gate gate7b sh -c "env $B CLEAR_FRAC=0 MAX_BURST_STEPS=2400 VD_SCALE=10 ./linearity_guard --run '$BIN' geom_box lin7b"
}

do_city() {
  [ -x "$BIN" ] || { echo "run '$0 setup' first"; exit 1; }
  local P="$REPO/tests"
  # Gate 8: production city (490x167x89, 15H wake buffer), 3 flow-through spin-up, burst to
  # 99 % clearance (CLEAR_FRAC 0.01; the default cap is 20 flow-throughs). Two inlet seeds.
  for s in 1000 2000; do
    step "city_s$s" sh -c "STEP4=1 ABL_SEED=$s GEOM_DIR=geom_prod OUT_DIR=city_s$s CHECK_EVERY=2000 TS_EVERY=1000 '$BIN' > city_s$s.log 2>&1"
  done
  gate gate8 python3 "$P/stage_c_J.py" geom_prod city_s1000 city_s2000
  # the wake must close before the outlet: near-ground centreline u_x over the last 15 % of x
  gate wake python3 - <<'EOF'
import numpy as np
b=open('city_s1000/umean_full.f32','rb').read(); h=np.frombuffer(b[:20],np.int32)
nx,ny,nz=h[:3]; u=np.frombuffer(b[20:],np.float32)[:nx*ny*nz].reshape(nz,ny,nx)
tail=u[1,5:-5,int(0.85*nx):-2]
f=float((tail<0).mean())
print(f"[wake] near-ground reversed-flow fraction over the last 15% of the domain: {100*f:.1f}%")
raise SystemExit(0 if f<0.02 else 1)
EOF
  step viz sh -c "python3 '$REPO/visualize_forward.py' city_s1000"
}

do_package() {
  local ts; ts="$(date +%Y%m%d-%H%M%S)"
  {
    echo "== OpenLB lab run $ts on $(hostname), $THREADS threads =="
    echo "repo commit: $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)"
    cat "$SUMMARY" 2>/dev/null || true
    grep -h -E 'MLUPs :' "$WORK"/out_*.log "$WORK"/city_s*.log 2>/dev/null | sed 's/^/throughput /' || true
  } > "$WORK/summary_$ts.txt"
  ( cd "$WORK" && tar czf "olb_lab_$ts.tar.gz" summary_$ts.txt logs *.log \
      $(ls -d out_*/meta*.txt out_*/*.png out_*/*.csv city_s*/meta*.txt city_s*/*.csv city_s*/figs lin7b/*/meta_flow.txt 2>/dev/null) )
  say "packaged $WORK/olb_lab_$ts.tar.gz"
  cat "$WORK/summary_$ts.txt"
}

case "${1:-status}" in
  setup)   do_setup ;;
  gates)   do_gates ;;
  city)    do_city ;;
  package) do_package ;;
  all)     do_setup; do_gates; do_city; do_package ;;
  status)  ls -1 "$WORK/done" 2>/dev/null | sed 's/^/done: /'; cat "$SUMMARY" 2>/dev/null || true ;;
  *) sed -n '2,20p' "$0"; exit 2 ;;
esac
