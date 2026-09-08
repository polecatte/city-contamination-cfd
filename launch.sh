#!/usr/bin/env bash
# launch.sh — start a long-running LBM job DETACHED so it survives SSH logout.
# Uses setsid + nohup + </dev/null (new session, no controlling terminal), so the
# job keeps running after you disconnect. Returns immediately; logs to logs/.
#
# USAGE
#   bash launch.sh validation            # rungs 2-4 oblique validation (run_oblique_validation.sh)
#   bash launch.sh diagnosis             # the 7 divergence experiments (run_oblique_diagnosis.sh)
#   bash launch.sh suite [mode]          # airflow_validation suite (mode default: all via run_airflow_validation.sh)
#   bash launch.sh -- <cmd> [args...]    # run any command detached
#
# ENV KNOBS pass straight through to the job, e.g.:
#   ARCH=-arch=sm_86 CCBIN=g++-10 H=16 SPIN=4000 AVG=3000 bash launch.sh validation
#   (suite uses bare arch: pass as `bash launch.sh suite all sm_86` — see below.)
#
# MANAGE
#   tail -f <logfile>                    # follow progress (path printed on launch)
#   kill $(cat logs/<job>.pid)           # stop the job
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p logs
job="${1:-}"; [ -n "$job" ] || { grep -E '^#( |$)' "$0" | sed 's/^# \{0,1\}//'; exit 2; }
shift || true
ts="$(date +%Y%m%d-%H%M%S)"

case "$job" in
  validation) name=oblique_validation; cmd=(bash run_oblique_validation.sh) ;;
  diagnosis)  name=oblique_diagnosis;  cmd=(bash run_oblique_diagnosis.sh) ;;
  suite)      name=airflow_suite
              # run_airflow_validation.sh takes an optional bare arch arg (default sm_86)
              # and runs mode "all"; extra args after `suite` are forwarded to it.
              cmd=(bash run_airflow_validation.sh "$@") ;;
  --)         name="job"; cmd=("$@") ;;
  *) echo "unknown job '$job' (use: validation | diagnosis | suite | -- <cmd>)"; exit 2 ;;
esac
[ "${#cmd[@]}" -gt 0 ] || { echo "no command to run"; exit 2; }

log="logs/${name}-${ts}.log"; pidf="logs/${name}.pid"

# Detach fully: new session (setsid), ignore SIGHUP (nohup), no stdin (</dev/null).
setsid nohup "${cmd[@]}" </dev/null >"$log" 2>&1 &
pid=$!
echo "$pid" > "$pidf"

echo "launched: ${cmd[*]}"
echo "  pid    : $pid   (also in $pidf)"
echo "  log    : $log"
echo "  follow : tail -f $log"
echo "  stop   : kill $pid   # or: kill \$(cat $pidf)"
echo "Safe to close this SSH session now."
