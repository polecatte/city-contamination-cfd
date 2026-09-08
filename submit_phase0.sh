#!/usr/bin/env bash
# submit_phase0.sh - submit Phase 0 with EVERY critical option on the sbatch
# command line. Command-line options override the script's #SBATCH directives and
# cannot be nullified by a prepended line or a parser quirk, so this works even
# when the embedded directives are being ignored.
#
# Usage:  ACCOUNT=your_alloc bash submit_phase0.sh
# Verify names first:  sinfo -o "%P %G %l"   (partition, gres, time cap)
set -euo pipefail

ACCOUNT=${ACCOUNT:-}                 # REQUIRED on ACES; export ACCOUNT=... before running
PARTITION=${PARTITION:-gpu}          # VERIFY with sinfo
GRES=${GRES:-gpu:h100:1}             # VERIFY with sinfo -o "%P %G"
TIME=${TIME:-12:00:00}
SCRIPT=${SCRIPT:-phase0_aces.slurm}

[ -n "$ACCOUNT" ] || { echo "set ACCOUNT (export ACCOUNT=your_alloc); find it with: myproject -l"; exit 1; }

sbatch \
  --job-name=lbm_phase0 \
  --time="$TIME" \
  --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=96G \
  --partition="$PARTITION" \
  --gres="$GRES" \
  --account="$ACCOUNT" \
  --output=phase0_%j.out --error=phase0_%j.err \
  "$SCRIPT"
