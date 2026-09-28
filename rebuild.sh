#!/usr/bin/env bash
# rebuild.sh — pull the latest sources and rebuild the OpenLB app.
# The app directory holds symlinks into this repo, so a pull is all the update needed.
set -euo pipefail
: "${OLB_ROOT:?OLB_ROOT is not set}"
BRANCH="${BRANCH:-claude/gracious-lovelace-3wlllj}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP="$OLB_ROOT/examples/urban/urban_flow"

git -C "$REPO" pull origin "$BRANCH"
cd "$APP"
make 2>&1 | tee ~/gate4.log | head -"${LINES_OUT:-40}"
echo "--- error count: $(grep -c 'error:' ~/gate4.log) (full log: ~/gate4.log) ---"
