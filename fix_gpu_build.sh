#!/usr/bin/env bash
# fix_gpu_build.sh — make the nvcc/GPU build compile, in place, on THIS machine.
#
# The GPU error `std_function.h: parameter packs not expanded` comes from nvcc's
# cudafe++ front-end choking on std::function when it parses lbm_gpu.h. This script
# guards std::function from nvcc (#ifndef __CUDACC__) directly in the two headers
# that are actually in this directory — so it fixes the files being compiled,
# regardless of any scp / copy / stale-file confusion. It is idempotent (safe to
# run repeatedly) and a no-op if the headers already avoid std::function.
#
# USAGE:  run it IN THE DIRECTORY YOU BUILD IN (where run_forward_city.sh lives):
#     bash fix_gpu_build.sh            # patch + clear object cache
#     bash fix_gpu_build.sh --build    # patch, clear cache, then rebuild on GPU
set -euo pipefail
cd "$(dirname "$0")"

for f in lbm_gpu.h lbm_solver.h; do
  [ -f "$f" ] || { echo "[fix] ERROR: $f is not in $(pwd) — cd into your build directory first."; exit 1; }
done
echo "[fix] build directory: $(pwd)"

python3 - <<'PY'
import re
def patch(path):
    lines = open(path).read().split('\n')
    out = []; changed = False
    for ln in lines:
        strip = ln.lstrip()
        is_comment = strip.startswith('//') or strip.startswith('*')
        needs = (not is_comment) and (
            ('std::function' in ln) or bool(re.match(r'\s*#\s*include\s*<functional>\s*$', ln)))
        already = bool(out) and out[-1].strip() == '#ifndef __CUDACC__'
        if needs and not already and '#ifndef' not in ln and '#endif' not in ln:
            out += ['#ifndef __CUDACC__', ln, '#endif']; changed = True
        else:
            out.append(ln)
    if changed:
        open(path, 'w').write('\n'.join(out)); print(f"[fix] guarded std::function in {path}")
    else:
        print(f"[fix] {path}: nothing to guard (already nvcc-safe)")
for p in ('lbm_gpu.h', 'lbm_solver.h'):
    patch(p)
PY

echo "[fix] clearing cached objects so nothing stale is reused"
rm -rf .obj_cell* *.o 2>/dev/null || true

if [ "${1:-}" = "--build" ]; then
  echo "[fix] rebuilding on GPU..."
  CELL="${CELL:-2.0}" bash run_forward_city.sh
else
  echo "[fix] done. Now build:   CELL=2.0 bash run_forward_city.sh"
fi
