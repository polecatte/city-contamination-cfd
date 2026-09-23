#!/usr/bin/env bash
# olbconfig.sh — switch an OpenLB tree between CPU-only and GPU (CUDA) builds.
#
#   ./olbconfig.sh cpu     # CPU_SISD, g++            (Phases 1-7)
#   ./olbconfig.sh gpu     # CPU_SISD GPU_CUDA, nvcc  (Phase 8)
#   ./olbconfig.sh show    # print the active config
#
# Requires OLB_ROOT. Writes $OLB_ROOT/config.mk, backing up the previous one.
# Switching platforms invalidates every object file: run `make clean` after.

set -euo pipefail

MODE="${1:-show}"
: "${OLB_ROOT:?OLB_ROOT is not set}"
[ -d "$OLB_ROOT/src" ] || { echo "ERROR: no src/ under OLB_ROOT=$OLB_ROOT" >&2; exit 1; }

CFG="$OLB_ROOT/config.mk"

# set a make variable: rewrite it in place if present, append if not.
# Only matches uncommented assignments, so '# CXX := ...' is left alone.
setvar() {
  local f="$1" k="$2" v="$3"
  if grep -qE "^[[:space:]]*${k}[[:space:]]*[:?+]?=" "$f"; then
    sed -i -E "s|^[[:space:]]*${k}[[:space:]]*[:?+]?=.*|${k} := ${v}|" "$f"
  else
    printf '%s := %s\n' "$k" "$v" >> "$f"
  fi
}

getvar() { grep -E "^[[:space:]]*$2[[:space:]]*[:?+]?=" "$1" 2>/dev/null | tail -1 || true; }

show() {
  echo "OLB_ROOT = $OLB_ROOT"
  if [ -f "$CFG" ]; then
    echo "--- $CFG ---"
    grep -vE '^[[:space:]]*(#|$)' "$CFG"
  else
    echo "no config.mk"
  fi
  echo "--- built libs ---"
  find "$OLB_ROOT" -name '*.a' -exec ls -lh {} \; 2>/dev/null || echo "none"
}

# Start from an OpenLB-provided template so we inherit any variable this
# release needs that we don't explicitly set. Prefer a mode-matching template.
seed_config() {
  local want="$1" pick=""
  if [ -d "$OLB_ROOT/config" ]; then
    # shellcheck disable=SC2012
    pick=$(ls "$OLB_ROOT/config"/*.mk 2>/dev/null | grep -iE "$want" | head -1 || true)
    [ -n "$pick" ] || pick=$(ls "$OLB_ROOT/config"/*.mk 2>/dev/null | head -1 || true)
  fi
  if [ -f "$CFG" ]; then
    cp "$CFG" "$CFG.bak.$(date +%Y%m%d-%H%M%S)"
  fi
  if [ -n "$pick" ] && [ ! -f "$CFG" ]; then
    cp "$pick" "$CFG"
    echo "seeded config.mk from $(basename "$pick")"
  elif [ -n "$pick" ]; then
    echo "editing existing config.mk (templates available: $(ls "$OLB_ROOT/config"/*.mk 2>/dev/null | xargs -n1 basename | tr '\n' ' '))"
  elif [ ! -f "$CFG" ]; then
    : > "$CFG"
    echo "WARNING: no config/ templates found; writing config.mk from scratch" >&2
  fi
}

case "$MODE" in
  cpu)
    seed_config 'cpu|default|gcc'
    setvar "$CFG" CXX                 'g++'
    setvar "$CFG" CC                  'gcc'
    setvar "$CFG" CXXFLAGS            '-O3 -std=c++20'
    setvar "$CFG" PARALLEL_MODE       'NONE'
    setvar "$CFG" PLATFORMS           'CPU_SISD'
    setvar "$CFG" FLOATING_POINT_TYPE 'double'
    echo "configured: CPU (CPU_SISD, g++, double, serial)"
    ;;

  gpu)
    seed_config 'gpu|cuda'

    # --- CUDA arch from the actual card ---
    ARCH=""
    if command -v nvidia-smi >/dev/null 2>&1; then
      ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
             | head -1 | tr -d ' .' || true)
    fi
    if [ -z "$ARCH" ]; then
      ARCH=86
      echo "WARNING: could not read compute_cap from nvidia-smi; defaulting CUDA_ARCH=86" >&2
    fi

    # --- host compiler: nvcc rejects gcc newer than its supported max ---
    CCBIN=""
    if command -v nvcc >/dev/null 2>&1; then
      NVCC_VER=$(nvcc --version | grep -oE 'release [0-9]+\.[0-9]+' | awk '{print $2}' | head -1)
      GCC_MAJ=$(gcc -dumpversion | cut -d. -f1)
      NV_MAJ=${NVCC_VER%%.*}; NV_MIN=${NVCC_VER##*.}
      # CUDA 12.0-12.3 tops out around gcc 12; later 12.x raised it.
      MAXGCC=99
      if [ "$NV_MAJ" = "12" ] && [ "$NV_MIN" -le 3 ]; then MAXGCC=12; fi
      if [ "$NV_MAJ" = "11" ]; then MAXGCC=11; fi
      if [ "$GCC_MAJ" -gt "$MAXGCC" ]; then
        for c in g++-12 g++-11; do
          if command -v "$c" >/dev/null 2>&1; then CCBIN="$c"; break; fi
        done
        if [ -n "$CCBIN" ]; then
          echo "NOTE: gcc $GCC_MAJ > CUDA $NVCC_VER max ($MAXGCC) — using -ccbin $CCBIN"
        else
          echo "WARNING: gcc $GCC_MAJ exceeds CUDA $NVCC_VER's supported max ($MAXGCC)," >&2
          echo "         and no g++-12/g++-11 found. Install one (apt install g++-12)" >&2
          echo "         or a newer CUDA toolkit, or nvcc will refuse to compile." >&2
        fi
      fi
    else
      echo "WARNING: nvcc not on PATH" >&2
    fi

    FLAGS='-O3 -std=c++20 --forward-unknown-to-host-compiler'
    [ -n "$CCBIN" ] && FLAGS="$FLAGS -ccbin $CCBIN"

    setvar "$CFG" CXX                 'nvcc'
    setvar "$CFG" CC                  'nvcc'
    setvar "$CFG" CXXFLAGS            "$FLAGS"
    setvar "$CFG" CUDA_CXX            'nvcc'
    setvar "$CFG" CUDA_ARCH           "$ARCH"
    setvar "$CFG" PARALLEL_MODE       'NONE'
    setvar "$CFG" PLATFORMS           'CPU_SISD GPU_CUDA'
    setvar "$CFG" FLOATING_POINT_TYPE 'double'
    echo "configured: GPU (CPU_SISD GPU_CUDA, nvcc, sm_$ARCH, double, single-GPU)"
    ;;

  show) show; exit 0 ;;
  *) echo "usage: $0 {cpu|gpu|show}" >&2; exit 2 ;;
esac

echo
show
echo
echo "NEXT — platform changed, so stale objects must go:"
echo "  cd \$OLB_ROOT && make clean"
if [ "$MODE" = gpu ]; then
  echo "  make -C external CXX=nvcc CC=nvcc"
else
  echo "  make -C external"
fi
echo "  make -j\$(nproc) 2>&1 | tee ~/build_lib.log"
