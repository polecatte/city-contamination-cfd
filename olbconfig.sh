#!/usr/bin/env bash
# olbconfig.sh — switch an OpenLB tree between CPU-only and GPU (CUDA) builds.
#
#   ./olbconfig.sh cpu     # serial CPU_SISD, g++     (Phases 1-4: port + geometry gates)
#   ./olbconfig.sh cpu-mt  # OpenMP CPU_SISD, g++     (Phases 5-7: many flow-through runs)
#   ./olbconfig.sh gpu     # single-GPU CUDA, nvcc    (Phase 8)
#   ./olbconfig.sh show    # print the active config
#   ./olbconfig.sh list    # list the templates this release ships
#
# Requires OLB_ROOT. Writes $OLB_ROOT/config.mk.
#
# DESIGN: this does NOT synthesise a config. OpenLB ships ~22 tested templates
# in config/, and they set variables beyond the obvious ones -- which rule file
# to include (default.mk vs default.mixed.mk), the separate flag set nvcc uses,
# and so on. Editing a config in place leaks state between modes: a CPU switch
# that leaves a GPU rule-file selection behind produces a CPU build that still
# invokes nvcc, with none of nvcc's flags set. So each mode REPLACES config.mk
# wholesale from a template, then patches only CUDA_ARCH.
#
# The pristine shipped config.mk is preserved as config.mk.orig on first run
# and is what `cpu` restores.

set -euo pipefail

MODE="${1:-show}"
: "${OLB_ROOT:?OLB_ROOT is not set}"
[ -d "$OLB_ROOT/src" ] || { echo "ERROR: no src/ under OLB_ROOT=$OLB_ROOT" >&2; exit 1; }

CFG="$OLB_ROOT/config.mk"
ORIG="$OLB_ROOT/config.mk.orig"
TDIR="$OLB_ROOT/config"

# Preserve the as-shipped config exactly once, before anything edits it.
if [ ! -f "$ORIG" ] && [ -f "$CFG" ]; then
  cp "$CFG" "$ORIG"
  echo "saved pristine config as config.mk.orig"
fi

backup() { if [ -f "$CFG" ]; then cp "$CFG" "$CFG.bak.$(date +%Y%m%d-%H%M%S)"; fi; }

# Rewrite a make variable: first uncommented assignment is replaced, any later
# duplicates of it are deleted, and an absent variable is appended.
setvar() {
  local f="$1" k="$2" v="$3"
  if grep -qE "^[[:space:]]*${k}[[:space:]]*[:?+]?=" "$f"; then
    sed -i -E "0,/^[[:space:]]*${k}[[:space:]]*[:?+]?=/s|^[[:space:]]*${k}[[:space:]]*[:?+]?=.*|${k} := ${v}|" "$f"
    sed -i -E "1,\$ { /^${k} := /!{ /^[[:space:]]*${k}[[:space:]]*[:?+]?=/d } }" "$f"
  else
    printf '%s := %s\n' "$k" "$v" >> "$f"
  fi
}

pick_template() {
  local pat="$1"
  [ -d "$TDIR" ] || return 1
  ls "$TDIR"/*.mk 2>/dev/null | grep -E "/(${pat})\.mk$" | head -1
}

show() {
  echo "OLB_ROOT = $OLB_ROOT"
  if [ -f "$CFG" ]; then
    echo "--- config.mk (uncommented) ---"
    grep -vE '^[[:space:]]*(#|$)' "$CFG"
    echo "--- sanity ---"
    if grep -qE '^[[:space:]]*[A-Z_]*CUDA' "$CFG"; then
      echo "CUDA vars present  : YES"
    else
      echo "CUDA vars present  : no"
    fi
    grep -qE '^[[:space:]]*PLATFORMS.*GPU_CUDA' "$CFG" \
      && echo "GPU_CUDA platform  : YES" || echo "GPU_CUDA platform  : no"
  else
    echo "no config.mk"
  fi
  echo "--- built libs ---"
  find "$OLB_ROOT" -name '*.a' -exec ls -lh {} \; 2>/dev/null || true
}

case "$MODE" in
  list)
    ls -1 "$TDIR"/*.mk 2>/dev/null | xargs -n1 basename || echo "no config/ directory"
    exit 0 ;;

  show) show; exit 0 ;;

  cpu)
    backup
    if [ -f "$ORIG" ]; then
      cp "$ORIG" "$CFG"
      echo "restored config.mk from pristine config.mk.orig"
    else
      t=$(pick_template 'cpu_gcc_openmpi') || t=""
      [ -n "$t" ] || { echo "ERROR: no pristine config and no cpu template found" >&2; exit 1; }
      cp "$t" "$CFG"
      echo "seeded config.mk from $(basename "$t")"
      setvar "$CFG" PARALLEL_MODE 'NONE'
      setvar "$CFG" MPIFLAGS ''
    fi
    # The pristine config is OpenLB's serial-gcc default; assert rather than assume.
    if grep -qE '^[[:space:]]*PLATFORMS.*GPU_CUDA' "$CFG"; then
      echo "WARNING: restored config still lists GPU_CUDA -- forcing CPU_SISD" >&2
      setvar "$CFG" PLATFORMS 'CPU_SISD'
    fi
    echo "configured: CPU (serial, g++)"
    ;;

  cpu-mt)
    backup
    # Shared-memory OpenMP on one box. Seeded from OpenLB's gcc CPU template for its flag
    # set, then de-MPI'd: that template sets CXX := mpic++ and PARALLEL_MODE := MPI, and
    # OMP mode needs neither (nor an MPI install). In 1.8 OMP parallelises each block's
    # collide loop over iX (core/platform/cpu/sisd/operator.h), so it scales on a
    # single-cuboid run; set OMP_NUM_THREADS to choose the thread count.
    t=$(pick_template 'cpu_gcc_openmpi') || t=""
    [ -n "$t" ] || { echo "ERROR: config/cpu_gcc_openmpi.mk not found" >&2; exit 1; }
    cp "$t" "$CFG"
    echo "seeded config.mk from $(basename "$t")"
    setvar "$CFG" CXX 'g++'
    setvar "$CFG" CC 'gcc'
    setvar "$CFG" PARALLEL_MODE 'OMP'
    setvar "$CFG" MPIFLAGS ''
    setvar "$CFG" OMPFLAGS '-fopenmp'
    setvar "$CFG" PLATFORMS 'CPU_SISD'
    grep -qE '^[[:space:]]*CXXFLAGS.*std=c\+\+20' "$CFG" \
      || echo "WARNING: CXXFLAGS in $(basename "$t") has no -std=c++20 -- check it" >&2
    echo "configured: CPU (OpenMP, g++) -- run with OMP_NUM_THREADS=\$(nproc)"
    ;;

  gpu)
    backup
    # Use OpenLB's own single-GPU template rather than synthesising one: it
    # carries the correct rule-file selection and nvcc flag set.
    t=$(pick_template 'gpu_only') || t=""
    if [ -z "$t" ]; then
      echo "ERROR: config/gpu_only.mk not found. Available:" >&2
      ls -1 "$TDIR"/*.mk 2>/dev/null | xargs -n1 basename >&2
      exit 1
    fi
    cp "$t" "$CFG"
    echo "seeded config.mk from $(basename "$t")"

    ARCH="${CUDA_ARCH:-}"                 # env override, e.g. CUDA_ARCH=86 for an RTX A4000
    if [ -z "$ARCH" ] && command -v nvidia-smi >/dev/null 2>&1; then
      ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
             | head -1 | tr -d ' .' || true)
    fi
    if [ -n "$ARCH" ]; then
      setvar "$CFG" CUDA_ARCH "$ARCH"
      echo "CUDA_ARCH := $ARCH (from nvidia-smi compute_cap)"
    else
      echo "WARNING: could not read compute_cap; leaving template's CUDA_ARCH as-is" >&2
    fi

    # urban_flow computes in double (typedef double T) and the CPU reference is double, so
    # the library is built double too; the shipped template says float.
    setvar "$CFG" FLOATING_POINT_TYPE 'double'
    # nvcc: $NVCC when given (e.g. a conda-forge CUDA 12.6, see lab_openlb.sh), else PATH.
    # CUDA 12.0 cannot compile OpenLB 1.8 at all (fields.h: consteval source_location).
    if [ -n "${NVCC:-}" ]; then setvar "$CFG" CXX "$NVCC"; setvar "$CFG" CC "$NVCC"; fi
    # host compiler for nvcc, when the default g++ is newer than this CUDA supports
    [ -n "${CUDA_HOST_CXX:-}" ] && echo "CXXFLAGS += -ccbin $CUDA_HOST_CXX" >> "$CFG"
    # OpenLB compiles with -rdc=true but links without an arch, so nvcc device-links for its
    # default sm_52: the binary then holds NO usable sm_$CUDA_ARCH kernels ("nvlink warning:
    # SM Arch ('sm_52') not found"). Device-link for the configured arch.
    echo 'LDFLAGS += --generate-code=arch=compute_$(CUDA_ARCH),code=sm_$(CUDA_ARCH)' >> "$CFG"
    # a non-system CUDA: find its libcudart at run time, not the distro's older one
    [ -n "${CUDA_LIBDIR:-}" ] && echo "LDFLAGS += -L$CUDA_LIBDIR -Xlinker -rpath=$CUDA_LIBDIR" >> "$CFG"

    # Every flag variable that must carry -std=c++20 -- OpenLB's headers hard
    # #error without it, and nvcc defaults to C++17.
    for v in CXXFLAGS CUDA_CXXFLAGS; do
      if grep -qE "^[[:space:]]*${v}[[:space:]]*[:?+]?=" "$CFG" \
         && ! grep -E "^[[:space:]]*${v}[[:space:]]*[:?+]?=" "$CFG" | grep -q 'std=c++'; then
        echo "WARNING: $v in $(basename "$t") has no -std=c++20 -- check it" >&2
      fi
    done
    echo "configured: GPU (single-GPU CUDA, from $(basename "$t"))"
    ;;

  *) echo "usage: $0 {cpu|cpu-mt|gpu|show|list}" >&2; exit 2 ;;
esac

echo
show
echo
echo "NEXT — platform changed, so stale objects must go:"
echo "  cd \$OLB_ROOT && make clean"
echo "  make -C external"
echo "  make -j\$(nproc) 2>&1 | tee ~/build_lib.log"
echo "Also clean any example you already built: make clean in its directory."
