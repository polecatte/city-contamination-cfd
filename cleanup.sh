#!/usr/bin/env bash
# cleanup.sh — declutter urban_openlbm: organize tests, quarantine dead files.
#
# Generated for the OpenLB-migration cleanup. Decisions applied:
#   • OLD ENGINE KEPT AS REFERENCE — the LBM solver, kernels, GPU, adjoint, abl_inlet,
#     main_cpu, forward_city and their build/run scripts are LEFT IN PLACE (OpenLB isn't
#     passing its gates yet, so they're still the only working engine).
#   • Tests + testing scripts → tests/
#   • Dead build artifacts (compiled binaries) + superseded docs → _to_delete/
#     (_to_delete/ is a quarantine you empty yourself — nothing is hard-deleted here.)
#
# Safe to re-run: each file is moved only if it still exists at the source. Uses
# `git mv` when inside a git work tree (preserves history), else plain `mv`.
#
# Run from the project root:   bash cleanup.sh

set -u
cd "$(dirname "$0")" || exit 1

USE_GIT=0
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then USE_GIT=1; fi

mkdir -p tests _to_delete
moved=0; skipped=0

mv1() { # mv1 <src> <destdir>
  local src="$1" dst="$2"
  if [ -e "$src" ]; then
    if [ "$USE_GIT" = 1 ] && git ls-files --error-unmatch "$src" >/dev/null 2>&1; then
      git mv -f "$src" "$dst/" && { echo "  git mv  $src -> $dst/"; moved=$((moved+1)); }
    else
      mv -f "$src" "$dst/" && { echo "  mv      $src -> $dst/"; moved=$((moved+1)); }
    fi
  else
    skipped=$((skipped+1))
  fi
}

echo "== moving tests + testing scripts -> tests/ =="
# C++ tests / validations / diagnostics / demos
for f in \
  test_poiseuille.cpp test_deposition.cpp test_psd_dep.cpp test_abl_inlet.cpp test_abl_run.cpp \
  adjoint_recip_test.cpp airflow_validation.cpp plume_validation.cpp oblique_validation.cpp \
  oblique_divergence_test.cpp overshoot_test.cpp diffusion_compare.cpp must_benchmark.cpp \
  must_score.cpp canopy_calib.cpp robustness_overnight.cpp ranking_stability.cpp \
  check_examples.cpp probe.cpp show_bins.cpp size_domains.cpp demo_indoor.cpp demo_infil.cpp \
  demo_occupancy.cpp polydisperse_demo.cpp lab_test.cpp abl_inlet_verify.cpp ; do
  mv1 "$f" tests
done
# Python tests / validation analysis / plotting
for f in \
  test_opt_logic.py adjoint_test.py airflow_validation.py airflow_validation_plots.py \
  lab_visualize.py sweep_audit.py exploit_scan.py verify_redundancy.py re_compare.py \
  plot_abl.py plot_deposition.py plot_diffusion.py plot_indoor.py plot_infil.py \
  plot_occupancy.py plot_psd.py plot_stability.py analyze_phase0.py viz_airflow.py \
  viz_borders.py render_material_map.py ; do
  mv1 "$f" tests
done
# Test / validation runner scripts
for f in \
  run_airflow_validation.sh run_forward_validation.sh run_oblique_validation.sh \
  run_oblique_diagnosis.sh run_labverify.sh package_labverify.sh run_stability_sweep.sh \
  re_test.sh run_overnight.sh ; do
  mv1 "$f" tests
done
# Verification output figures (test artifacts)
for f in abl_inlet_verify.png material_map.png ; do
  mv1 "$f" tests
done

echo "== quarantining dead build artifacts + superseded docs -> _to_delete/ =="
# Compiled binaries checked into the repo (regenerate from the .cpp when needed)
for f in gen_density_city gen_paramspace gen_rd_city gen_zoning_demo plume_validation ; do
  mv1 "$f" _to_delete
done
# Superseded docs: resolved-bug diagnoses + pre-OpenLB roadmaps (kept models' docs stay)
for f in \
  OBLIQUE_DIVERGENCE_DIAGNOSIS.md OVERNIGHT_DIAGNOSTIC_NOTE.md \
  TECHNICAL_STATUS_AND_ROADMAP.md IMPROVEMENT_PLAN.md ; do
  mv1 "$f" _to_delete
done

echo
echo "done: moved $moved, skipped-missing $skipped"
echo "review _to_delete/ then remove it yourself:  rm -rf _to_delete"
echo "NOTE: test sources compile from the project root with -I. (e.g. g++ -I. tests/test_poiseuille.cpp)."
echo "      A few old validation RUN scripts reference sources by bare name; if you rebuild them"
echo "      from tests/, add -I.. or run them from the project root."
