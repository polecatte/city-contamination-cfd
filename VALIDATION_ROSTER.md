# Validation Roster — Urban LBM Wind & Dispersion Model

A living checklist of verification and validation work for the airflow and dispersion
models. Update the **Status** column as items land. The distinction that organizes this
file: **verification** = "the code solves the equations correctly" (analytical /
self-consistency), **validation** = "the model reproduces reality" (comparison to measured
data). The suite today is strong on verification and lighter on validation-against-data,
and the dispersion side is thinner than the airflow side.

**Status legend:** ✅ done & passing · 🟡 implemented, sanity-band only (not yet a
quantitative dataset match) · ⏳ queued (implemented, run pending) · ⬜ not yet built.

---

## Run-now confidence pass

For "is the model working" confidence without new development, run the existing battery:

```bash
ARCH=-arch=sm_86 CCBIN=g++-10 bash launch.sh suite        # all T_* tests, detached
ARCH=-arch=sm_86 CCBIN=g++-10 H=8 SMOKE=0 bash launch.sh validation   # oblique ladder
```

Beyond pass/fail, eyeball two things: that `T_resolution`'s Xr/H is asymptoting (not still
drifting at H=32), and that `T_diffusion`'s D_eff is physically sensible against `u*·κ·z`
(not merely above the numerical floor). That plus the oblique ladder is a fair
"core machinery is sound" bar.

---

## Airflow model

### Current

| Test | What it checks | Tolerance / band | Status |
|---|---|---|---|
| `test_poiseuille` | Analytical channel: velocity vs parabola | analytical match | ✅ |
| `T_stability` | No NaN/blow-up across configs | finite | ✅ |
| `T_determinism` | Bitwise/stat reproducibility | reproducible | ✅ |
| `T_resolution` | Grid convergence of Xr/H (H=16/24/32) | Xr/H asymptotes | 🟡 tabulated, add asymptote assertion |
| `T_wale` | WALE SGS: ν_t≈0 in pure shear, ≫ in wake | channel ≲1e-3·UH; wake/channel ≫10 | ✅ |
| `T_abl` | ABL streamwise homogeneity (aligned) | inlet→outlet drift <10% | ✅ |
| `T_mass` | Mass conservation over wind sweep {0,15,30,45}° | closure <2%, divnorm <5% | ✅ (oblique-gated) |
| **Oblique ladder** | Homogeneity (streamwise/lateral), rotation invariance, incompressibility | sw/lat <10%, rot <10%, div <5% | ✅ H=8; ⏳ **H=16 queued** |
| `T_cube` | Reattachment length + reversal features | Xr/H ∈ [1.0, 2.5] | 🟡 sanity band |
| `T_cp` | 4-face Cp | windward +0.6..0.8, leeward −0.2..−0.4, side suction | 🟡 sanity band, 4 points |
| `T_reynolds` | Re-independence at high Re | quantity plateaus | ✅ |
| `T_lateral` | Lateral BC behavior | — | ✅ |
| `T_shell` | Voxelization / shell cells | — | ✅ |

### Pending (for the paper)

| Item | Target dataset / reference | Proposed tolerance | Priority | Status |
|---|---|---|---|---|
| **H=16 oblique production run** | (self) | same as H=8 ladder | run | ⏳ queued |
| **Quantitative cube benchmark** (`cubebench`) | Silsoe cube / Martinuzzi & Tropea; Castro & Robins | windward peak >0.3; roof/side <−0.2; Xr/H∈[1.0,2.3] | high | 🟡 implemented, sanity-banded |
| **Inflow turbulence statistics** (`inflow`) | Panofsky & Dutton σ ratios; RFG integral scale | RMS σ-dev <15%; L_int/L_turb∈[0.3,3] | high | ✅ implemented & **passing** (CPU-verified: σ→1%, L_int/L=1.56) |
| **Street canyon** (`canyon`) | Oke 1988 skimming regime (W/H=1) | near-ground reversed + near-roof forward | high | 🟡 implemented, regime check |
| **Grid convergence / GCI** (`gridconv`) | Roache 1994 (Xr/H, r=1.5, H=16/24/36) | monotonic + GCI reported | med | 🟡 implemented, runs finite |
| Vortex-shedding Strouhal | literature St for the bluff body | St within ~15% | low (if unsteady claimed) | ⬜ |

---

## Dispersion model

### Current

| Test | What it checks | Tolerance / band | Status |
|---|---|---|---|
| `T_massbudget` | Scalar conservation: emitted = deposited + airborne + outflux (passive + depositing) | ≤2% creation | ✅ |
| `T_diffusion` | Effective crosswind diffusivity from plume spread, σ_y²=2·D_eff·x/U | D_eff ≫ numerical floor | 🟡 vs floor only, not vs physical K |

### Pending (for the paper)

| Item | Target / reference | Proposed tolerance | Priority | Status |
|---|---|---|---|---|
| **Analytical Gaussian plume** | uniform-flow point source, theory | centerline decay & σ_y,σ_z within ~15%; D_eff = K_turb via Sc_t | high | ⬜ |
| **Dispersion dataset + metrics** | CEDVAL C1/B1, MUST, or Thompson/EPA point source | FAC2 ≥0.5, \|FB\| ≤0.3, NMSE ≤4 (Chang & Hanna) | high | ⬜ |
| **Settling / deposition correctness** | Stokes terminal velocity; deposition-velocity model | terminal v within ~10%; v_d within model | med | ⬜ |
| **Numerical-diffusion bound** | grid/Peclet study; concentration grid-convergence | numerical D ≪ physical D; C converges | med | ⬜ |
| **Sc_t sensitivity** | sweep Sc_t (0.3–1.0) | document plume-spread sensitivity | med | ⬜ |
| Fluctuating concentration | peak/RMS c, averaging-time convergence | exposure/odor use-cases | low | ⬜ |

---

## Cross-cutting

| Item | Note | Status |
|---|---|---|
| Grid convergence — velocity | `T_resolution` (Xr/H) | 🟡 add asymptote check |
| Grid convergence — concentration | concentration converges slower than velocity; needs its own study | ⬜ |
| Averaging-time convergence | mean flow shown stationary (Δ<3e-3) in oblique ladder; add for concentration statistics | 🟡 flow only |

---

## Top three that most move a reviewer

1. **Quantitative cube benchmark** (Silsoe / Martinuzzi) — replaces the sanity band with a dataset match.
2. **Inflow turbulence statistics** — validates the second-order quantities that drive dispersion.
3. **One dispersion dataset scored with FAC2/FB/NMSE** — the formal bar for the dispersion model.

## Notes

- Tolerances marked "proposed" are starting points; tighten once a reference case is chosen.
- Oblique-wind BC fix (two-inlet per-face) validated at H=8 (all rungs pass; legacy path
  demonstrably broken at the same conditions) — see `OBLIQUE_DIVERGENCE_DIAGNOSIS.md`.
