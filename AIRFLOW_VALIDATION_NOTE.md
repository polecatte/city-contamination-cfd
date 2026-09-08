# Overnight airflow-model validation — design & rationale

**Goal.** Find where the airflow model (D3Q19 MRT‑LBM + WALE LES, single‑face
sheared‑ABL inlet, zero‑gradient lateral/top boundaries, permeable‑SHELL
buildings) disagrees with physical reality. Not "does it run" — *does it produce
the right flow*. The suite is built to be run unattended on the A4000 and to emit
a pass/fail against published reference values, so a disagreement is a number, not
an impression.

The driver is `airflow_validation.cpp`; the analysis is `airflow_validation.py`;
the overnight runner is `run_airflow_validation.sh`. **All of it ships UNTESTED —
there is no GPU/nvcc in the authoring container** — so treat the first run as a
shakedown and sanity‑check the cheap tests (T1–T3) before trusting T4–T6.

---

## What "consistent with reality" means here

Two separate things, tested in order — a code that fails verification can't be
validated:

* **Verification** (are we solving the equations correctly?): mass conservation,
  incompressibility, an analytical benchmark with a known closed‑form answer,
  steady‑state convergence.
* **Validation** (are these the right equations for urban wind?): does the
  evolved flow reproduce wind‑tunnel/guideline behaviour of bluff‑body wakes,
  boundary layers, and turbulence.

## This model's *specific* suspected failure modes

Generic CFD checklists waste overnight hours. These are the failure modes that
*this* architecture is most likely to exhibit, and each test below targets one:

1. **Lateral/top zero‑gradient boundaries** (`lbm_kernels_cpu.cpp` clamped
   pull‑streaming) can reflect or block cross‑flow, contaminating the near‑building
   field — and there is no fresh ABL inflow on ±y, so oblique wind has no y‑fetch.
   → **T1 (mass closure, incl. oblique), T5 (lateral confinement).**
2. **ABL horizontal homogeneity.** The single most cited urban‑CFD failure
   (Blocken et al. 2007): the imposed inlet profile decays before it reaches the
   buildings because inlet, wall treatment, and turbulence model are inconsistent.
   The inlet *generator* is validated offline (`test_abl_inlet.cpp`); whether the
   LBM *maintains* the profile downstream is not. → **T2.**
3. **WALE behaving like Smagorinsky.** WALE's defining property (Nicoud & Ducros
   1999) is that ν_t → 0 in pure shear and ~y³ at walls; a common implementation
   bug leaves ν_t spuriously large in shear, over‑damping the flow. → **T3.**
4. **Bluff‑body wake wrong length.** The core urban quantity. Too‑diffusive
   schemes or excess ν_t shorten/lengthen the recirculation behind a building
   away from the measured ~1.4–2.5 H. → **T4.**
5. **Low effective Reynolds number.** The lattice viscosity floor puts the sim at
   Re ≈ U·H/ν_eff ~ 10²–10³, far below the atmosphere (~10⁷). Sharp‑edged
   separation is Re‑independent, so the *gross* wake is defensible, but
   reattachment and wake turbulence are Re‑sensitive at low Re. This is a real
   model‑reality gap to *quantify*, not assume. → **T6.**

---

## The suite

Each test: hypothesis it falsifies → setup → diagnostic → reference value (cited)
→ pass band. Tiered cheap→expensive so a failure aborts early.

### T1 — Mass conservation & incompressibility  *(verification, cheap)*
* **Setup.** Single solid cube (GROUND, H≈20 cells) in a COST‑732 domain;
  uniform then ABL inlet; wind 0° and (for the oblique check) 15/30/45°.
* **Diagnostic.** Net boundary volume flux ∮u·n over all six faces, normalised by
  inflow Q_in; and volume‑RMS of the incompressibility residual |∇·u|·Δx/U_inlet.
* **Reference.** Steady incompressible flow ⇒ ∮u·n = 0 and ∇·u = 0; LBM is weakly
  compressible so the Mach proxy U_LU/c_s should stay <0.1 (Krüger et al. 2017,
  *The Lattice Boltzmann Method*).
* **Pass.** |∮u·n|/Q_in < 1 %, RMS|∇·u|·Δx/U < 2 %. **Oblique sub‑test is the gate
  for activating `wind_direction`**: if closure degrades past 0° the lateral BC is
  starving the y‑fetch (the staged‑parameter caveat).

### T2 — ABL horizontal homogeneity  *(validation, cheap)*
* **Setup.** Empty rough domain, `inlet_profile=1`, long fetch (~30 H_ref).
* **Diagnostic.** Mean U(z) and inferred u\* at inlet, mid, and outlet stations;
  drift between stations.
* **Reference.** A consistent neutral ABL (Richards & Hoxey 1993, JWEIA 46‑47:145)
  must advect with *no* streamwise change in an empty domain; sustained profiles
  are the benchmark, decay is the classic artifact (Blocken, Stathopoulos &
  Carmeliet 2007, Atmos. Environ. 41:238).
* **Pass.** max |ΔU(z)|/U_ref < 10 % and |Δu\*|/u\* < 10 % over the fetch.

### T3 — WALE eddy‑viscosity sanity  *(turbulence model, cheap)*
* **Setup.** (a) Fully‑developed Poiseuille channel (pure shear); (b) the T4 cube.
* **Diagnostic.** Normalised peak ν_t (from `copy_mean_flow_to_host`) in each.
* **Reference.** WALE (Nicoud & Ducros 1999, Flow Turbul. Combust. 62:183) gives
  ν_t ≈ 0 in pure shear, O(1–100)·ν_mol in genuine 3‑D wake turbulence.
* **Pass.** ν_t,channel / (U·H) ≲ 1e‑3 **and** ν_t,wake / ν_t,channel ≫ 10. The
  ratio is the discriminator and needs no absolute ν calibration.

### T4 — Surface‑mounted cube wake  *(the key validation, moderate)*
* **Setup.** Solid cube, H≈20 cells, ABL inlet, AIJ domain (5H up / 15H down /
  ≥5H lateral / ≥5H top).
* **Diagnostic.** Mean reattachment length X_r/H = distance from the leeward face
  to where near‑ground centreline u_x returns positive; plus presence/length of a
  roof separation bubble and an upstream stagnation/horseshoe reversal.
* **Reference.** Wall‑mounted cube X_r/H ≈ 1.4–2.5 (Martinuzzi & Tropea 1993,
  J. Fluids Eng. 115:85; AIJ benchmark, Tominaga et al. 2008, JWEIA 96:1749;
  Yoshie et al. 2007, JWEIA 95:1551). RANS k‑ε over‑predicts to ~2.5–3 H from
  stagnation‑point TKE; LES/WALE should land ~1.4–1.8 H. Roof flow over a cube
  typically does **not** reattach.
* **Pass.** X_r/H ∈ [1.0, 2.5]; reversed near‑ground flow present in the wake;
  upstream base reversal present. Outside [0.5, 4] ⇒ a real problem (over‑diffusion
  or wrong separation).

### T5 — Lateral‑boundary confinement  *(BC artifact, moderate)*
* **Setup.** The T4 cube at lateral clearance 3 H, 5 H, 8 H.
* **Diagnostic.** X_r/H vs clearance, and max |u| adjacent to the ±y boundary.
* **Reference.** With ≥5 H lateral clearance and blockage <3 %, the result must be
  insensitive to further clearance (COST 732; Franke et al. 2007). Persistent
  drift ⇒ the zero‑gradient ±y BC is contaminating the field.
* **Pass.** |X_r(8H) − X_r(5H)| / X_r(5H) < 5 %; near‑boundary |u| not anomalously
  high vs free stream.

### T6 — Effective Reynolds number & Re‑sensitivity  *(scale gap, moderate)*
* **Setup.** The T4 cube at two viscosity floors (τ=0.515 and 0.530).
* **Diagnostic.** Report Re_eff = U·H/ν_eff for each; measure ΔX_r/H between them.
* **Reference.** Sharp‑edged bluff‑body flow is ~Re‑independent above Re~10⁴
  (Cook 1990, *The designer's guide to wind loading*; Hoxey et al.). At the
  sim's Re~10²–10³ the gross wake should still be roughly fixed by the edges;
  strong Re‑sensitivity flags an under‑resolved‑turbulence regime.
* **Pass (interpretive).** Report Re_eff explicitly; flag if |ΔX_r/H| > 25 %
  between the two τ. This test *documents* a known limitation rather than gating.

### T7 — Cube‑face pressure coefficients  *(validation, moderate)*
Enabled by the new `copy_density_to_host` accessor (mean density → mean pressure
p = c_s²·ρ). On the T4 cube, Cp = (p−p_ref)/(½ρ_ref U_ref²) on the windward,
leeward, roof, and side faces.
* **Reference.** Silsoe cube (Richards, Hoxey & Short 2001, JWEIA 89:1553):
  windward stagnation Cp ≈ +0.6–0.8, leeward ≈ −0.2 to −0.4, roof/side separation
  suction ≈ −0.6 to −1.0.
* **Pass.** windward ∈ [0.4, 1.0]; leeward ∈ [−0.6, 0]; roof and side < −0.2.

### T8 — Permeable‑SHELL cube  *(the configuration actually optimised, moderate)*
Production buildings are permeable SHELL, not solid. Repeat the cube as SHELL at
perm = 0.005 (production office envelope) and 0.05 (leakier).
* **Reference.** No single closed value, so this is a *consistency* test: a porous
  bluff body sheds the wake deficit through the envelope, so X_r must **shorten
  monotonically** as permeability rises, and at the near‑impermeable perm=0.005 it
  must sit **close to the solid** result.
* **Pass.** |X_r(0.005) − X_r(solid)|/X_r(solid) < 25 %; X_r(0.05) ≤ X_r(0.005).

---

## Analytical‑solution tests (closed‑form verification)

### A1 — Poiseuille channel
Already implemented rigorously in `test_poiseuille.cpp`: developed profile vs the
analytical parabola and recovery of the input lattice viscosity ν = c_s²(τ−½).
The runner builds and runs it; its output lands in `av_run.log`.

### A2 — Blasius flat‑plate boundary layer  *(new)*
Uniform inlet over a no‑slip plate; WALE gives ν_t≈0 so the laminar BL should be
Blasius. Two **viscosity‑independent** signatures are checked, so no ν calibration
is needed:
* δ99 must grow ∝ x^0.5 (fit the log–log slope);
* shape factor H = δ\*/θ → 2.59 (Schlichting, *Boundary‑Layer Theory*).
* **Pass.** exponent ∈ [0.40, 0.60]; H ∈ [2.2, 3.0].

### Not feasible without new solver features (stated honestly)
* **Taylor–Green vortex** (the gold‑standard bulk‑viscosity decay test, u∝e^{−2νk²t})
  needs **periodic boundaries and a prescribed initial field**; this solver is
  inlet‑driven with no periodic/IC mode. Worth adding as a dedicated verification
  harness.
* **Couette flow** needs a **moving wall**; walls here are no‑slip bounce‑back only.

---

## Sizing for one overnight on the A4000 (16 GB)

LBM footprint ≈ 2×19 distributions + ~10 fields ≈ 200 B/cell.
* T4/T5 cube, H=20: AIJ box ≈ 420×220×120 ≈ 11 M cells ≈ 2.2 GB — comfortable;
  could refine to H=30 if time allows.
* T2 ABL fetch ≈ 200×40×60; T3 channel ≈ 256×4×8 — negligible.
* T1 reuses the T4 grid.
Total ≈ 6–10 warm‑ups to steady state. At a few hundred MLUPS each run is minutes
to ~1 h; the suite fits an 8–10 h window with margin.

## Running
```
bash run_airflow_validation.sh        # nohup+disown; writes av_*.csv/.log
python3 airflow_validation.py         # reads outputs, prints PASS/FAIL table
```

## What this suite cannot catch (honest limits)
* It validates the **mean** flow, mean pressure, and ν_t. Unsteady wake statistics
  (shedding spectra, TKE budgets) are out of scope here — the production path
  freezes the mean anyway, which is itself a modelling choice worth a separate test.
* The mean‑density accessor is **new and untested** (CUDA path mirrored line‑for‑line
  from the velocity accumulator but not run); sanity‑check that T7 Cp values are
  finite and O(1) before trusting them.
* Reference bands are wind‑tunnel/guideline ranges, not a single truth; treat a
  near‑miss as "investigate," not "fail."
* Taylor–Green and Couette are out of reach without periodic/moving‑wall BCs (above).
