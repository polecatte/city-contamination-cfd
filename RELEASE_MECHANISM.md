# The release mechanism (accidental burst)

The forward-dose regime models an **accidental release**: a fixed mass M of
contaminant enters the domain in a short burst, then no more, and is transported on
the live turbulent flow until it deposits or advects out. This document defines the
release profile, why it is shaped that way, and how it is discretised.

## 1. The idealised release: an impulse

The physical target is an instantaneous release — all mass M delivered at one
instant t=0:

    s(x,t) = M · ρ_s(x) · δ(t)

where ρ_s(x) is the spatial distribution of the source (a point, or a set of
candidate cells), and δ(t) is the Dirac delta in time. This is the impulse (Green's
function) release: the resulting dosage field is the transport operator's response
to a unit impulse, and every other time-profile is a time-convolution of it.

## 2. The discrete release: a finite pulse

A true δ(t) cannot be represented on a finite time step, so the mass is injected
uniformly over a short **pulse** of n steps:

    n = round(τ / Δt) ,     injected mass per step = M / n

so that Σ over the n steps returns exactly M (mass-conserving; verified). Setting
τ = 0 gives n = 1 — a single-step injection, the closest discrete approximation to
δ(t). The default is τ ≈ 2 s.

**Why a finite pulse rather than one step.** On a *live* turbulent flow, a
one-instant release commits the entire plume to the single eddy configuration
present at that instant, which is the noisiest possible profile. Spreading the
injection over a few seconds lets it sample a short range of turbulent states, which
trims the sharpest single-instant sensitivity at no cost to fidelity — provided the
pulse is short (§3).

## 3. Why ~2 s is still "instantaneous"

The relevant comparison is the pulse width τ against the flow's **eddy-turnover
time** — the timescale over which the near-source turbulence rearranges,
approximately

    t_eddy ≈ H / U

for building height H and local wind U. For an urban canopy (H ≈ 10–20 m,
U ≈ 2–4 m/s) this is ≈ 5–15 s. A 2 s pulse is a small fraction of one turnover, so
the release deposits into essentially one evolving eddy field — the plume's fate,
and the dose, are indistinguishable from the impulse. The release is therefore
**impulse-equivalent** and is reported as instantaneous.

The equivalence breaks only if τ approaches or exceeds t_eddy, at which point the
release genuinely samples distinct turbulent states and becomes a real
short-duration source (a different, more self-averaged object) — not an error, but
no longer an impulse. Keeping τ ≈ 2 s ≪ t_eddy avoids that regime.

## 4. What the pulse does — and does not — do for noise

The pulse width is a **second-order** noise control, not the primary one:

- A 2 s pulse against a ~10 s turnover averages over ~1/5 of an eddy cycle. It
  removes the sharpest "which instant did I release at" sensitivity.
- It does **not** average over the large-scale plume meander (does the plume take
  this street or that one) that plays out over many turnovers and dominates the
  run-to-run scatter.

That large-scale variability is removed only by the **ensemble over turbulence
seeds** (FORWARD_LIVE.md): N bursts, each with an independent RFG realisation,
averaged. So the pulse is a minor robustness gain layered on top of the ensemble,
which does the real variance reduction — do not expect the pulse to let you shrink N.

## 5. Self-termination

Because the source is off after the pulse and the contaminant is particulate
(deposition + Stokes settling active), the mass leaves: M = M_out + M_dep. Phase B
runs until the airborne fraction falls below clearance_frac (default 1% ⇒ 99%
cleared), then stops. This makes the cumulative dosage

    Θ(x) = ∫₀^∞ C(x,t) dt

**converge to a finite, window-independent value** — the key advantage of an
accidental (self-terminating) release over a continuous source, whose ∫C dt would
grow without bound and require an arbitrary cutoff. The dosage is dt-independent
(it is the time-integral of a conserved mass); only instantaneous *peak*
concentration would be dt-sensitive, and that is not the reported metric.

## 6. Implementation

`Config`: `burst_release` (on), `pulse_seconds` (τ, default 2.0; 0 ⇒ single step),
`clearance_frac` (0.01). Mechanism (solver Phase B, no kernel surgery): the source
rate is set to Q = M/(n·Δt) for the first n steps, then 0; airborne mass is polled
and the run terminates at 99% clearance. Mass accounting reports emitted =
M, deposited, airborne, outflow. `abl_seed` varies per ensemble member.

## 7. For the paper

State it as: *"An instantaneous accidental release of mass M is modelled as a finite
injection pulse of ~2 s (τ ≪ the canopy eddy-turnover time H/U ≈ 5–15 s, hence
impulse-equivalent), transported on the live flow until 99% of the mass has
deposited or advected from the domain. The population dose Θ = ∫C dt is thus
self-terminating and window-independent; per-release turbulent variability is
quantified by an ensemble of N independent realisations."* Report the pulse width,
the clearance threshold, and N.
