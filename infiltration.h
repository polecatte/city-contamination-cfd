#pragma once
// infiltration.h — Outdoor->indoor particle infiltration model.
//
// MOTIVATION. The current voxelizer makes buildings PERMEABLE shells (partial
// bounce-back, beta=0.005-0.02) so the LBM momentum field flows THROUGH them.
// Real buildings are aerodynamically solid; the porous-media treatment is
// physically justified only for vegetation/parks (Merlier, Jacob & Sagaut 2018,
// Atmos. Environ. 195, 89-103). The audit recommended: make buildings SOLID
// (full bounce-back -> correct wakes/canyons, per COST 732 / Tominaga et al.
// 2008) and account for indoor exposure with a SEPARATE, literature-grounded
// infiltration model rather than bulk-permeable flow.
//
// MODEL. With no indoor sources, the equilibrium indoor/outdoor ratio is the
// infiltration factor (Liu & Nazaroff 2001, Atmos. Environ. 35, 4451-4462;
// Chen & Zhao 2011, Atmos. Environ. 45, 275-288):
//
//       F_inf = (P * a) / (a + k)
//
//   P = penetration factor through the envelope (size-dependent, 0..1)
//   a = air-exchange rate (1/h)            [building leakage / ventilation]
//   k = indoor particle loss rate (1/h)    [deposition onto indoor surfaces]
//
// Measured F_inf for PM2.5 spans ~0.3-0.82 across the literature (Chen & Zhao
// 2011). Penetration P peaks near unity for accumulation-mode particles
// (~0.1-1 um) and falls for ultrafine (diffusional capture in cracks) and
// coarse (gravitational/impaction capture) particles (Liu & Nazaroff 2003,
// Aerosol Sci. Technol. 37, 565-573; Stephens & Siegel 2012, Indoor Air 22,
// 501-513; NRC 2016 review). The indoor deposition loss k is itself size- and
// surface-dependent (Lai & Nazaroff 2000, J. Aerosol Sci. 31, 463-476).
//
// USAGE IN THE EXPOSURE CHAIN. Buildings voxelized SOLID; occupants attached to
// each building. The exposure stage samples the outdoor (or facade) air
// concentration C_out and time-integrated air concentration around each
// building and forms the indoor exposure as F_inf * C_out. This decouples the
// (correct, solid) flow from the (separately parameterized, calibratable)
// infiltration, removing the uncalibrated permeable-shell knob the optimizer
// could exploit.

#include <cmath>
#include <algorithm>

namespace infil {

// Size-dependent envelope penetration factor P(d_p).
// Smooth log-normal-in-diameter bump peaking at ~0.3 um (P~0.95), with reduced
// penetration for ultrafine and coarse particles. Calibrated to the qualitative
// curves of Liu & Nazaroff (2003) and the size dependence summarized by
// Stephens & Siegel (2012); intended to be re-fit to a specific crack geometry.
inline double penetration_factor(double dp_m) {
    double dp_um = dp_m * 1e6;                       // metres -> microns
    if (dp_um <= 0) return 1.0;
    const double Pmax = 0.95;                        // near-unity for ideal size
    const double mu   = std::log(0.3);               // peak at ~0.3 um
    const double sig  = 1.1;                          // breadth in ln(dp)
    double z = (std::log(dp_um) - mu) / sig;
    double P = Pmax * std::exp(-0.5 * z * z);
    return std::min(1.0, std::max(0.0, P));
}

// Indoor deposition loss-rate k (1/h), crude size dependence: a U-shape with a
// minimum for accumulation-mode particles (Lai & Nazaroff 2000). Defaults are
// order-of-magnitude; expose for calibration/sensitivity.
inline double indoor_loss_rate(double dp_m) {
    double dp_um = std::max(1e-3, dp_m * 1e6);
    // diffusion-dominated (small) + gravity/impaction-dominated (large)
    double k = 0.10 + 0.05 / dp_um + 0.20 * dp_um;   // 1/h
    return k;
}

// Infiltration factor F_inf = P a / (a + k). a = air-exchange rate (1/h).
inline double infiltration_factor(double dp_m, double air_exchange_per_h,
                                  double indoor_loss_per_h = -1.0) {
    double P = penetration_factor(dp_m);
    double a = std::max(0.0, air_exchange_per_h);
    double k = (indoor_loss_per_h >= 0.0) ? indoor_loss_per_h : indoor_loss_rate(dp_m);
    if (a + k <= 0) return 0.0;
    return P * a / (a + k);
}

// ── Indoor air filtration (HVAC recirculation / portable air cleaner) ───────
// Filtration adds a first-order indoor removal rate lambda_filt (1/h). The
// steady indoor/outdoor ratio of the well-mixed indoor mass balance with no
// indoor sources (Nazaroff 2004, Indoor Air 14(s7):175-183; Chen & Zhao 2011)
// is:
//
//   C_in/C_out = ( P*a_inf + (1-eta_mv)*a_mech ) / ( a_inf + a_mech + k + lambda_filt )
//
//   a_inf      = infiltration air-exchange rate (unfiltered, penetration P)
//   a_mech     = mechanical outdoor-air supply (filtered at efficiency eta_mv)
//   lambda_filt= recirculating filtration loss; for a portable cleaner this is
//                CADR/V (Shaughnessy & Sextro 2006), for an HVAC recirculation
//                loop it is eta_filter * recirc_rate. HEPA eta ~ 0.999; a
//                MERV-13 unit delivers CADR ~ 200-700 m3/h.

// Convert a clean-air delivery rate (m3/h) and room volume (m3) to a loss rate (1/h).
inline double cadr_to_lambda(double cadr_m3ph, double room_vol_m3) {
    return (room_vol_m3 > 0.0) ? cadr_m3ph / room_vol_m3 : 0.0;
}

// Full indoor/outdoor ratio including filtration and mechanical ventilation.
// Defaults reduce to the plain infiltration factor (lambda_filt=a_mech=0).
inline double indoor_io_ratio(double dp_m, double a_inf_per_h,
                              double lambda_filt_per_h = 0.0,
                              double a_mech_per_h = 0.0, double eta_mv = 0.0,
                              double indoor_loss_per_h = -1.0) {
    double P = penetration_factor(dp_m);
    double k = (indoor_loss_per_h >= 0.0) ? indoor_loss_per_h : indoor_loss_rate(dp_m);
    double a_inf = std::max(0.0, a_inf_per_h);
    double a_mech = std::max(0.0, a_mech_per_h);
    double denom = a_inf + a_mech + k + std::max(0.0, lambda_filt_per_h);
    if (denom <= 0.0) return 0.0;
    return (P * a_inf + (1.0 - eta_mv) * a_mech) / denom;
}

} // namespace infil
