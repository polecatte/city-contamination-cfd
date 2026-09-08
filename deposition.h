#pragma once
// deposition.h — Size-dependent dry deposition velocity, replacing the constant
// per-surface DEP_* values with a physically-grounded, particle-size-resolved
// surface capture.
//
// Model: Zhang et al. (2001), Atmos. Environ. 35, 549-560, built on the Slinn
// (1982) vegetated-canopy framework. The surface (quasi-laminar) deposition
// velocity is
//   v_ds = eps0 * u* * (E_B + E_IM + E_IN) * R1     (collection by the surface)
// and the total deposition velocity adds gravitational settling in parallel:
//   v_d  = v_g + v_ds
// with collection efficiencies
//   E_B  = Sc^(-gamma)                  Brownian diffusion (Sc = nu/D)
//   E_IM = (St/(alpha+St))^2            inertial impaction
//   E_IN = 0.5 (d_p/A)^2                interception
//   R1   = exp(-sqrt(St))              particle rebound (sticking) factor
// The competition (E_B falls with size, E_IM/E_IN/v_g rise) yields the
// characteristic deposition-velocity MINIMUM in the accumulation mode
// (~0.1-1 um), confirmed across the literature (Petroff & Zhang 2010; Emerson
// et al. 2020 PNAS). Per-surface parameters {A, alpha, gamma} follow Zhang's
// land-use classes; mapping the project's usage types: PARK -> vegetation
// (small collector A -> strong interception/impaction, "parks scrub more"),
// buildings/ground -> urban/smooth (large A -> weak interception). This DERIVES
// DEP_PARK >> DEP_BIZ from physics instead of hand-set constants.
//
// In the LBM the resolved flow already carries particles through the turbulent
// surface layer (the aerodynamic resistance R_a), so the per-cell sink uses the
// SURFACE term v_ds (+v_g); u* is a representative friction velocity (ideally
// the local resolved near-wall shear; a scalar default is provided).

#include "city_builder7.h"   // Usage enum
#include <cmath>
#include <algorithm>

namespace dep {

// Land-use-like parameters per usage (Zhang et al. 2001, Table 3 style).
struct LUC { double A_m; double alpha; double gamma; };
inline LUC luc_for(city::Usage u) {
    switch (u) {
        case city::PARK:     return {0.002, 1.0, 0.56}; // vegetation: A~2 mm foliage
        case city::RES_LOW:  return {0.005, 1.2, 0.56}; // low-rise + some yard veg
        case city::RES_HIGH: return {0.010, 1.5, 0.56}; // urban
        case city::BUSINESS: return {0.010, 1.5, 0.56}; // urban smooth
        default:             return {0.010, 1.5, 0.56}; // ground/pavement
    }
}

// Gravitational settling velocity (Stokes + Cunningham); matches lbm_solver.cpp.
inline double settling_velocity(double dp, double rho_p) {
    if (dp <= 0) return 0.0;
    const double rho_air = 1.204, g = 9.81, lambda = 6.6e-8, nu = 1.5e-5;
    double mu = nu * rho_air;
    double Kn = 2.0 * lambda / dp;
    double Cc = 1.0 + Kn * (1.257 + 0.4 * std::exp(-1.1 / Kn));
    double w = (rho_p - rho_air) * g * dp * dp * Cc / (18.0 * mu);
    return std::max(0.0, w);
}

// Total dry deposition velocity v_d(d_p) for a usage/surface (m/s).
inline double deposition_velocity(city::Usage u, double dp, double rho_p,
                                  double u_star = 0.5) {
    if (dp <= 0) return 0.0;
    const double rho_air = 1.204, nu = 1.5e-5, g = 9.81;
    const double kB = 1.380649e-23, T = 293.0;
    double mu = nu * rho_air;
    double Kn = 2.0 * 6.6e-8 / dp;
    double Cc = 1.0 + Kn * (1.257 + 0.4 * std::exp(-1.1 / Kn));
    double D  = kB * T * Cc / (3.0 * M_PI * mu * dp);   // Brownian diffusivity
    double Sc = nu / D;
    double vg = settling_velocity(dp, rho_p);

    LUC p = luc_for(u);
    // Stokes number for vegetated collectors (Zhang 2001): St = vg*u*/(g*A)
    double St = vg * u_star / (g * p.A_m);
    double E_B  = std::pow(Sc, -p.gamma);
    double E_IM = std::pow(St / (p.alpha + St), 2.0);
    double E_IN = 0.5 * std::pow(dp / p.A_m, 2.0);
    double R1   = std::exp(-std::sqrt(std::max(0.0, St)));   // rebound/sticking
    const double eps0 = 3.0;
    double v_ds = eps0 * u_star * (E_B + E_IM + E_IN) * R1;  // surface capture
    return vg + v_ds;
}

} // namespace dep
