#pragma once
// psd.h — Polydisperse particle size distribution for the contaminant source.
//
// Real contaminant aerosols are NOT monosized. In aerosol science the
// airborne activity is described by a LOGNORMAL distribution parameterized by
// the Mass Median Aerodynamic Diameter (MMAD) and geometric standard
// deviation (GSD):
//     MMAD is the right descriptor when deposition is governed by sedimentation
//     and impaction (d > ~0.5 um); the thermodynamic diameter matters below that.
//   - Typical atmospheric aerosol defaults: MMAD ~ 1 um, GSD ~ 2 (measured
//     accidental-release size distributions span ~0.6 um fine plumes to ~6 um
//     coarse resuspension). See Hinds (1999), Aerosol Technology, 2nd ed., Wiley,
//     for lognormal size distributions and aerodynamic diameter.
//
// This module discretizes the lognormal mass distribution into N sections
// (the standard sectional approach; e.g. Sartelet/Polair3D uses the mean
// diameter of each section), returning per-bin mass fraction and a
// representative diameter. Each bin is then transported with its OWN settling
// velocity (Stokes+Cunningham) and size-dependent deposition (deposition.h),
// and the exposure is summed over bins weighted by mass fraction.
//
// Aerodynamic vs physical diameter: MMAD is aerodynamic (unit-density sphere
// with the same settling speed). The solver's settling uses the PHYSICAL
// diameter with the real density rho_p, so we convert
//   d_phys = d_ae * sqrt(rho0 / rho_p)   (rho0 = 1000 kg/m^3, shape factor chi=1)
// which reproduces the same terminal velocity (Hinds 1999, Aerosol Technology).

#include <cmath>
#include <vector>

namespace psd {

struct Bin {
    double d_ae;    // representative aerodynamic diameter (m)
    double d_phys;  // physical (geometric) diameter for the solver settling (m)
    double frac;    // mass fraction in this bin (sums to 1)
};

struct Lognormal {
    double MMAD;    // mass median aerodynamic diameter (m)
    double GSD;     // geometric standard deviation (>1)
    double rho_p;   // particle density (kg/m^3) for aero<->phys conversion
};

// Discretize into N equal-log-width bins over [MMAD/GSD^n_sig, MMAD*GSD^n_sig].
inline std::vector<Bin> discretize(const Lognormal& p, int N, double n_sig = 3.0) {
    const double rho0 = 1000.0;
    double lnmed = std::log(p.MMAD), lnsig = std::log(p.GSD);
    double lo = lnmed - n_sig * lnsig, hi = lnmed + n_sig * lnsig;
    auto Phi = [](double z) { return 0.5 * std::erfc(-z / std::sqrt(2.0)); }; // lognormal CDF
    std::vector<Bin> bins; bins.reserve(N);
    double total = 0.0;
    for (int i = 0; i < N; ++i) {
        double a = lo + (hi - lo) * i / N;
        double b = lo + (hi - lo) * (i + 1) / N;
        double frac = Phi((b - lnmed) / lnsig) - Phi((a - lnmed) / lnsig);
        double d_ae = std::exp(0.5 * (a + b));            // section mean diameter
        double d_phys = d_ae * std::sqrt(rho0 / p.rho_p);
        bins.push_back({d_ae, d_phys, frac});
        total += frac;
    }
    for (auto& b : bins) b.frac /= total;                 // renormalize truncated tails
    return bins;
}

} // namespace psd
