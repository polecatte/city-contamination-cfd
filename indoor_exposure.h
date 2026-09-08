#pragma once
// indoor_exposure.h — Indoor occupant exposure for the SOLID-building regime.
//
// With buildings voxelized solid (voxelize(..., solid_buildings=true)), occupants
// are not in the fluid grid. This stage computes each building's indoor air
// concentration from the OUTDOOR concentration sampled at its facade, via the
// infiltration + filtration model (infiltration.h):
//
//     C_in(building) = (C_in/C_out) * C_out(facade)
//
// and the occupant-weighted indoor exposure sum(occupants * C_in). This is the
// piece that feeds the airborne inhalation exposure term in exposure_objective.py
// (deposited-surface exposure continues to use the deposition field). It replaces the
// uncalibrated permeable-shell pathway with a calibratable, literature-grounded
// model (Liu & Nazaroff 2001; Chen & Zhao 2011; Nazaroff 2004).

#include "city_builder7.h"
#include "infiltration.h"
#include <vector>
#include <cmath>

namespace city {

struct IndoorScenario {
    double dp_m       = 0.5e-6; // representative diameter (MASS-MEAN, not the PM
                                // cutoff — cutoff underestimates class infiltration)
    double a_inf      = 0.5;    // infiltration air-exchange rate (1/h)
    double a_mech     = 0.0;    // mechanical outdoor-air supply (1/h)
    double eta_mv     = 0.0;    // mechanical-supply filter efficiency
    double lambda_filt= 0.0;    // indoor filtration loss (1/h): CADR/V or eta*recirc
};

struct BuildingExposure {
    int    idx; double occupants, C_out, io, C_in, exposure;
};

struct IndoorExposureResult {
    std::vector<BuildingExposure> per_building;
    double total_occupants = 0, total_indoor_exposure = 0,
           total_outdoor_exposure = 0, io = 0;
};

// conc: outdoor concentration (or TIAC) field at a representative height,
// row-major [ny*nx] (e.g. concentration_z1.bin written by the solver).
inline IndoorExposureResult indoor_exposure(
        const Result& r, const std::vector<float>& conc, int nx, int ny,
        const IndoorScenario& sc) {
    IndoorExposureResult out{};
    auto at = [&](int x, int y) -> double {
        if (x < 0 || y < 0 || x >= nx || y >= ny) return 0.0;
        return (double)conc[(size_t)y * nx + x];
    };
    // I/O ratio is set by particle size + ventilation + filtration (uniform
    // across buildings here); spatial variation of exposure comes from the
    // local facade concentration and the occupant count.
    out.io = infil::indoor_io_ratio(sc.dp_m, sc.a_inf, sc.lambda_filt,
                                    sc.a_mech, sc.eta_mv);
    for (size_t bi = 0; bi < r.blocks.size(); ++bi) {
        const Block& b = r.blocks[bi];
        if (b.usage == PARK) continue;
        double occ = b.eff_inh;
        if (occ <= 0) continue;
        // average outdoor concentration over the one-cell ring around the
        // building envelope (facade-adjacent fluid cells)
        int x0 = b.bx0 - 1, x1 = b.bx1, y0 = b.by0 - 1, y1 = b.by1;
        double s = 0; int n = 0;
        for (int x = x0; x <= x1; ++x) { s += at(x, y0); s += at(x, y1); n += 2; }
        for (int y = y0; y <= y1; ++y) { s += at(x0, y); s += at(x1, y); n += 2; }
        double Cout = (n > 0) ? s / n : 0.0;
        double Cin = out.io * Cout;
        out.per_building.push_back({(int)bi, occ, Cout, out.io, Cin, occ * Cin});
        out.total_occupants        += occ;
        out.total_indoor_exposure  += occ * Cin;
        out.total_outdoor_exposure += occ * Cout;
    }
    return out;
}

} // namespace city
