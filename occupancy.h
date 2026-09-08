#pragma once
// occupancy.h — Where people are, and when: the microenvironment time budget
// and the STREET (outdoor pedestrian/in-transit) inhabitance pathway.
//
// TIME BUDGET SOURCE. Klepeis et al. (2001), "The National Human Activity
// Pattern Survey (NHAPS)," J. Expo. Anal. Environ. Epidemiol. 11(3):231-252
// (n=9386, EPA-funded, built explicitly to feed population-exposure models).
// 24-h population-average microenvironment fractions:
//     residence indoors  0.69
//     other indoor       0.18   (workplace, stores, other buildings)
//     in-vehicle         0.055
//     outdoors           0.076
// Corroboration: Canadian CHAPS-2 (Matz et al. 2014): 88.9% indoor, 5.8%
// outdoor, 5.3% vehicle. Time-of-day breakdown: Tsang & Klepeis (1996), EPA
// Final Report EPA/600/R-96/148 (3-hour diary segments).
//
// EXPOSURE-PATHWAY MAPPING (see city_builder7.h for the building/park side):
//   residence  -> residential buildings   (indoor, infiltration-reduced)
//   other-indoor-> business/mixed buildings(indoor, infiltration-reduced)
//   outdoors(park share)   -> porous parks (near-outdoor)
//   in-vehicle + outdoors(sidewalk share)  -> STREET (full outdoor, this file)
// Street occupants breathe undiluted plume-level outdoor air with NO
// infiltration reduction, so per-capita they are the most exposed group — which
// is why they must be modeled explicitly rather than folded into "home."

#include "voxelize.h"   // VoxelGrid, CellType, city::Result/Params
#include <vector>
#include <cmath>
#include <algorithm>

namespace city {

// ── Microenvironment split ──────────────────────────────────────────────────
struct TimeBudget { double residence, other_indoor, in_vehicle, outdoors; };

// 24-h population average (NHAPS / Klepeis et al. 2001). This is the ONLY
// occupancy weighting used. We deliberately do NOT model a time-of-release
// (diurnal) occupancy: the event time is unknown and must be treated as
// adversarial, so exposure is weighted uniformly over all 24 hours of the day.
// Using the daily average is the expectation of occupancy over a uniformly-
// distributed release time; a time-of-day profile would let the layout overfit
// to one hour's population pattern (same robustness principle as the fixed
// population and the wind-rose aggregation). NHAPS also tabulates
// weekday/weekend and 3-hour segments (Tsang & Klepeis 1996, EPA/600/R-96/148)
// if a specific scenario is ever required, but the unknown-time default is this.
inline TimeBudget nhaps_average() { return {0.69, 0.18, 0.055, 0.076}; }

// Street-time fraction (in-vehicle + sidewalk share of outdoors).
inline double street_fraction(const TimeBudget& b, double park_share = 0.5) {
    return b.in_vehicle + (1.0 - park_share) * b.outdoors;
}

// ── Street inhabitance field ────────────────────────────────────────────────
// Pedestrians/commuters are placed on the ROAD NETWORK: ground-level (z=1)
// fluid cells inside the city footprint. Pedestrian density tracks local
// activity, so each non-park building sheds its occupant count onto its
// facade-adjacent road cells; the field is then normalized to the total street
// population. (Mass-weighted pedestrian density is the standard assumption
// in micro-environmental exposure models, e.g. EPA APEX/SHEDS street modules.)
struct StreetField {
    int nx, ny, z;                 // sampling slice (z=1, pedestrian level)
    std::vector<float> weight;     // per-cell street occupants (sums to total)
    double total = 0;
};

inline StreetField build_street_occupancy(const Params& p, const Result& r,
                                          const VoxelGrid& g, double street_pop) {
    StreetField sf; sf.nx = g.nx; sf.ny = g.ny; sf.z = 1;
    sf.weight.assign((size_t)g.nx * g.ny, 0.0f);
    auto is_road = [&](int x, int y) {
        if (x < 0 || y < 0 || x >= g.nx || y >= g.ny) return false;
        return g.type[g.idx(x, y, 1)] == CELL_FLUID;   // open ground-level air
    };
    double wsum = 0;
    for (const auto& b : r.blocks) {
        if (b.usage == PARK || b.eff_inh <= 0) continue;
        // facade-adjacent ring (one cell outside the footprint)
        int x0 = b.x0 - 1, x1 = b.x1, y0 = b.y0 - 1, y1 = b.y1;
        std::vector<std::pair<int,int>> ring;
        for (int x = x0; x <= x1; ++x) { if (is_road(x,y0)) ring.push_back({x,y0}); if (is_road(x,y1)) ring.push_back({x,y1}); }
        for (int y = y0; y <= y1; ++y) { if (is_road(x0,y)) ring.push_back({x0,y}); if (is_road(x1,y)) ring.push_back({x1,y}); }
        if (ring.empty()) continue;
        double per = b.eff_inh / ring.size();           // inhabitance-weighted shedding
        for (auto& c : ring) { sf.weight[(size_t)c.second*g.nx + c.first] += (float)per; wsum += per; }
    }
    if (wsum > 0) {                                     // normalize to street population
        double k = street_pop / wsum;
        for (auto& w : sf.weight) w = (float)(w * k);
        sf.total = street_pop;
    }
    return sf;
}

// Street exposure = sum over road cells of (street occupants * outdoor conc).
// vehicle_io < 1 optionally credits in-vehicle cabin filtration for the
// transit sub-fraction (car cabins: I/O ~ 0.4-0.7 with recirculation); default
// 1.0 = full outdoor (conservative).
inline double street_exposure(const StreetField& sf, const std::vector<float>& conc,
                              int nx, int ny, double vehicle_io = 1.0) {
    double e = 0;
    for (int y = 0; y < ny && y < sf.ny; ++y)
        for (int x = 0; x < nx && x < sf.nx; ++x)
            e += (double)sf.weight[(size_t)y*sf.nx + x] * conc[(size_t)y*nx + x];
    return e * vehicle_io;
}

} // namespace city
