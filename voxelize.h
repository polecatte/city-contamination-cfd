#pragma once
// voxelize.h v7 — 3D grid rasterizer for LBM
// All structures (buildings and parks) are hollow rectangular prisms:
//   SHELL = permeable envelope (partial bounce-back)
//   INDOOR = interior air (carries inhabitance weight)
// Parks have the same geometry as buildings but with high shell permeability.
#include "city_builder7.h"
#include "deposition.h"   // size-dependent dry deposition (Zhang et al. 2001)
#include <cstdint>

namespace city {

// ── Cell types for LBM ──────────────────────────────────────
enum CellType : uint8_t {
    CELL_FLUID   = 0,  // outdoor air — normal LBM streaming + collision
    CELL_GROUND  = 1,  // solid ground — full bounce-back
    CELL_SHELL   = 2,  // structure envelope — partial bounce-back (permeable)
    CELL_INDOOR  = 3,  // interior air — normal fluid, carries inhabitance weight
};

constexpr int MIN_HOLLOW_DIM = 3; // min cells in each dim for hollow interior

// ── Permeability by usage type ──────────────────────────────
inline float usage_permeability(Usage u) {
    switch(u) {
        case BUSINESS:    return PERM_BIZ;      // 0.005 — sealed curtain wall
        case RES_HIGH:    return PERM_RES;      // 0.010 — masonry with windows
        case RES_LOW:     return PERM_RES_LOW;  // 0.020 — wood frame houses
        case PARK:        return PERM_PARK;     // 0.800 — shrubbery walls + tree canopy roof
        default:          return 0.0f;
    }
}

// ── Dry-deposition velocity by surface type (m/s) ───────────
// Turbulent + Brownian capture velocity of the SURFACE, excluding
// gravitational settling (that is added separately from particle size).
// These are size-independent base values (a first-order simplification of
// the full resistance/impaction model) and are meant to be tuned. Order of
// magnitude follows urban PM dry-deposition literature: smooth sealed
// surfaces capture weakly; vegetated parks capture strongly (foliage
// impaction + interception), hence "parks scrub more".
constexpr float DEP_BIZ     = 0.001f;  // glass/curtain wall — smooth, low capture
constexpr float DEP_RES     = 0.0015f; // masonry
constexpr float DEP_RES_LOW = 0.0030f; // textured low-rise, some yard vegetation
constexpr float DEP_PARK    = 0.0300f; // tree canopy / shrubs — strong capture
constexpr float DEP_GROUND  = 0.0020f; // pavement / bare soil at z=0

inline float usage_deposition(Usage u) {
    switch(u) {
        case BUSINESS:    return DEP_BIZ;
        case RES_HIGH:    return DEP_RES;
        case RES_LOW:     return DEP_RES_LOW;
        case PARK:        return DEP_PARK;
        default:          return 0.0f;
    }
}

// ── Voxel grid output ───────────────────────────────────────
struct VoxelGrid {
    int nx, ny, nz;
    double cell_size;
    std::vector<uint8_t> type;
    std::vector<float> perm;
    std::vector<float> inh;
    std::vector<float> dep_vel;   // per-cell surface deposition velocity (m/s); 0 for fluid/indoor
    inline size_t idx(int x, int y, int z) const {
        return (size_t)z * ny * nx + y * nx + x;
    }
};

// ── Rasterize city to voxel grid ────────────────────────────
// solid_buildings: when true, non-park buildings are rasterized as SOLID
// (full bounce-back, perm=0) for aerodynamic correctness (COST 732; Tominaga
// et al. 2008), and their occupants are NOT placed in grid INDOOR cells —
// indoor exposure is computed separately by the infiltration model
// (infiltration.h; Liu & Nazaroff 2001; Chen & Zhao 2011; Nazaroff 2004).
// Parks remain porous (Merlier et al. 2018). Default false = legacy permeable.
// dep_dp: if > 0, per-surface deposition velocity is computed size-dependently
// for this particle diameter (Zhang et al. 2001, deposition.h) instead of the
// constant usage_deposition() values — used for polydisperse, per-bin transport.
// Vertical domain sizing (COST 732; Franke et al. 2006): 5H headroom above the
// tallest building. Delegates to nz_cost732() so production and the sweep use the
// SAME sizing rule. Power-of-two rounding is obsolete (it inflated nz up to ~2× and
// made the objective step-discontinuous in the tallest dimension); nz_cost732 gives
// the smooth ceil(6H/Δx). Still exposed as a free function so a sweep can size nz
// ONCE for its tallest design and pass that fixed value to every voxelize() call.
inline int nz_for_max_height_cells(int max_height_cells){
    return nz_cost732((double)max_height_cells * CELL);   // == max(2, 6·max_height_cells)
}
inline int nz_for_max_height_m(double max_height_m){
    return nz_for_max_height_cells((int)std::ceil(max_height_m / CELL));
}

// If nz_fixed > 0 it is used directly (sweeps pass a constant sized for the tallest
// design); buildings taller than the box are clipped to nz-1 as before. If <= 0,
// nz is sized to THIS design's tallest building (legacy per-design behavior).
inline VoxelGrid voxelize(const Params& p, const Result& r,
                          bool solid_buildings = false,
                          double dep_dp = -1.0, double dep_rho = 1800.0,
                          double dep_ustar = 0.5, int nz_fixed = 0) {
    int nx = r.nx_cells, ny = r.ny_cells;

    int max_hc = 0;
    for (auto& b : r.blocks)
        if (b.height_cells > max_hc) max_hc = b.height_cells;
    int nz = (nz_fixed > 0) ? nz_fixed : nz_for_max_height_cells(max_hc);

    VoxelGrid g;
    g.nx = nx; g.ny = ny; g.nz = nz;
    g.cell_size = CELL;
    size_t total = (size_t)nx * ny * nz;
    g.type.assign(total, CELL_FLUID);
    g.perm.assign(total, 0.0f);
    g.inh.assign(total, 0.0f);
    g.dep_vel.assign(total, 0.0f);

    // z=0: solid ground everywhere
    for (int y = 0; y < ny; ++y)
        for (int x = 0; x < nx; ++x) {
            g.type[g.idx(x, y, 0)] = CELL_GROUND;
            g.dep_vel[g.idx(x, y, 0)] = DEP_GROUND;
        }

    // Rasterize each block as a HOLLOW rectangular prism: a one-cell-thick SHELL
    // (permeable envelope) enclosing INDOOR air. This lets wind partially
    // penetrate (partial bounce-back ∝ β) and lets the interior carry the
    // building's occupants for the exposure calculation.
    for (auto& b : r.blocks) {
        int hc = b.height_cells;
        if (hc < 1) continue;

        int x0 = std::max(0, b.x0), x1 = std::min(nx, b.x1);
        int y0 = std::max(0, b.y0), y1 = std::min(ny, b.y1);
        int bw = x1 - x0, bd = y1 - y0;
        if (bw <= 0 || bd <= 0) continue;
        int hz = std::min(hc, nz - 1);

        float beta = usage_permeability(b.usage);   // envelope permeability
        float dvel = (dep_dp > 0.0)
                   ? (float)dep::deposition_velocity(b.usage, dep_dp, dep_rho, dep_ustar)
                   : usage_deposition(b.usage);     // surface deposition velocity
        // Only prisms ≥3 cells in every dimension can have a genuine interior;
        // smaller ones are filled solid-shell (no INDOOR void).
        bool hollow = (bw >= MIN_HOLLOW_DIM
                    && bd >= MIN_HOLLOW_DIM
                    && hz >= MIN_HOLLOW_DIM);

        // Distribute the building's effective inhabitance uniformly over its
        // air cells: prefer the interior (INDOOR) volume (w−2)(d−2)(h−2); if the
        // prism is too small to be hollow, spread over the whole footprint.
        int n_indoor = 0;
        if (hollow) {
            int iw = bw - 2, id = bd - 2, ih = hz - 2;
            n_indoor = (ih > 0) ? iw * id * ih : 0;
        }
        int n_total = bw * bd * hz;
        float inh_per_cell = 0.0f;
        if (n_indoor > 0)
            inh_per_cell = (float)(b.eff_inh / n_indoor);
        else if (n_total > 0)
            inh_per_cell = (float)(b.eff_inh / n_total);

        // A cell is SHELL if it lies on any of the 6 faces (walls/floor/roof),
        // else INDOOR. z starts at 1 (z=0 is the global ground plane).
        for (int z = 1; z <= hz; ++z) {
            for (int y = y0; y < y1; ++y) {
                for (int x = x0; x < x1; ++x) {
                    size_t ci = g.idx(x, y, z);

                    bool on_xwall = (x == x0 || x == x1 - 1);
                    bool on_ywall = (y == y0 || y == y1 - 1);
                    bool on_floor = (z == 1);
                    bool on_roof  = (z == hz);
                    bool is_shell = on_xwall || on_ywall || on_floor || on_roof;

                    bool make_solid = solid_buildings && (b.usage != PARK);
                    if (make_solid) {
                        // Aerodynamically solid: full bounce-back (perm=0), no
                        // through-flow, facade still deposits. Occupants are
                        // tracked per-building (b.eff_inh) for the infiltration
                        // model, NOT injected into grid cells.
                        g.type[ci] = CELL_SHELL;
                        g.perm[ci] = 0.0f;
                        g.dep_vel[ci] = dvel;
                    } else if (!hollow || is_shell) {
                        g.type[ci] = CELL_SHELL;
                        g.perm[ci] = beta;
                        g.dep_vel[ci] = dvel;
                        if (!hollow) g.inh[ci] += inh_per_cell;
                    } else {
                        g.type[ci] = CELL_INDOOR;
                        g.inh[ci] += inh_per_cell;
                    }
                }
            }
        }
    }

    return g;
}

// ── Export voxel grid to binary ──────────────────────────────
inline void export_voxels(const char* fn, const VoxelGrid& g) {
    FILE* f = fopen(fn, "wb");
    int hdr[4] = {g.nx, g.ny, g.nz, (int)(g.cell_size * 1000)};
    fwrite(hdr, sizeof(int), 4, f);
    fwrite(g.type.data(), 1, g.type.size(), f);
    fwrite(g.perm.data(), sizeof(float), g.perm.size(), f);
    fwrite(g.inh.data(), sizeof(float), g.inh.size(), f);
    fclose(f);
}

// ── Print summary ───────────────────────────────────────────
inline void print_voxel_summary(const VoxelGrid& g) {
    int cnt[4] = {};
    double total_inh = 0;
    for (size_t i = 0; i < g.type.size(); ++i) {
        if (g.type[i] < 4) cnt[g.type[i]]++;
        total_inh += g.inh[i];
    }
    size_t total = g.type.size();
    printf("  Voxel grid: %d x %d x %d = %zu cells (%.1f MB)\n",
           g.nx, g.ny, g.nz, total, total * (1+4+4) / 1e6);
    printf("  FLUID:  %9d (%5.1f%%)\n", cnt[0], cnt[0]*100.0/total);
    printf("  GROUND: %9d (%5.1f%%)\n", cnt[1], cnt[1]*100.0/total);
    printf("  SHELL:  %9d (%5.1f%%)\n", cnt[2], cnt[2]*100.0/total);
    printf("  INDOOR: %9d (%5.1f%%)\n", cnt[3], cnt[3]*100.0/total);
    printf("  Inhabitance in grid: %.0f\n", total_inh);
}

} // namespace city
