#pragma once
// openlb_geometry.h — geometry bridge: urban voxel grid  →  OpenLB SuperGeometry
// material map.
//
// PART OF THE OpenLB MIGRATION (OPENLB_MIGRATION_PLAN.md, Stage A → Stage B).
//
// The scene build (city_builder7.h + city_zoning.h + voxelize.h) already produces a
// VoxelGrid whose per-cell CellType + permeability classify every cell. OpenLB's
// SuperGeometry instead wants a per-cell *material number* that selects the dynamics /
// boundary condition applied there. This header stamps OpenLB material numbers
// directly from the existing voxel grid — cell-for-cell, no STL round-trip — so the
// new solid / porous / ground / inlet classification matches the old voxelization
// exactly (§6.3 "geometry bridge" of the plan).
//
// MATERIAL NUMBERING (NSE / airflow lattice)
// ------------------------------------------
//   0  MAT_DONOTHING  reserved "outside" — unused in the interior (OpenLB convention)
//   1  MAT_FLUID      bulk outdoor air                 → BGK/WALE fluid dynamics
//   2  MAT_WALL       solid buildings                  → smooth no-slip bounce-back
//   3  MAT_INLET      upwind domain face (fluid)       → velocity BC (ABL/RFG inlet)
//   4  MAT_OUTLET     downwind domain face (fluid)     → pressure/outflow BC
//   5  MAT_SLIP       lateral + top faces (fluid)      → free-slip / symmetry
//   6  MAT_POROUS     park canopy (porous shell+air)   → PorousBGKdynamics
//   7  MAT_GROUND     ground plane                     → ROUGH-wall function (z0) — kept
//                     separate from buildings so the ABL wall-function corrective (which
//                     preserves horizontal homogeneity) applies to the floor only.
//
// Ω (the burst-release source set) is deliberately NOT a material number. It is a
// property of the ADVECTION–DIFFUSION lattice, not the NSE geometry: a single cell can
// be both an NSE velocity inlet AND a scalar-release cell. So Ω is carried in its own
// `source_mask.u8` array (identical to the one forward_city.cpp already writes) and is
// consumed by the AD source post-processor in Stage B, independently of this map. This
// keeps the material map a pure statement of the airflow geometry.
//
// FILE FORMAT
// -----------
//   material_map.dat : 5-int32 header [nx, ny, nz, dx*1000, ncomp=1] then nx*ny*nz
//                      int32 material numbers (z-major, matching VoxelGrid::idx).
// The 5-int self-describing header is the same interchange convention the rest of the
// project uses (forward_city.cpp), so the viz suite / Stage B loaders need no new parser.

#include "voxelize.h"
#include <cstdint>
#include <cstdio>
#include <vector>
#include <string>
#include <cmath>

namespace city {

// ── OpenLB material numbers ─────────────────────────────────
enum OlbMaterial : int32_t {
    MAT_DONOTHING = 0,
    MAT_FLUID     = 1,
    MAT_WALL      = 2,   // solid buildings — smooth no-slip bounce-back
    MAT_INLET     = 3,
    MAT_OUTLET    = 4,
    MAT_SLIP      = 5,
    MAT_POROUS    = 6,
    MAT_GROUND    = 7,   // ground plane — ROUGH wall (wall function w/ z0), kept SEPARATE
                         // from buildings so the ABL wall-function corrective targets only
                         // the floor (buildings stay smooth no-slip per COST 732).
};

// A shell cell counts as a porous PARK envelope (vs a solid building) when its
// permeability is high. voxelize.h gives parks PERM_PARK (0.8) and solid buildings
// perm 0; the 0.5 split is unambiguous between those.
constexpr float POROUS_PERM_THRESHOLD = 0.5f;

struct MaterialMap {
    int nx, ny, nz;
    double cell_size;
    std::vector<int32_t> mat;   // per-cell material number, size nx*ny*nz
    inline size_t idx(int x, int y, int z) const {
        return (size_t)z * ny * nx + (size_t)y * nx + x;
    }
};

// ── Stamp OpenLB materials from the voxel grid ──────────────
// wind_deg selects which lateral face is the velocity inlet. For the project's default
// 0° (+x) wind the inlet is the x=0 plane and the outlet is x=nx-1; 180° swaps them;
// 90°/270° use the y faces. Only axis-aligned inlets are handled (matching the solver's
// wind_angle usage); an oblique wind falls back to the +x convention with a note.
//
// Assignment order (guarantees exactly one material per cell):
//   (1) every cell defaults to MAT_FLUID
//   (2) by cell type:  GROUND + solid SHELL → WALL;  park SHELL + INDOOR → POROUS
//   (3) domain-face overlay, applied to FLUID cells only so a building or the ground
//       that happens to sit on a face is never reclassified as an inlet/outlet/slip:
//       inlet face → INLET, opposite face → OUTLET, remaining lateral + top → SLIP.
inline MaterialMap build_material_map(const VoxelGrid& g, double wind_deg = 0.0) {
    MaterialMap m;
    m.nx = g.nx; m.ny = g.ny; m.nz = g.nz; m.cell_size = g.cell_size;
    const int nx = g.nx, ny = g.ny, nz = g.nz;
    const size_t N = (size_t)nx * ny * nz;
    m.mat.assign(N, MAT_FLUID);

    // (2) type-driven interior classification
    for (size_t i = 0; i < N; ++i) {
        switch (g.type[i]) {
            case CELL_FLUID:  m.mat[i] = MAT_FLUID;  break;
            case CELL_GROUND: m.mat[i] = MAT_GROUND; break;   // rough-wall floor (own material)
            case CELL_INDOOR: m.mat[i] = MAT_POROUS; break;   // park interior air
            case CELL_SHELL:
                m.mat[i] = (g.perm[i] >= POROUS_PERM_THRESHOLD) ? MAT_POROUS : MAT_WALL;
                break;
            default:          m.mat[i] = MAT_WALL; break;      // unknown → treat as solid
        }
    }

    // (3) domain-boundary overlay on fluid cells only.
    // Normalize wind to the nearest axis; default +x.
    double wd = std::fmod(wind_deg, 360.0); if (wd < 0) wd += 360.0;
    enum { PX, PY, NX_, NY_ } inflow;
    if      (wd >= 45.0  && wd < 135.0) inflow = PY;   // wind toward +y  → inlet at y=0
    else if (wd >= 135.0 && wd < 225.0) inflow = NX_;  // toward −x        → inlet at x=nx-1
    else if (wd >= 225.0 && wd < 315.0) inflow = NY_;  // toward −y        → inlet at y=ny-1
    else                                inflow = PX;   // toward +x        → inlet at x=0

    auto set_face = [&](int x, int y, int z, int32_t v) {
        size_t i = m.idx(x, y, z);
        if (m.mat[i] == MAT_FLUID) m.mat[i] = v;   // fluid only
    };
    // inlet / outlet on the two wind-aligned faces
    for (int z = 0; z < nz; ++z) for (int y = 0; y < ny; ++y) {
        int xin = (inflow == PX) ? 0 : (inflow == NX_ ? nx - 1 : -1);
        int xout= (inflow == PX) ? nx - 1 : (inflow == NX_ ? 0 : -1);
        if (xin >= 0)  set_face(xin,  y, z, MAT_INLET);
        if (xout >= 0) set_face(xout, y, z, MAT_OUTLET);
    }
    for (int z = 0; z < nz; ++z) for (int x = 0; x < nx; ++x) {
        int yin = (inflow == PY) ? 0 : (inflow == NY_ ? ny - 1 : -1);
        int yout= (inflow == PY) ? ny - 1 : (inflow == NY_ ? 0 : -1);
        if (yin >= 0)  set_face(x, yin,  z, MAT_INLET);
        if (yout >= 0) set_face(x, yout, z, MAT_OUTLET);
    }
    // remaining lateral faces + top → free-slip (fluid only; inlet/outlet already claimed)
    for (int z = 0; z < nz; ++z) for (int x = 0; x < nx; ++x) {
        set_face(x, 0,      z, MAT_SLIP);
        set_face(x, ny - 1, z, MAT_SLIP);
    }
    for (int z = 0; z < nz; ++z) for (int y = 0; y < ny; ++y) {
        set_face(0,      y, z, MAT_SLIP);
        set_face(nx - 1, y, z, MAT_SLIP);
    }
    for (int y = 0; y < ny; ++y) for (int x = 0; x < nx; ++x)
        set_face(x, y, nz - 1, MAT_SLIP);   // top

    return m;
}

// ── Export material map (5-int header + int32 payload) ──────
inline void export_material_map(const char* fn, const MaterialMap& m) {
    FILE* f = fopen(fn, "wb");
    if (!f) { fprintf(stderr, "[openlb_geom] WARN cannot write %s\n", fn); return; }
    int hdr[5] = { m.nx, m.ny, m.nz, (int)std::lround(m.cell_size * 1000.0), 1 };
    fwrite(hdr, sizeof(int), 5, f);
    fwrite(m.mat.data(), sizeof(int32_t), m.mat.size(), f);
    fclose(f);
}

// ── Material histogram + reconciliation against the voxel grid ──
// Returns true iff every voxel cell is accounted for by exactly one material and the
// per-class sums reconcile — this is the Step-2 gate ("material counts match the old
// voxelizer").
inline bool material_reconcile(const VoxelGrid& g, const MaterialMap& m, bool verbose = true) {
    const size_t N = g.type.size();
    // voxel-side counts
    long v_fluid=0, v_ground=0, v_shell_solid=0, v_shell_porous=0, v_indoor=0;
    for (size_t i = 0; i < N; ++i) {
        switch (g.type[i]) {
            case CELL_FLUID:  ++v_fluid; break;
            case CELL_GROUND: ++v_ground; break;
            case CELL_INDOOR: ++v_indoor; break;
            case CELL_SHELL:
                if (g.perm[i] >= POROUS_PERM_THRESHOLD) ++v_shell_porous; else ++v_shell_solid;
                break;
        }
    }
    // material-side counts
    long m_cnt[8] = {0,0,0,0,0,0,0,0};
    for (size_t i = 0; i < N; ++i) if (m.mat[i] >= 0 && m.mat[i] <= 7) ++m_cnt[m.mat[i]];
    long m_fluidfaces = m_cnt[MAT_FLUID] + m_cnt[MAT_INLET] + m_cnt[MAT_OUTLET] + m_cnt[MAT_SLIP];

    // reconciliation identities
    bool ok = true;
    auto check = [&](const char* name, long a, long b) {
        bool eq = (a == b); if (!eq) ok = false;
        if (verbose) printf("  [%s] %-34s  %ld %s %ld\n", eq?"PASS":"FAIL", name, a, eq?"==":"!=", b);
    };
    if (verbose) {
        printf("\n[openlb_geom] material histogram (%zu cells)\n", N);
        const char* nm[8] = {"DONOTHING","FLUID","WALL","INLET","OUTLET","SLIP","POROUS","GROUND"};
        for (int k = 0; k < 8; ++k)
            printf("  MAT %d %-10s %10ld (%5.2f%%)\n", k, nm[k], m_cnt[k], 100.0*m_cnt[k]/N);
        printf("\n[openlb_geom] reconciliation against voxel grid\n");
    }
    check("GROUND = ground cells",          m_cnt[MAT_GROUND], v_ground);
    check("WALL = solid building shells",   m_cnt[MAT_WALL],   v_shell_solid);
    check("POROUS = park shells + indoor",  m_cnt[MAT_POROUS], v_shell_porous + v_indoor);
    check("fluid+inlet+outlet+slip = FLUID", m_fluidfaces,     v_fluid);
    check("total cells conserved",          (long)N,
          m_cnt[0]+m_cnt[MAT_FLUID]+m_cnt[MAT_WALL]+m_cnt[MAT_INLET]+m_cnt[MAT_OUTLET]+m_cnt[MAT_SLIP]+m_cnt[MAT_POROUS]+m_cnt[MAT_GROUND]);
    if (verbose) printf("[openlb_geom] gate: %s\n", ok ? "PASS — material counts match the old voxelizer"
                                                        : "FAIL — mapping does not reconcile");
    return ok;
}

} // namespace city
