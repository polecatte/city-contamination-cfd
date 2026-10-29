// gen_gate6_geom.cpp — synthetic Stage-A geometries for the Phase-5 airflow gates
// (OPENLB_PORT_STATUS_AND_VERIFICATION.md §3, Phase 5).
//
//   CASE=abl   Gate 6a: empty domain with a ground plane. Isolates the floor treatment:
//              any drift of the mean profile along the fetch is the solver + wall, since
//              the inlet reproduces its target log law to 0.10 %.
//   CASE=cube  Gate 6b: one surface-mounted cube of H cells in an AIJ-sized domain
//              (5H upstream, 15H downstream, 5H each side, 5H above the roof).
//   CASE=box   Gates 7a-c (Phase 6): 40^3 empty box with a ground plane, a small ground-level
//              source patch Omega near the inlet (source_mask.u8) and a constant ground
//              deposition velocity DEP_VD m/s (dep_vel.f32). Run with STEP4_UNIFORM_U.
//   CASE=obstacles  rectangular blocks from OBST_FILE, one per line "xc yc length width height
//              angle_deg" (metres; angle of the length axis from +x, counter-clockwise; '#'
//              comments), e.g. the MUST container array (tests/must_score.py). The array is
//              turned by OBST_ROT degrees counter-clockwise about its centre so that the wind
//              blows along +x: OBST_ROT = minus the direction the wind blows TOWARD in the file's
//              frame (counter-clockwise from its +x). Set in a COST 732 domain: UP_H, DOWN_H, LAT_H, TOP_H times the
//              tallest block (defaults 5, 15, 5, 5). SRC_X SRC_Y SRC_Z (m, the file's frame, z
//              above the ground surface) put a one-cell scalar source there (source_mask.u8).
//              meta_geom.txt records the transform so observation points can be mapped onto
//              the grid: file-frame point p -> rotated R(rot_deg)(p - array_centre), cell index
//              (x' - x0_rot)/dx, (y' - y0_rot)/dx; cell k's centre is (k - 1/2) dx above the
//              ground surface (the bounce-back wall lies half-way between cells 0 and 1).
//
// The face overlay (inlet / outlet / slip / frame) is NOT reimplemented here: the scene is
// built as a VoxelGrid and passed through the same city::build_material_map the city uses,
// so the gates exercise exactly the boundary layout the production map has.
//
// Build:  g++ -O2 -std=c++17 -I. gen_gate6_geom.cpp -o gen_gate6_geom
// Run:    CASE=abl  OUT_DIR=geom_abl  ./gen_gate6_geom
//         CASE=cube OUT_DIR=geom_cube ./gen_gate6_geom
//         CASE=box  OUT_DIR=geom_box  ./gen_gate6_geom

#include "openlb_geometry.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <array>
#include <algorithm>

using namespace city;
static double envd(const char* k,double d){const char* e=getenv(k);return e?atof(e):d;}
static int    envi(const char* k,int d){const char* e=getenv(k);return e?atoi(e):d;}

int main() {
    const std::string CASE = getenv("CASE") ? getenv("CASE") : "abl";
    const std::string OUT  = getenv("OUT_DIR") ? getenv("OUT_DIR") : ("geom_"+CASE);
    { std::string c="mkdir -p '"+OUT+"'"; if(system(c.c_str())){} }
    const double dx = envd("DX", 4.0);

    int nx, ny, nz, H = 0, cx0 = 0, cy0 = 0;
    // CASE=obstacles: blocks in the rotated frame (centre, half sizes, angle) and the grid origin
    struct Block { double xc, yc, hl, hw, h, ang; };
    std::vector<Block> blocks;
    double rot = 0, acx = 0, acy = 0, x0r = 0, y0r = 0, Hmax = 0;
    if (CASE == "abl") {
        // Default: the city domain's fetch and height (177 x 89 at dx=4) so the drift is
        // measured over the length the city actually sees. The flow is laterally homogeneous
        // between free-slip walls, so ny only needs to hold enough turbulence to average;
        // 40 cells is 160 m, several integral scales (ABL_LTURB = 20 m).
        nx = envi("NX", 177); ny = envi("NY", 40); nz = envi("NZ", 89);
    } else if (CASE == "cube") {
        H  = envi("CUBE_H", 10);                      // H/dx >= 10 (audit Phase 5b)
        const int up = envi("UP_H", 5), down = envi("DOWN_H", 15), lat = envi("LAT_H", 5), top = envi("TOP_H", 5);
        nx = (up + 1 + down) * H;
        ny = (2*lat + 1) * H;
        nz = 1 + (1 + top) * H;                       // ground layer + cube + headroom
        cx0 = up * H; cy0 = lat * H;
    } else if (CASE == "box") {
        nx = envi("NX", 40); ny = envi("NY", 40); nz = envi("NZ", 40);
    } else if (CASE == "obstacles") {
        const char* fn = getenv("OBST_FILE");
        if (!fn) { fprintf(stderr, "CASE=obstacles needs OBST_FILE (xc yc length width height angle_deg per line)\n"); return 2; }
        std::ifstream in(fn);
        if (!in) { fprintf(stderr, "cannot read %s\n", fn); return 2; }
        std::vector<std::array<double,6>> raw;
        for (std::string line; std::getline(in, line); ) {
            const size_t c = line.find('#'); if (c != std::string::npos) line.resize(c);
            std::replace(line.begin(), line.end(), ',', ' ');
            std::istringstream ls(line); std::array<double,6> v;
            if (ls >> v[0] >> v[1] >> v[2] >> v[3] >> v[4] >> v[5]) raw.push_back(v);
        }
        if (raw.empty()) { fprintf(stderr, "no blocks in %s\n", fn); return 2; }
        double xmn = 1e300, xmx = -1e300, ymn = 1e300, ymx = -1e300;
        for (auto& v : raw) { xmn = std::min(xmn, v[0]); xmx = std::max(xmx, v[0]); ymn = std::min(ymn, v[1]); ymx = std::max(ymx, v[1]); }
        acx = 0.5 * (xmn + xmx); acy = 0.5 * (ymn + ymx);
        rot = envd("OBST_ROT", 0.0);
        const double cr = std::cos(rot * M_PI / 180), sr = std::sin(rot * M_PI / 180);
        double bx0 = 1e300, bx1 = -1e300, by0 = 1e300, by1 = -1e300;
        for (auto& v : raw) {
            const double px = v[0] - acx, py = v[1] - acy;
            Block b{cr * px - sr * py, sr * px + cr * py, 0.5 * v[2], 0.5 * v[3], v[4], (v[5] + rot) * M_PI / 180};
            const double ca = std::abs(std::cos(b.ang)), sa = std::abs(std::sin(b.ang));
            const double ex = b.hl * ca + b.hw * sa, ey = b.hl * sa + b.hw * ca;   // half extents of the box
            bx0 = std::min(bx0, b.xc - ex); bx1 = std::max(bx1, b.xc + ex);
            by0 = std::min(by0, b.yc - ey); by1 = std::max(by1, b.yc + ey);
            Hmax = std::max(Hmax, b.h); blocks.push_back(b);
        }
        const double up = envd("UP_H", 5), down = envd("DOWN_H", 15), lat = envd("LAT_H", 5), top = envd("TOP_H", 5);
        x0r = bx0 - up * Hmax; y0r = by0 - lat * Hmax;
        nx = (int)std::ceil((bx1 + down * Hmax - x0r) / dx) + 1;
        ny = (int)std::ceil((by1 + lat * Hmax - y0r) / dx) + 1;
        nz = 1 + (int)std::ceil((1 + top) * Hmax / dx) + 1;
        printf("[gen_gate6_geom] obstacles: %zu blocks from %s, rotated %g deg, tallest %.2f m (%.1f cells)\n",
               blocks.size(), fn, rot, Hmax, Hmax / dx);
    } else {
        fprintf(stderr, "CASE must be abl, cube, box or obstacles\n"); return 2;
    }

    VoxelGrid g;
    g.nx = nx; g.ny = ny; g.nz = nz; g.cell_size = dx;
    const size_t N = (size_t)nx*ny*nz;
    g.type.assign(N, CELL_FLUID); g.perm.assign(N, 1.0f);
    g.inh.assign(N, 0.0f); g.dep_vel.assign(N, 0.0f);
    for (int y=0;y<ny;++y) for (int x=0;x<nx;++x) { g.type[g.idx(x,y,0)] = CELL_GROUND; g.perm[g.idx(x,y,0)] = 0.f; }
    // The cube is solid through its whole volume: shells with perm 0 map to MAT_WALL.
    if (H > 0)
        for (int z=1; z<=H; ++z) for (int y=cy0; y<cy0+H; ++y) for (int x=cx0; x<cx0+H; ++x) {
            g.type[g.idx(x,y,z)] = CELL_SHELL; g.perm[g.idx(x,y,z)] = 0.f;
        }

    // blocks: a cell is solid when its centre lies inside a block's footprint and below its top
    // (cell k's centre is (k - 1/2) dx above the ground surface)
    long nSolid = 0;
    for (auto& b : blocks) {
        const double ca = std::cos(b.ang), sa = std::sin(b.ang), r = std::hypot(b.hl, b.hw);
        const int i0 = std::max(0, (int)std::floor((b.xc - r - x0r) / dx)), i1 = std::min(nx - 1, (int)std::ceil((b.xc + r - x0r) / dx));
        const int j0 = std::max(0, (int)std::floor((b.yc - r - y0r) / dx)), j1 = std::min(ny - 1, (int)std::ceil((b.yc + r - y0r) / dx));
        for (int i = i0; i <= i1; ++i) for (int j = j0; j <= j1; ++j) {
            const double px = x0r + i * dx - b.xc, py = y0r + j * dx - b.yc;
            if (std::abs(ca * px + sa * py) > b.hl || std::abs(-sa * px + ca * py) > b.hw) continue;
            for (int k = 1; k < nz - 1 && (k - 0.5) * dx < b.h; ++k)
                if (g.type[g.idx(i, j, k)] != CELL_SHELL) { g.type[g.idx(i, j, k)] = CELL_SHELL; g.perm[g.idx(i, j, k)] = 0.f; ++nSolid; }
        }
    }
    if (!blocks.empty()) printf("[gen_gate6_geom] obstacles: %ld solid cells\n", nSolid);

    MaterialMap m = build_material_map(g, 0.0);
    const bool ok = material_reconcile(g, m, true);
    export_material_map((OUT + "/material_map.dat").c_str(), m);

    if (CASE == "box") {
        // Omega: 3 x 4 ground-level cells, 6 cells downstream of the inlet, centred in y.
        std::vector<uint8_t> src(N, 0); long nOm = 0;
        for (int y=ny/2-2; y<ny/2+2; ++y) for (int x=6; x<9; ++x) { src[g.idx(x,y,1)] = 1; ++nOm; }
        const float vdep = (float)envd("DEP_VD", 2e-3);          // m/s; the city's max is 2 mm/s
        std::vector<float> vd(N, 0.f);
        for (int y=0;y<ny;++y) for (int x=0;x<nx;++x) vd[g.idx(x,y,0)] = vdep;
        int h[5] = {nx, ny, nz, (int)std::lround(dx*1000.0), 1};
        FILE* fs = fopen((OUT + "/source_mask.u8").c_str(), "wb");
        if (fs) { fwrite(h, sizeof(int), 5, fs); fwrite(src.data(), 1, N, fs); fclose(fs); }
        FILE* fv = fopen((OUT + "/dep_vel.f32").c_str(), "wb");
        if (fv) { fwrite(h, sizeof(int), 5, fv); fwrite(vd.data(), sizeof(float), N, fv); fclose(fv); }
        printf("[gen_gate6_geom] box: Omega = %ld cells, ground v_d = %.3g m/s\n", nOm, vdep);
    }

    bool srcOk = true;
    if (CASE == "obstacles" && getenv("SRC_X")) {                 // one-cell scalar source
        const double cr = std::cos(rot * M_PI / 180), sr = std::sin(rot * M_PI / 180);
        const double px = envd("SRC_X", 0) - acx, py = envd("SRC_Y", 0) - acy, pz = envd("SRC_Z", 1.0);
        const int i = (int)std::lround((cr * px - sr * py - x0r) / dx), j = (int)std::lround((sr * px + cr * py - y0r) / dx);
        const int k = (int)std::lround(pz / dx + 0.5);
        std::vector<uint8_t> src(N, 0);
        if (i < 1 || i >= nx - 1 || j < 1 || j >= ny - 1 || k < 1 || k >= nz - 1 || m.mat[g.idx(i, j, k)] != MAT_FLUID) {
            fprintf(stderr, "[gen_gate6_geom] source (%g, %g, %g) m -> cell (%d, %d, %d) is not a fluid cell\n",
                    envd("SRC_X", 0), envd("SRC_Y", 0), pz, i, j, k);
            srcOk = false;
        } else {
            src[g.idx(i, j, k)] = 1;
            int h[5] = {nx, ny, nz, (int)std::lround(dx*1000.0), 1};
            FILE* fs = fopen((OUT + "/source_mask.u8").c_str(), "wb");
            if (fs) { fwrite(h, sizeof(int), 5, fs); fwrite(src.data(), 1, N, fs); fclose(fs); }
            printf("[gen_gate6_geom] obstacles: source cell (%d, %d, %d)\n", i, j, k);
        }
    }

    FILE* f = fopen((OUT + "/meta_geom.txt").c_str(), "w");
    if (f) {
        fprintf(f, "case %s\ngrid %d %d %d\ndx_m %.4f\n", CASE.c_str(), nx, ny, nz, dx);
        if (H > 0) fprintf(f, "cube_H_cells %d\ncube_x0 %d\ncube_y0 %d\n", H, cx0, cy0);
        if (CASE == "obstacles")
            fprintf(f, "rot_deg %.9g\narray_cx %.9g\narray_cy %.9g\nx0_rot %.9g\ny0_rot %.9g\nH_max_m %.6g\nblocks %zu\n",
                    rot, acx, acy, x0r, y0r, Hmax, blocks.size());
        fclose(f);
    }
    printf("[gen_gate6_geom] %s: %dx%dx%d dx=%.2f m%s -> %s/material_map.dat\n", CASE.c_str(),
           nx, ny, nz, dx, H>0 ? (" cube H=" + std::to_string(H) + " cells").c_str() : "", OUT.c_str());
    return ok && srcOk ? 0 : 1;
}
