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
#include <string>

using namespace city;
static double envd(const char* k,double d){const char* e=getenv(k);return e?atof(e):d;}
static int    envi(const char* k,int d){const char* e=getenv(k);return e?atoi(e):d;}

int main() {
    const std::string CASE = getenv("CASE") ? getenv("CASE") : "abl";
    const std::string OUT  = getenv("OUT_DIR") ? getenv("OUT_DIR") : ("geom_"+CASE);
    { std::string c="mkdir -p '"+OUT+"'"; if(system(c.c_str())){} }
    const double dx = envd("DX", 4.0);

    int nx, ny, nz, H = 0, cx0 = 0, cy0 = 0;
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
    } else {
        fprintf(stderr, "CASE must be abl, cube or box\n"); return 2;
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

    FILE* f = fopen((OUT + "/meta_geom.txt").c_str(), "w");
    if (f) {
        fprintf(f, "case %s\ngrid %d %d %d\ndx_m %.4f\n", CASE.c_str(), nx, ny, nz, dx);
        if (H > 0) fprintf(f, "cube_H_cells %d\ncube_x0 %d\ncube_y0 %d\n", H, cx0, cy0);
        fclose(f);
    }
    printf("[gen_gate6_geom] %s: %dx%dx%d dx=%.2f m%s -> %s/material_map.dat\n", CASE.c_str(),
           nx, ny, nz, dx, H>0 ? (" cube H=" + std::to_string(H) + " cells").c_str() : "", OUT.c_str());
    return ok ? 0 : 1;
}
