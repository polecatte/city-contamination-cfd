// gen_openlb_geom.cpp — STAGE A driver for the OpenLB migration.
//
// Replicates forward_city.cpp's scene build (density city → solid-building voxelization
// → source set Ω) and then emits the OpenLB Stage-B inputs:
//     material_map.dat  (OpenLB SuperGeometry material numbers, openlb_geometry.h)
//     source_mask.u8    (Ω burst-release cells — identical to forward_city's writer)
//     geom_type.u8      (raw voxel cell types — kept for cross-checking / viz)
//     meta_geom.txt     (self-describing metadata)
// It prints the voxel summary and the material-map reconciliation, which is the Step-2
// gate ("material counts match the old voxelizer").
//
// The city-build knobs mirror forward_city.cpp's defaults (except BUF_DOWN, see below) so
// the city itself is the SAME; every knob is env-overridable. Grid resolution is the compile-time
// CELL_SIZE_M (build with -DCELL_SIZE_M=2.0 to match the coarse production run).
//
// Build:  g++ -O3 -std=c++17 -DCELL_SIZE_M=4.0 gen_openlb_geom.cpp -o gen_openlb_geom
// Run:    OUT_DIR=geom_out ./gen_openlb_geom

#include "city_zoning.h"
#include "voxelize.h"
#include "openlb_geometry.h"
#include "occupancy.h"
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace city;
static double envd(const char* k, double d){ const char* e=getenv(k); return e? atof(e):d; }
static int    envi(const char* k, int    d){ const char* e=getenv(k); return e? atoi(e):d; }

int main(){
    setvbuf(stdout, nullptr, _IOLBF, 0);
    std::string OUT = getenv("OUT_DIR") ? getenv("OUT_DIR") : "geom_out";
    { std::string c = "mkdir -p '" + OUT + "'"; if (system(c.c_str())) {} }

    // ── city morphology — forward_city.cpp defaults ──
    const double CITY_M = envd("CITY_M", 600.0);
    const double POP    = envd("POP",   20000.0);
    // Domain extents follow COST Action 732 (Franke et al. 2007) / AIJ (Tominaga et al. 2008):
    // 5 H upstream, 6 H to each side (5 H gives 3.1 % blockage here), 5 H above the tallest building, 15 H downstream, with
    // H = H_REF_M for every design (a domain that resized with the design would make J step
    // discontinuous). DOMAIN=compact restores the pre-guideline extents (40 m upstream, 35 m
    // lateral, 3 x the design's own height above) for quick tests only.
    const bool COMPACT = getenv("DOMAIN") && std::string(getenv("DOMAIN")) == "compact";
    // Downstream buffer: 15 x the tallest building (COST 732, Franke et al. 2007) so the
    // city's wake closes before the outlet. At 70 m (inherited from forward_city) the near-
    // ground flow was reversed over the last ~100 m, i.e. the wake reached the pressure
    // outlet. Deliberately a CONSTANT rather than 15*maxH of the current design: a domain
    // that resizes with the design makes J step-discontinuous across designs (the same reason
    // nz_cost732 replaced power-of-two rounding). 88 m is the production city's tallest
    // building at the default knobs; re-derive it if the design space grows taller.
    // Builder revision 9 (default): heights scaled so total floor area houses the population,
    // capped at the height limit H_MAX, which therefore also sets the domain (every design
    // fits the same guideline domain). BUILDER=legacy: revision 8 and the earlier fixed 88 m.
    const bool LEGACY_BUILDER = getenv("BUILDER") && std::string(getenv("BUILDER")) == "legacy";
    const double H_MAX = envd("H_MAX", 100.0);
    const double H_REF_M = LEGACY_BUILDER ? 88.0 : H_MAX;
    const double BUF_DOWN  = envd("BUF_DOWN",  15.0 * H_REF_M);   // 1320 m
    const double BUF_UP    = envd("BUF_UP",  COMPACT ? 40.0 : 5.0 * H_REF_M);    // 440 m
    // 6 H rather than 5 H: at 5 H the production city still blocks 3.1 % (> 3 %)
    const double BUF_LAT   = envd("BUF_LAT", COMPACT ? 35.0 : 6.0 * H_REF_M);    // 528 m
    const double HEADROOM_H = envd("HEADROOM_H", COMPACT ? 3.0 : 5.0);         // above H_REF_M
    // Wind (toward, deg from +x) in [0, 90]: 0 = the original +x inflow; an oblique angle puts the
    // inflow on the x=0 and y=0 faces and the outflow on the other two (openlb_geometry.h). The
    // other rose directions come from ORIENT, an exact grid symmetry of the city (below), so only
    // 0 < WIND_DEG <= 45 is ever needed (tools/wind_rose.py maps a direction to the pair).
    const double WIND_DEG   = envd("WIND_DEG",   0.0);
    const int    ORIENT     = envi("ORIENT", 0);
    if (WIND_DEG < 0 || WIND_DEG > 90 || ORIENT < 0 || ORIENT > 7) {
        fprintf(stderr, "[gen_openlb_geom] WIND_DEG must be in [0, 90] and ORIENT in 0..7 (got %g, %d)\n", WIND_DEG, ORIENT);
        return 2; }
    if (COMPACT && WIND_DEG != 0) { fprintf(stderr, "[gen_openlb_geom] oblique wind needs the guideline domain\n"); return 2; }
    // Buffers by face. Upwind faces: 5 H (x) and the 6 H lateral (y, >= 5 H). Downwind faces:
    // the 15 H wake length projected on each axis, at least the 6 H lateral. At 0 deg this is
    // exactly the axis layout (5 H / 15 H in x, 6 H each side in y).
    const double wrad = WIND_DEG * M_PI / 180.0;
    const double BUF_XM = WIND_DEG < 90 ? BUF_UP : BUF_LAT;
    const double BUF_YM = BUF_LAT;
    const double BUF_XP = std::max(BUF_LAT, BUF_DOWN * std::cos(wrad));
    const double BUF_YP = std::max(BUF_LAT, BUF_DOWN * std::sin(wrad));

    Params p{};
    p.city_w = CITY_M; p.city_h = CITY_M;
    p.population_total = POP;
    p.wind_direction = 0.0;
    p.base_height    = BASE_HEIGHT_M;
    p.block_w   = envd("BLOCK_W", 60.0);
    p.block_d   = envd("BLOCK_D", 60.0);
    p.roughness = envd("ROUGH",    0.5);
    p.road_w_x  = p.road_w_y = envd("STREET_W", 20.0);
    p.park_fraction = envd("PARK_FRAC", 0.15);
    p.cbd_peak = 40.0; p.cbd_decay = 8e-6; p.patchiness = 0.0;
    p.cbd_aspect=1; p.cbd_angle=0; p.biz_inner_frac=0; p.biz_aspect=1;
    p.park_centrality = 0.5;
    p.buf_xn = BUF_XM; p.buf_xp = BUF_XP; p.buf_yn = BUF_YM; p.buf_yp = BUF_YP;
    p.rev = LEGACY_BUILDER ? 8 : 9;
    p.h_max = LEGACY_BUILDER ? 0.0 : H_MAX;
    p.Sx = p.buf_xn + p.city_w + p.buf_xp;
    p.Sy = p.buf_yn + p.city_h + p.buf_yp;
    double dom_cx = p.buf_xn + p.city_w*0.5, dom_cy = p.buf_yn + p.city_h*0.5;
    p.cbd_x=dom_cx; p.cbd_y=dom_cy; p.biz_center_x=dom_cx; p.biz_center_y=dom_cy;
    p.source_x=p.buf_xn-10; p.source_y=dom_cy;

    MixedZoningParams mzp{};
    mzp.dens_enable       = true;
    mzp.peak_height       = envd("PEAK_H",         45.0);   // legacy builder only
    mzp.height_conc       = envd("HEIGHT_CONC",     0.6);   // rev 9: 0 uniform canopy .. 1 heights follow density
    mzp.emp_centers       = envi("EMP_CENTERS",     3);
    mzp.emp_centrality    = envd("EMP_CTR",         0.55);
    mzp.emp_concentration = envd("EMP_CONC",        0.5);
    mzp.res_gradient      = envd("RES_GRAD",        1.0);
    mzp.jh_mix            = envd("JH_MIX",          0.5);
    mzp.biz_scatter  = 0.0; mzp.park_scatter = 0.0;

    printf("[gen_openlb_geom] density city: %.0fx%.0f m, pop=%.0f, dx=%.2f m, wind=%.0f deg\n",
           CITY_M, CITY_M, POP, (double)CELL, WIND_DEG);
    Result r0 = generate(p);
    Result r  = rezone(p, r0, mzp);
    printf("[gen_openlb_geom] city: blocks=%d biz=%d maxH=%.0fm pop=%.0f inhab=%.0f\n",
           r.num_blocks, r.counts[1], r.max_height, r.population, r.total_inhabitance);
    // ORIENT: rotate / mirror the finished layout about the city centre by an element of the
    // square's symmetry group: k = 0..3 rotate 90 k deg counter-clockwise, k = 4..7 mirror x
    // first. Exact on the cell grid (square city, whole cells), so the voxels, release set and
    // receptors of an oriented city are those of the original, moved.
    if (ORIENT) {
        const int W = (int)std::lround(CITY_M / CELL), cx0 = (int)std::lround(p.buf_xn / CELL), cy0 = (int)std::lround(p.buf_yn / CELL);
        auto pt = [&](double u, double v, double& uo, double& vo) {
            if (ORIENT >= 4) u = W - u;
            for (int i = 0; i < (ORIENT & 3); ++i) { const double t = u; u = W - v; v = t; }
            uo = u; vo = v; };
        auto rect = [&](int& x0, int& x1, int& y0, int& y1) {
            double ax, ay, bx, by; pt(x0 - cx0, y0 - cy0, ax, ay); pt(x1 - cx0, y1 - cy0, bx, by);
            const int nx0 = cx0 + (int)std::lround(std::min(ax, bx)), nx1 = cx0 + (int)std::lround(std::max(ax, bx));
            const int ny0 = cy0 + (int)std::lround(std::min(ay, by)), ny1 = cy0 + (int)std::lround(std::max(ay, by));
            x0 = nx0; x1 = nx1; y0 = ny0; y1 = ny1; };
        for (auto& b : r.blocks) {
            rect(b.x0, b.x1, b.y0, b.y1); rect(b.bx0, b.bx1, b.by0, b.by1);
            double uo, vo; pt(b.center_x / CELL - cx0, b.center_y / CELL - cy0, uo, vo);
            b.center_x = (cx0 + uo) * CELL; b.center_y = (cy0 + vo) * CELL;
        }
        printf("[gen_openlb_geom] city oriented by symmetry %d (%s %d deg)\n", ORIENT,
               ORIENT >= 4 ? "mirror x, then rotate" : "rotate", 90 * (ORIENT & 3));
    }
    if (!LEGACY_BUILDER)
        printf("[gen_openlb_geom] floor area %.0f m2 for a need of %.0f m2 (height scale %.3f, height limit %.0f m)%s\n",
               r.gfa_m2, r.gfa_target_m2, r.height_scale, H_MAX, r.feasible ? "" : "  INFEASIBLE");

    // STATS_ONLY=1: layout statistics on one line, no voxels or files (parameter surveys and
    // the optimizer's feasibility pre-check). Frontal area for wind along +x.
    if (envi("STATS_ONLY", 0)) {
        double cityA = CITY_M * CITY_M, foot = 0, front = 0, hsum = 0, hfa = 0, fa = 0; int nb = 0, npk = 0;
        for (const auto& b : r.blocks) {
            if (b.usage == PARK) { ++npk; continue; }
            const double h = b.height_cells * (double)CELL, A = b.footprint_m2, f = A * b.n_floors;
            ++nb; foot += A; front += (b.y1 - b.y0) * (double)CELL * h; hsum += h; hfa += h * f; fa += f;
        }
        printf("STATS feasible=%d buildings=%d parks=%d max_h=%.1f mean_h=%.2f fa_weighted_h=%.2f lambda_p=%.4f "
               "lambda_f=%.4f gfa=%.0f gfa_target=%.0f height_scale=%.4f biz=%d\n",
               (int)r.feasible, nb, npk, r.max_height, nb ? hsum / nb : 0.0, fa > 0 ? hfa / fa : 0.0,
               foot / cityA, front / cityA, r.gfa_m2, r.gfa_target_m2, r.height_scale, r.counts[1]);
        return r.feasible ? 0 : 3;
    }
    if (!r.feasible) {
        fprintf(stderr, "[gen_openlb_geom] INFEASIBLE: at the %.0f m height limit the blocks hold %.0f of the %.0f m2 "
                "of floor area the population needs\n", H_MAX, r.gfa_m2, r.gfa_target_m2);
        return 3;
    }
    if (r.max_height > H_REF_M)
        printf("[gen_openlb_geom] WARNING: tallest building %.0f m exceeds H_REF_M = %.0f m; "
               "BUF_DOWN = %.0f m is under 15 H_max for this design\n", r.max_height, H_REF_M, BUF_DOWN);

    // ── voxelize solid buildings; relaxed vertical headroom (matches forward_city) ──
    int maxHC=0; for(auto& b: r.blocks) maxHC=std::max(maxHC, b.height_cells);
    // guideline: fixed height (1 + HEADROOM_H) x H_REF_M for every design; compact: the old
    // design-dependent relaxed headroom
    int nz_relaxed = COMPACT ? std::max(4, maxHC + (int)std::ceil(HEADROOM_H*(double)maxHC) + 1)
                             : (int)std::ceil((1.0 + HEADROOM_H) * H_REF_M / (double)CELL) + 1;
    // size-resolved surface deposition (Zhang 2001) for a representative bin — feeds the
    // Step-4 deposition sink via dep_vel.f32. DEP_DP<=0 falls back to constant per-usage v_d.
    const double DEP_DP=envd("DEP_DP",2.5e-6), DEP_RHO=envd("DEP_RHO",1800.0), DEP_USTAR=envd("DEP_USTAR",0.5);
    VoxelGrid g = voxelize(p, r, /*solid_buildings=*/true, DEP_DP, DEP_RHO, DEP_USTAR, /*nz_fixed=*/nz_relaxed);
    print_voxel_summary(g);
    // blockage ratio: the buildings' frontal area seen by the (+x) wind over the domain
    // cross-section above the ground; COST 732 asks for < 3 %
    double blockage = 0;
    { long fa = 0;
      for (int z = 1; z < g.nz; ++z) for (int y = 0; y < g.ny; ++y) {
          bool hit = false;
          for (int x = 0; x < g.nx && !hit; ++x) { size_t id = g.idx(x, y, z);
              hit = g.type[id] == CELL_SHELL && g.perm[id] < 0.5f; }
          if (hit) ++fa; }
      blockage = (double)fa / ((double)g.ny * (g.nz - 1));
      if (WIND_DEG > 0) {                // oblique: the y-facing cross-section matters as well
          long fy = 0;
          for (int z = 1; z < g.nz; ++z) for (int x = 0; x < g.nx; ++x) {
              bool hit = false;
              for (int y = 0; y < g.ny && !hit; ++y) { size_t id = g.idx(x, y, z); hit = g.type[id] == CELL_SHELL && g.perm[id] < 0.5f; }
              if (hit) ++fy; }
          blockage = std::max(blockage, (double)fy / ((double)g.nx * (g.nz - 1)));
      }
      printf("[gen_openlb_geom] domain %s: upstream %.0f m, lateral %.0f m, top %.0f m, downstream %.0f m; "
             "blockage %.2f%% %s\n", COMPACT ? "COMPACT (not guideline)" : "COST 732", p.buf_xn, std::max(p.buf_yn, p.buf_yp),
             (g.nz - 1) * (double)CELL, p.buf_xp, 100 * blockage, blockage < 0.03 ? "(< 3 %, ok)" : "(ABOVE the 3 % guideline)"); }

    const int nx=g.nx, ny=g.ny, nz=g.nz; const size_t N=(size_t)nx*ny*nz;
    const double dx = g.cell_size;
    auto IDX=[&](int x,int y,int z){ return (size_t)z*ny*nx + (size_t)y*nx + x; };

    // ── release set Ω ──
    // environs (default): one release cell per plan cell over the city plus a ring of
    // ENVIRON_FRAC x city size around it (300 m here, inside the guideline buffers), at ground
    // level, or on the first fluid cell above the roof where a building stands. Uniform per unit
    // area and the same plan area for every design, so J compares designs on one release
    // distribution. release_zone.u8 tiles it ZONES x ZONES (tail constraint, labelled tracers).
    // legacy: forward_city's rule, open ground cells (streets + parks) over the city + 40 m
    // upstream, 35 m lateral, OMEGA_DOWN = 70 m downstream; design-dependent (no roofs), kept for
    // comparison with the earlier results. Default for DOMAIN=compact, where the ring does not fit.
    const bool OMEGA_LEGACY = getenv("OMEGA") ? std::string(getenv("OMEGA")) == "legacy" : COMPACT;
    const double ENV_FRAC = envd("ENVIRON_FRAC", 0.5);
    const int ZONES = std::max(1, std::min(15, envi("ZONES", 4)));
    double ringUp, ringDown, ringLat;
    if (OMEGA_LEGACY) { ringUp = 40.0; ringLat = 35.0; ringDown = envd("OMEGA_DOWN", 70.0); }
    else { ringUp = ringDown = ringLat = ENV_FRAC * CITY_M; }
    if (ringUp > p.buf_xn + 1e-9 || ringLat > std::min(p.buf_yn, p.buf_yp) + 1e-9 || ringDown > p.buf_xp + 1e-9) {
        fprintf(stderr, "[gen_openlb_geom] FATAL release ring (%.0f/%.0f/%.0f m) does not fit the buffers "
                "(%.0f/%.0f/%.0f m); use OMEGA=legacy or larger buffers\n", ringUp, ringLat, ringDown, BUF_UP, BUF_LAT, BUF_DOWN);
        return 2;
    }
    // legacy keeps the old loop's bounds exactly (it reached the boundary rows); environs stays
    // one cell inside the domain faces
    const int lo = OMEGA_LEGACY ? 0 : 1;
    const int rx0 = std::max(lo, (int)std::lround((p.buf_xn - ringUp) / dx));
    const int rx1 = OMEGA_LEGACY ? std::min(nx, (int)std::floor((p.buf_xn + CITY_M + ringDown) / dx))   // exclusive
                                 : std::min(nx - 1, (int)std::lround((p.buf_xn + CITY_M + ringDown) / dx));
    const int ry0 = std::max(lo, (int)std::lround((p.buf_yn - ringLat) / dx));
    const int ry1 = std::min(ny - lo, (int)std::lround((p.buf_yn + CITY_M + ringLat) / dx));
    std::vector<uint8_t> srcmask(N, 0), zone(N, 0);
    long nOmega = 0, nRoof = 0;
    for (int y = ry0; y < ry1; ++y) for (int x = rx0; x < rx1; ++x) {
        int z = 1;
        if (OMEGA_LEGACY) {
            const size_t id = IDX(x, y, z); const uint8_t t = g.type[id];
            if (!(t == CELL_FLUID || (t == CELL_SHELL && g.perm[id] > 0.5f))) continue;   // streets + parks
        } else {
            while (z < nz - 1 && g.type[IDX(x, y, z)] == CELL_SHELL && g.perm[IDX(x, y, z)] < 0.5f) ++z;   // above a roof
            if (z >= nz - 1) continue;
            if (z > 1) ++nRoof;
        }
        const size_t id = IDX(x, y, z);
        srcmask[id] = 1; ++nOmega;
        const int zx = std::min(ZONES - 1, (x - rx0) * ZONES / std::max(1, rx1 - rx0));
        const int zy = std::min(ZONES - 1, (y - ry0) * ZONES / std::max(1, ry1 - ry0));
        zone[id] = (uint8_t)(1 + zx * ZONES + zy);
    }
    printf("[gen_openlb_geom] release Omega (%s): x %.0f..%.0f m, y %.0f..%.0f m; %ld cells (%ld on roofs) in %d zones\n",
           OMEGA_LEGACY ? "legacy: ground, streets + parks" : "environs: uniform, ground or roof",
           rx0 * dx, rx1 * dx, ry0 * dx, ry1 * dx, nOmega, nRoof, ZONES * ZONES);

    // ── per-cell surface deposition velocity (size-resolved) → Step-4 sink ──
    { FILE* f=fopen((OUT+"/dep_vel.f32").c_str(),"wb");
      if(f){ int h[5]={nx,ny,nz,(int)std::lround(dx*1000.0),1}; fwrite(h,sizeof(int),5,f);
             fwrite(g.dep_vel.data(),sizeof(float),N,f); fclose(f);
             printf("[gen_openlb_geom] wrote dep_vel.f32 (rep. bin dp=%.2g m)\n", DEP_DP);} }

    // ── receptor field w: person-time per cell, so that J = <w, Theta> / M is dose per release ──
    // The builder already turns population into person-time per building (eff_inh, NHAPS
    // time budget, city_builder7.h): residents x 0.69 (home), workers x 0.18 (other indoor),
    // park visitors x 0.038. The remaining 0.093 (in-vehicle + sidewalk) is the street
    // pathway of occupancy.h. Each pathway breathes:
    //   indoor  : the air at the building's envelope (fluid cells touching walls or roof),
    //             reduced by the infiltration factor F_INF (Chen & Zhao 2011, PM2.5 0.3-0.82);
    //   park    : the air in the park's ground-level cells, unfiltered;
    //   street  : ground-level road cells next to each building, weighted by that building's
    //             person-time and normalised to the street total (occupancy.h), unfiltered.
    // RECEPTOR=legacy restores forward_city's weights (indoor x F_IN 0.87 on top of the time
    // budget, 0.075 per cell in a band around the built area, no parks) for comparisons.
    const bool REC_LEGACY = getenv("RECEPTOR") && std::string(getenv("RECEPTOR")) == "legacy";
    const double z_ped = envd("Z_PED", 2.0);
    float FINF=(float)envd("FINF",0.62), F_IN=(float)envd("F_IN",0.87), F_OUT=(float)envd("F_OUT",0.075);
    if(FINF>1.f) FINF=1.f;
    const bool FLAT_W = envi("FLAT_W",0)!=0;
    const int zped = std::min(nz-1, std::max(1,(int)std::llround(z_ped/dx)));
    std::vector<float> w(N,0.f); std::vector<int> stamp(N,-1);
    // RECEPTOR_PARTS=1: also write the weight of each pathway separately (receptor_w_indoor /
    // _park / _street.f32), so J can be split by where people are (tests/pathways.py)
    const bool PARTS = envi("RECEPTOR_PARTS", 0) != 0;
    std::vector<float> wIn(PARTS ? N : 0, 0.f), wPk(PARTS ? N : 0, 0.f), wSt(PARTS ? N : 0, 0.f);
    double wEnv=0, wPark=0; long nEnvTot=0, nParkTot=0; int bxmin=nx,bxmax=-1,bymin=ny,bymax=-1;
    for(int bi=0;bi<(int)r.blocks.size();++bi){
        const Block& b=r.blocks[bi];
        int x0=std::max(0,b.x0),x1=std::min(nx,b.x1),y0=std::max(0,b.y0),y1=std::min(ny,b.y1);
        if(x1<=x0||y1<=y0) continue;
        if(b.usage==PARK){                                   // park visitors: ground-level park air
            if(REC_LEGACY || b.eff_inh<=0.0) continue;
            std::vector<size_t> cells;
            for(int y=y0;y<y1;++y)for(int x=x0;x<x1;++x){ size_t id=IDX(x,y,zped);
                if(g.type[id]==CELL_FLUID || (g.type[id]==CELL_SHELL && g.perm[id]>0.5f)) cells.push_back(id); }
            if(cells.empty()) continue;
            const double per=b.eff_inh/(double)cells.size();
            for(size_t id:cells){ w[id]+=(float)per; wPark+=per; ++nParkTot; if(PARTS) wPk[id]+=(float)per; }
            continue;
        }
        int hc=b.height_cells; if(hc<1) continue;
        int hz=std::min(hc,nz-1);
        bxmin=std::min(bxmin,x0);bxmax=std::max(bxmax,x1-1);bymin=std::min(bymin,y0);bymax=std::max(bymax,y1-1);
        if(b.eff_inh<=0.0) continue;
        std::vector<size_t> env; const int dxn[6]={1,-1,0,0,0,0},dyn[6]={0,0,1,-1,0,0},dzn[6]={0,0,0,0,1,-1};
        for(int z=1;z<=hz;++z)for(int y=y0;y<y1;++y)for(int x=x0;x<x1;++x){
            size_t sid=IDX(x,y,z); if(g.type[sid]!=CELL_SHELL) continue;
            for(int d=0;d<6;++d){ int xx=x+dxn[d],yy=y+dyn[d],zz=z+dzn[d];
                if(xx<0||xx>=nx||yy<0||yy>=ny||zz<1||zz>=nz) continue;
                size_t nid=IDX(xx,yy,zz); if(g.type[nid]!=CELL_FLUID) continue;
                if(stamp[nid]==bi) continue; stamp[nid]=bi; env.push_back(nid); } }
        if(env.empty()) continue;
        const double fin = REC_LEGACY ? (double)F_IN : 1.0;   // legacy counted indoor time twice
        double per=fin*FINF*(FLAT_W?1.0:b.eff_inh)/(double)env.size();
        for(size_t nid:env){ w[nid]+=(float)per; wEnv+=per; ++nEnvTot; if(PARTS) wIn[nid]+=(float)per; }
    }
    double wPed=0; long nPed=0;
    if(REC_LEGACY){
      if(bxmax>=bxmin){ int mg=std::max(2,(int)std::llround(10.0/dx));
        int px0=std::max(1,bxmin-mg),px1=std::min(nx-2,bxmax+mg),py0=std::max(1,bymin-mg),py1=std::min(ny-2,bymax+mg);
        int z=std::min(zped,nz-2);
        for(int y=py0;y<=py1;++y)for(int x=px0;x<=px1;++x){ size_t id=IDX(x,y,z);
            if(g.type[id]==CELL_FLUID){ w[id]+=F_OUT; wPed+=F_OUT; ++nPed; if(PARTS) wSt[id]+=F_OUT; } } }
    } else {
        // street occupants (occupancy.h): P x (in-vehicle + sidewalk share of outdoors)
        const double streetPop = r.population * street_fraction(nhaps_average(), OUTDOOR_PARK_SHARE);
        StreetField sf = build_street_occupancy(p, r, g, streetPop);
        for(int y=0;y<ny;++y)for(int x=0;x<nx;++x){ const float v=sf.weight[(size_t)y*nx+x];
            if(v>0.f){ w[IDX(x,y,sf.z)]+=v; wPed+=v; ++nPed; if(PARTS) wSt[IDX(x,y,sf.z)]+=v; } }
    }
    const double wsum=wEnv+wPark+wPed;
    if (PARTS) for (auto pr : {std::make_pair("indoor", &wIn), std::make_pair("park", &wPk), std::make_pair("street", &wSt)}) {
        FILE* f=fopen((OUT+"/receptor_w_"+pr.first+".f32").c_str(),"wb");
        if(f){ int h[5]={nx,ny,nz,(int)std::lround(dx*1000.0),1}; fwrite(h,sizeof(int),5,f);
               fwrite(pr.second->data(),sizeof(float),N,f); fclose(f);} }
    { FILE* f=fopen((OUT+"/receptor_w.f32").c_str(),"wb");
      if(f){ int h[5]={nx,ny,nz,(int)std::lround(dx*1000.0),1}; fwrite(h,sizeof(int),5,f);
             fwrite(w.data(),sizeof(float),N,f); fclose(f);} }
    printf("[gen_openlb_geom] receptor w (%s): indoor %.4g on %ld envelope cells (F_INF %.2f), park %.4g on %ld cells, "
           "street %.4g on %ld cells; wsum %.4g (population %.0f)\n", REC_LEGACY ? "legacy" : "person-time",
           wEnv, nEnvTot, (double)FINF, wPark, nParkTot, wPed, nPed, wsum, r.population);
    if(wsum<=0){ fprintf(stderr,"[gen_openlb_geom] WARN empty receptor field\n"); }

    // ── build + write the OpenLB material map ──
    MaterialMap mm = build_material_map(g, WIND_DEG);
    export_material_map((OUT+"/material_map.dat").c_str(), mm);

    // raw voxel types + source mask (5-int / 4-int compatible with the rest of the repo)
    { FILE* f=fopen((OUT+"/geom_type.u8").c_str(),"wb");
      if(f){ int h[5]={nx,ny,nz,(int)std::lround(dx*1000.0),1}; fwrite(h,sizeof(int),5,f); fwrite(g.type.data(),1,N,f); fclose(f);} }
    { FILE* f=fopen((OUT+"/source_mask.u8").c_str(),"wb");
      if(f){ int h[5]={nx,ny,nz,(int)std::lround(dx*1000.0),1}; fwrite(h,sizeof(int),5,f); fwrite(srcmask.data(),1,N,f); fclose(f);} }
    { FILE* f=fopen((OUT+"/release_zone.u8").c_str(),"wb");
      if(f){ int h[5]={nx,ny,nz,(int)std::lround(dx*1000.0),1}; fwrite(h,sizeof(int),5,f); fwrite(zone.data(),1,N,f); fclose(f);} }
    // per-cell surface DRY-DEPOSITION velocity (m/s) — consumed by Stage B's deposition
    // sink (Step 4). 0 on fluid/indoor; usage-specific on building/park/ground faces.
    { FILE* f=fopen((OUT+"/dep_vel.f32").c_str(),"wb");
      if(f){ int h[5]={nx,ny,nz,(int)std::lround(dx*1000.0),1}; fwrite(h,sizeof(int),5,f); fwrite(g.dep_vel.data(),sizeof(float),N,f); fclose(f);} }

    // ── Step-2 GATE: reconcile material counts against the voxel grid ──
    bool gate = material_reconcile(g, mm, /*verbose=*/true);

    // ── metadata ──
    { FILE* f=fopen((OUT+"/meta_geom.txt").c_str(),"w"); if(f){
        fprintf(f,"# gen_openlb_geom — OpenLB Stage-A geometry-bridge metadata\n");
        fprintf(f,"grid_nx %d\ngrid_ny %d\ngrid_nz %d\ndx_m %.4f\n",nx,ny,nz,dx);
        fprintf(f,"wind_deg %.4f\norient %d\nbuf_xm_m %.1f\nbuf_xp_m %.1f\nbuf_ym_m %.1f\nbuf_yp_m %.1f\n",
                WIND_DEG, ORIENT, p.buf_xn, p.buf_xp, p.buf_yn, p.buf_yp);
        fprintf(f,"receptor_mode %s\nw_indoor %.6g\nw_park %.6g\nw_street %.6g\nf_inf %.4f\npopulation %.1f\n",
                REC_LEGACY ? "legacy" : "person_time", wEnv, wPark, wPed, (double)FINF, r.population);
        fprintf(f,"omega_source_cells %ld\nomega_mode %s\nomega_roof_cells %ld\nenviron_frac %.3f\nrelease_zones %d\n",
                nOmega, OMEGA_LEGACY ? "legacy" : "environs", nRoof, OMEGA_LEGACY ? 0.0 : ENV_FRAC, ZONES * ZONES);
        fprintf(f,"material_scheme 0=donothing 1=fluid 2=wall(buildings) 3=inlet 4=outlet 5=slip 6=porous(parks) 7=ground(rough-wall)\n");
        fprintf(f,"material_map_layout 5xint32[nx,ny,nz,dx*1000,ncomp=1] then nx*ny*nz int32\n");
        fprintf(f,"reconcile_gate %s\n", gate?"PASS":"FAIL");
        fprintf(f,"builder_rev %d\nh_max_m %.1f\ngfa_m2 %.1f\ngfa_target_m2 %.1f\nheight_scale %.5f\n",
                p.rev, LEGACY_BUILDER ? 0.0 : H_MAX, r.gfa_m2, r.gfa_target_m2, r.height_scale);
        fprintf(f,"domain %s\nbuf_up_m %.1f\nbuf_lat_m %.1f\nbuf_down_m %.1f\nh_ref_m %.1f\nblockage %.5f\n",
                COMPACT ? "compact" : "cost732", BUF_UP, BUF_LAT, BUF_DOWN, H_REF_M, blockage);
        fclose(f);} }

    printf("[gen_openlb_geom] wrote material_map.dat, source_mask.u8, geom_type.u8, meta_geom.txt to %s/\n", OUT.c_str());
    printf("[gen_openlb_geom] STEP-2 GATE: %s\n", gate?"PASS":"FAIL");
    return gate ? 0 : 1;
}
