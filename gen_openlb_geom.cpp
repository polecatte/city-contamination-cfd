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
    constexpr double H_REF_M = 88.0;
    const double BUF_DOWN  = envd("BUF_DOWN",  15.0 * H_REF_M);   // 1320 m
    const double BUF_UP    = envd("BUF_UP",  COMPACT ? 40.0 : 5.0 * H_REF_M);    // 440 m
    // 6 H rather than 5 H: at 5 H the production city still blocks 3.1 % (> 3 %)
    const double BUF_LAT   = envd("BUF_LAT", COMPACT ? 35.0 : 6.0 * H_REF_M);    // 528 m
    const double HEADROOM_H = envd("HEADROOM_H", COMPACT ? 3.0 : 5.0);         // above H_REF_M
    const double WIND_DEG   = envd("WIND_DEG",   0.0);

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
    p.buf_xn = BUF_UP; p.buf_xp = BUF_DOWN; p.buf_yn = p.buf_yp = BUF_LAT;
    p.Sx = p.buf_xn + p.city_w + p.buf_xp;
    p.Sy = p.buf_yn + p.city_h + p.buf_yp;
    double dom_cx = p.buf_xn + p.city_w*0.5, dom_cy = p.buf_yn + p.city_h*0.5;
    p.cbd_x=dom_cx; p.cbd_y=dom_cy; p.biz_center_x=dom_cx; p.biz_center_y=dom_cy;
    p.source_x=p.buf_xn-10; p.source_y=dom_cy;

    MixedZoningParams mzp{};
    mzp.dens_enable       = true;
    mzp.peak_height       = envd("PEAK_H",         45.0);
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
      printf("[gen_openlb_geom] domain %s: upstream %.0f m, lateral %.0f m, top %.0f m, downstream %.0f m; "
             "blockage %.2f%% %s\n", COMPACT ? "COMPACT (not guideline)" : "COST 732", BUF_UP, BUF_LAT,
             (g.nz - 1) * (double)CELL, BUF_DOWN, 100 * blockage, blockage < 0.03 ? "(< 3 %, ok)" : "(ABOVE the 3 % guideline)"); }

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
    if (ringUp > BUF_UP + 1e-9 || ringLat > BUF_LAT + 1e-9 || ringDown > BUF_DOWN + 1e-9) {
        fprintf(stderr, "[gen_openlb_geom] FATAL release ring (%.0f/%.0f/%.0f m) does not fit the buffers "
                "(%.0f/%.0f/%.0f m); use OMEGA=legacy or larger buffers\n", ringUp, ringLat, ringDown, BUF_UP, BUF_LAT, BUF_DOWN);
        return 2;
    }
    // legacy keeps the old loop's bounds exactly (it reached the boundary rows); environs stays
    // one cell inside the domain faces
    const int lo = OMEGA_LEGACY ? 0 : 1;
    const int rx0 = std::max(lo, (int)std::lround((BUF_UP - ringUp) / dx));
    const int rx1 = OMEGA_LEGACY ? std::min(nx, (int)std::floor((BUF_UP + CITY_M + ringDown) / dx))   // exclusive
                                 : std::min(nx - 1, (int)std::lround((BUF_UP + CITY_M + ringDown) / dx));
    const int ry0 = std::max(lo, (int)std::lround((BUF_LAT - ringLat) / dx));
    const int ry1 = std::min(ny - lo, (int)std::lround((BUF_LAT + CITY_M + ringLat) / dx));
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

    // ── receptor field w (F_inf infiltration weighting; ported from forward_city.cpp) ──
    // w = f_in·F_inf·(per-building inhabitance, spread over its outdoor envelope cells)
    //   + f_out (pedestrian band around the built area at z_ped). This is the Stage-C
    // contraction weight: J = (1/|Omega|) Σ_x w(x)·Θ(x).
    const double z_ped = envd("Z_PED", 2.0);
    float FINF=(float)envd("FINF",0.62), F_IN=(float)envd("F_IN",0.87), F_OUT=(float)envd("F_OUT",0.075);
    if(FINF>1.f) FINF=1.f;
    const bool FLAT_W = envi("FLAT_W",0)!=0;
    const int zped = std::min(nz-1, std::max(1,(int)std::llround(z_ped/dx)));
    std::vector<float> w(N,0.f); std::vector<int> stamp(N,-1);
    double wEnv=0; long nEnvTot=0; int bxmin=nx,bxmax=-1,bymin=ny,bymax=-1;
    for(int bi=0;bi<(int)r.blocks.size();++bi){
        const Block& b=r.blocks[bi];
        if(b.usage==PARK) continue; int hc=b.height_cells; if(hc<1) continue;
        int x0=std::max(0,b.x0),x1=std::min(nx,b.x1),y0=std::max(0,b.y0),y1=std::min(ny,b.y1);
        if(x1<=x0||y1<=y0) continue; int hz=std::min(hc,nz-1);
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
        double per=(double)F_IN*FINF*(FLAT_W?1.0:b.eff_inh)/(double)env.size();
        for(size_t nid:env){ w[nid]+=(float)per; wEnv+=per; ++nEnvTot; }
    }
    double wPed=0; long nPed=0;
    if(bxmax>=bxmin){ int mg=std::max(2,(int)std::llround(10.0/dx));
        int px0=std::max(1,bxmin-mg),px1=std::min(nx-2,bxmax+mg),py0=std::max(1,bymin-mg),py1=std::min(ny-2,bymax+mg);
        int z=std::min(zped,nz-2);
        for(int y=py0;y<=py1;++y)for(int x=px0;x<=px1;++x){ size_t id=IDX(x,y,z);
            if(g.type[id]==CELL_FLUID){ w[id]+=F_OUT; wPed+=F_OUT; ++nPed; } } }
    double wsum=wEnv+wPed;
    { FILE* f=fopen((OUT+"/receptor_w.f32").c_str(),"wb");
      if(f){ int h[5]={nx,ny,nz,(int)std::lround(dx*1000.0),1}; fwrite(h,sizeof(int),5,f);
             fwrite(w.data(),sizeof(float),N,f); fclose(f);} }
    printf("[gen_openlb_geom] receptor w: indoor %ld cells (w=%.4g) + outdoor %ld cells (w=%.4g), wsum=%.4g\n",
           nEnvTot,wEnv,nPed,wPed,wsum);
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
        fprintf(f,"wind_deg %.1f\n",WIND_DEG);
        fprintf(f,"omega_source_cells %ld\nomega_mode %s\nomega_roof_cells %ld\nenviron_frac %.3f\nrelease_zones %d\n",
                nOmega, OMEGA_LEGACY ? "legacy" : "environs", nRoof, OMEGA_LEGACY ? 0.0 : ENV_FRAC, ZONES * ZONES);
        fprintf(f,"material_scheme 0=donothing 1=fluid 2=wall(buildings) 3=inlet 4=outlet 5=slip 6=porous(parks) 7=ground(rough-wall)\n");
        fprintf(f,"material_map_layout 5xint32[nx,ny,nz,dx*1000,ncomp=1] then nx*ny*nz int32\n");
        fprintf(f,"reconcile_gate %s\n", gate?"PASS":"FAIL");
        fprintf(f,"domain %s\nbuf_up_m %.1f\nbuf_lat_m %.1f\nbuf_down_m %.1f\nh_ref_m %.1f\nblockage %.5f\n",
                COMPACT ? "compact" : "cost732", BUF_UP, BUF_LAT, BUF_DOWN, H_REF_M, blockage);
        fclose(f);} }

    printf("[gen_openlb_geom] wrote material_map.dat, source_mask.u8, geom_type.u8, meta_geom.txt to %s/\n", OUT.c_str());
    printf("[gen_openlb_geom] STEP-2 GATE: %s\n", gate?"PASS":"FAIL");
    return gate ? 0 : 1;
}
