// demo_indoor.cpp — end-to-end indoor exposure with filtration, on REAL solver
// output. Small city -> SOLID buildings -> LBM (ABL inlet) -> outdoor C field
// -> per-building indoor exposure via infiltration + filtration sweep.
#include "city_builder7.h"
#include "voxelize.h"
#include "lbm_solver.h"
#include "indoor_exposure.h"
#include <cstdio>
#include <vector>
#include <cstdint>
using namespace city;

int main(){
    Params p{};
    p.city_w=120; p.city_h=120; p.block_w=40; p.block_d=24; p.base_height=8;
    p.cbd_peak=24; p.cbd_decay=2e-5; p.cbd_aspect=1; p.cbd_angle=0;
    p.biz_inner_frac=0.05; p.biz_aspect=1; p.park_centrality=0.5; p.park_fraction=0.10;
 p.roughness=0.15; p.road_w_x=20; p.road_w_y=20; p.population_total=20000;
    p.wind_direction=0;
    p.buf_xn=p.buf_xp=p.buf_yn=p.buf_yp=60;        // small buffers for a fast demo
    p.source_x=40; p.source_y=p.buf_yn+p.city_h*0.5; ensure_source_buffer(p);
    compute_domain(p);
    double cx=p.buf_xn+p.city_w*0.5, cy=p.buf_yn+p.city_h*0.5;
    p.cbd_x=cx;p.cbd_y=cy;p.biz_center_x=cx;p.biz_center_y=cy;

    Result r=generate(p); print_summary(p,r);
    VoxelGrid g=voxelize(p,r,/*solid_buildings=*/true);   // <-- SOLID buildings
    print_voxel_summary(g);
    int solid=0,indoor=0; for(auto t:g.type){ if(t==CELL_SHELL)solid++; if(t==CELL_INDOOR)indoor++; }
    printf("[CHECK] solid-building SHELL cells=%d (full bounce-back); INDOOR cells=%d are POROUS PARKS (occupants stay near-outdoor) — non-park occupants handled by infiltration\n",solid,indoor);

    lbm::Config c{};
    c.nx=g.nx;c.ny=g.ny;c.nz=g.nz;c.cell_size=g.cell_size;
    c.U_inlet=5;c.wind_angle=0;c.nu_phys=1.5e-5f;c.Cw=0.325f;c.Sc_t=0.7f;c.D_mol=1e-5f;
    c.source_x=p.source_x;c.source_y=p.source_y;c.source_z=4;c.Q_source=1;
    c.particle_diam=0;c.particle_density=0;                 // passive tracer field
    c.max_steps=20000;c.check_interval=200;c.conv_threshold=1e-4f;
    c.max_warmup=300;c.avg_threshold=8e-3f;c.release_time=12.0f;
    c.inlet_profile=1;c.abl_z0=0.7f;c.abl_zref=30.0f;c.abl_Lturb=30.0f;c.abl_nmodes=80;
    c.abl_sigu_ratio=2.5f;c.abl_sigv_ratio=1.9f;c.abl_sigw_ratio=1.25f;

    lbm::Solver s(c);
    s.load_geometry(g.type.data(),g.perm.data(),g.inh.data(),g.dep_vel.data());
    auto res=s.run();
    s.export_concentration("conc_demo.bin",1);

    // read outdoor concentration field
    FILE*f=fopen("conc_demo.bin","rb"); int hdr[2]; size_t rd=fread(hdr,4,2,f);(void)rd;
    int nx=hdr[0],ny=hdr[1]; std::vector<float> conc((size_t)nx*ny);
    rd=fread(conc.data(),4,(size_t)nx*ny,f); fclose(f);

    // ── filtration sweep (indoor air cleaning as a protective action) ──
    struct Sc{const char*name; double lam;};
    Sc scen[]={{"none (lambda=0)",0.0},{"portable cleaner (CADR/V~2/h)",2.0},
               {"HEPA (CADR/V~5/h)",5.0},{"strong (~10/h)",10.0}};
    printf("\n=== Indoor exposure vs indoor filtration (dp=0.5um mass-mean, a_inf=0.5/h) ===\n");
    double base=0;
    FILE*csv=fopen("indoor_demo.csv","w"); fprintf(csv,"lambda_filt,io,total_indoor_exposure,reduction_pct\n");
    for(auto&sc:scen){
        IndoorScenario S; S.dp_m=0.5e-6; S.a_inf=0.5; S.lambda_filt=sc.lam;
        auto ie=indoor_exposure(r,conc,nx,ny,S);
        if(sc.lam==0.0) base=ie.total_indoor_exposure;
        double red=(base>0)?100.0*(1.0-ie.total_indoor_exposure/base):0.0;
        printf("  %-28s io=%.3f  total_indoor_exposure=%.4e  (-%.0f%% vs none)  occ=%.0f\n",
               sc.name, ie.io, ie.total_indoor_exposure, red, ie.total_occupants);
        fprintf(csv,"%.1f,%.4f,%.6e,%.2f\n",sc.lam,ie.io,ie.total_indoor_exposure,red);
    }
    fclose(csv);
    printf("\n[mass budget] emitted=%.3e deposited=%.3e (%.1f%%) airborne=%.3e\n",
           res.total_emitted,res.total_deposited,
           100.0*res.total_deposited/std::max(1e-30,res.total_emitted),res.total_airborne);
    return 0;
}
