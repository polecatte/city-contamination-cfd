// demo_occupancy.cpp — three exposure pathways on a real run, using the 24-h
// NHAPS occupancy (no time-of-release assumption). Shows street occupants are
// the most exposed per-capita (full outdoor, no infiltration).
#include "city_builder7.h"
#include "voxelize.h"
#include "lbm_solver.h"
#include "indoor_exposure.h"
#include "occupancy.h"
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
    p.wind_direction=0; p.buf_xn=p.buf_xp=p.buf_yn=p.buf_yp=60;
    p.source_x=40; p.source_y=p.buf_yn+p.city_h*0.5; ensure_source_buffer(p); compute_domain(p);
    double cx=p.buf_xn+p.city_w*0.5, cy=p.buf_yn+p.city_h*0.5;
    p.cbd_x=cx;p.cbd_y=cy;p.biz_center_x=cx;p.biz_center_y=cy;
    Result r=generate(p);

    VoxelGrid g=voxelize(p,r,/*solid_buildings=*/true);
    lbm::Config c{}; c.nx=g.nx;c.ny=g.ny;c.nz=g.nz;c.cell_size=g.cell_size;
    c.U_inlet=5;c.wind_angle=0;c.nu_phys=1.5e-5f;c.Cw=0.325f;c.Sc_t=0.7f;c.D_mol=1e-5f;
    c.source_x=p.source_x;c.source_y=p.source_y;c.source_z=4;c.Q_source=1;
    c.particle_diam=0;c.particle_density=0;
    c.max_steps=20000;c.check_interval=200;c.conv_threshold=1e-4f;
    c.max_warmup=300;c.avg_threshold=8e-3f;c.release_time=12.0f;
    c.inlet_profile=1;c.abl_z0=0.7f;c.abl_zref=30;c.abl_Lturb=30;c.abl_nmodes=80;
    c.abl_sigu_ratio=2.5f;c.abl_sigv_ratio=1.9f;c.abl_sigw_ratio=1.25f;
    lbm::Solver s(c);
    s.load_geometry(g.type.data(),g.perm.data(),g.inh.data(),g.dep_vel.data());
    auto res=s.run(); s.export_concentration("conc_occ.bin",1);
    FILE*f=fopen("conc_occ.bin","rb"); int hdr[2]; size_t rd=fread(hdr,4,2,f);(void)rd;
    int nx=hdr[0],ny=hdr[1]; std::vector<float> conc((size_t)nx*ny);
    rd=fread(conc.data(),4,(size_t)nx*ny,f); fclose(f);

    // ── occupancy split (24-h NHAPS average; no time-of-day) ──
    double pop=r.population;
    TimeBudget tb=nhaps_average();
    double street_pop = street_fraction(tb, OUTDOOR_PARK_SHARE) * pop;

    // building (indoor) occupants and exposure via infiltration
    IndoorScenario sc; sc.dp_m=0.5e-6; sc.a_inf=0.5; sc.lambda_filt=0.0;
    auto ie = indoor_exposure(r, conc, nx, ny, sc);

    // park occupants and exposure (near-outdoor, sampled over park footprints)
    auto at=[&](int x,int y){ return (x<0||y<0||x>=nx||y>=ny)?0.0:(double)conc[(size_t)y*nx+x]; };
    double park_occ=0, park_exp=0;
    for(auto&b:r.blocks){ if(b.usage!=PARK||b.eff_inh<=0) continue;
        double s2=0;int n=0; for(int y=b.y0;y<b.y1;++y)for(int x=b.x0;x<b.x1;++x){s2+=at(x,y);++n;}
        double Co=(n>0)?s2/n:0; park_occ+=b.eff_inh; park_exp+=b.eff_inh*Co; }

    // street occupants and exposure (full outdoor, no infiltration)
    StreetField sf=build_street_occupancy(p,r,g,street_pop);
    double street_exp=street_exposure(sf,conc,nx,ny,/*vehicle_io=*/1.0);

    double indoor_occ=ie.total_occupants, indoor_exp=ie.total_indoor_exposure;
    double placed=indoor_occ+park_occ+street_pop;
    printf("\n=== Occupancy split (24-h NHAPS average; no time-of-release) ===\n");
    printf("  geometric population            : %.0f\n", pop);
    printf("  indoor (buildings)  occ=%.0f  exposure=%.4e  per-capita=%.4e\n",
           indoor_occ, indoor_exp, indoor_occ>0?indoor_exp/indoor_occ:0);
    printf("  park (porous)       occ=%.0f  exposure=%.4e  per-capita=%.4e\n",
           park_occ, park_exp, park_occ>0?park_exp/park_occ:0);
    printf("  STREET (road net)   occ=%.0f  exposure=%.4e  per-capita=%.4e\n",
           street_pop, street_exp, street_pop>0?street_exp/street_pop:0);
    double leak = pop - placed;
    printf("  placed=%.0f  population=%.0f  UNPLACED=%.0f (%.0f%%)\n",
           placed, pop, leak, 100.0*leak/pop);
    if (leak > 1.0)
        printf("  ^ CONSERVATION LEAK: work-time population dropped because this\n"
               "    test city has zero business floor area (no workplace to place\n"
               "    them in). Builder should reallocate unplaceable workers.\n");
    FILE*o=fopen("occupancy_demo.csv","w");
    fprintf(o,"pathway,occupants,exposure,per_capita\n");
    fprintf(o,"indoor,%.1f,%.6e,%.6e\n",indoor_occ,indoor_exp,indoor_occ>0?indoor_exp/indoor_occ:0);
    fprintf(o,"park,%.1f,%.6e,%.6e\n",park_occ,park_exp,park_occ>0?park_exp/park_occ:0);
    fprintf(o,"street,%.1f,%.6e,%.6e\n",street_pop,street_exp,street_pop>0?street_exp/street_pop:0);
    fclose(o);
    // export the street pedestrian-density field for visualization
    FILE*sfb=fopen("street_field.bin","wb"); int h2[2]={nx,ny};
    fwrite(h2,4,2,sfb); fwrite(sf.weight.data(),4,(size_t)nx*ny,sfb); fclose(sfb);
    return 0;
}
