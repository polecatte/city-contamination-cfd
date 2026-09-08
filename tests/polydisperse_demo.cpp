// polydisperse_demo.cpp — multi-bin (polydisperse) transport via warm-flow reuse.
// Warm the flow once; per activity bin, transport on the SAME deterministic flow
// with that bin's settling (Stokes+Cunningham) and size-dependent deposition
// (Zhang 2001); accumulate mass-weighted deposition (deposited-surface exposure) and air
// concentration (airborne inhalation). No kernel change (Option A).
#include "city_builder7.h"
#include "voxelize.h"
#include "lbm_solver.h"
#include "psd.h"
#include <cstdio>
#include <vector>
#include <cstdint>
using namespace city;

static std::vector<float> read_field(const char*fn,int&nx,int&ny){
    FILE*f=fopen(fn,"rb"); int h[2]; size_t r=fread(h,4,2,f);(void)r; nx=h[0];ny=h[1];
    std::vector<float> a((size_t)nx*ny); r=fread(a.data(),4,(size_t)nx*ny,f); fclose(f); return a;
}

int main(){
    Params p{};
    p.city_w=120;p.city_h=120;p.block_w=40;p.block_d=24;p.base_height=8;
    p.cbd_peak=24;p.cbd_decay=2e-5;p.cbd_aspect=1;p.cbd_angle=0;p.biz_inner_frac=0.05;p.biz_aspect=1;
p.park_centrality=0.5;p.park_fraction=0.10;p.roughness=0.15;p.road_w_x=20; p.road_w_y=20;
    p.population_total=20000;p.wind_direction=0;
    p.buf_xn=p.buf_xp=p.buf_yn=p.buf_yp=60;
    p.source_x=40;p.source_y=p.buf_yn+p.city_h*0.5;ensure_source_buffer(p);compute_domain(p);
    double cx=p.buf_xn+p.city_w*0.5,cy=p.buf_yn+p.city_h*0.5;
    p.cbd_x=cx;p.cbd_y=cy;p.biz_center_x=cx;p.biz_center_y=cy;
    Result r=generate(p);

    // Mass-weighted PSD (broad, to show size contrast): MMAD=3um GSD=2.2, 4 bins
    psd::Lognormal L{3.0e-6,2.2,1800.0};
    auto bins=psd::discretize(L,2);

    auto base_cfg=[&](VoxelGrid&g){
        lbm::Config c{}; c.nx=g.nx;c.ny=g.ny;c.nz=g.nz;c.cell_size=g.cell_size;
        c.U_inlet=5;c.wind_angle=0;c.nu_phys=1.5e-5f;c.Cw=0.325f;c.Sc_t=0.7f;c.D_mol=1e-5f;
        c.source_x=p.source_x;c.source_y=p.source_y;c.source_z=4;c.Q_source=1;
        c.particle_density=1800;c.check_interval=200;c.conv_threshold=1e-4f;c.max_steps=20000;
        c.inlet_profile=1;c.abl_z0=0.7f;c.abl_zref=30;c.abl_Lturb=30;c.abl_nmodes=64;
        c.abl_sigu_ratio=2.5f;c.abl_sigv_ratio=1.9f;c.abl_sigw_ratio=1.25f; return c;
    };

    // ── Pass 0: warm the flow once, save checkpoint ──
    VoxelGrid g0=voxelize(p,r,true);                       // solid buildings
    lbm::Config cw=base_cfg(g0); cw.particle_diam=0; cw.max_warmup=120; cw.release_time=0.05f;
    { lbm::Solver s(cw); s.load_geometry(g0.type.data(),g0.perm.data(),g0.inh.data(),g0.dep_vel.data());
      s.run(); s.save_flow_checkpoint("warm.ckpt"); }
    printf("\n[poly] warm flow saved; transporting %zu size bins on the same flow\n",bins.size());

    int nx=g0.nx,ny=g0.ny; std::vector<double> dep_tot((size_t)nx*ny,0), conc_tot((size_t)nx*ny,0);
    printf("\n=== per-bin transport (MMAD=3um GSD=2.2) ===\n");
    printf("bin  d_ae(um)  frac   dep_frac(%%)  airborne(%%)\n");
    for(size_t i=0;i<bins.size();++i){
        auto&b=bins[i];
        VoxelGrid g=voxelize(p,r,true,b.d_phys,1800.0,0.5);  // size-dependent deposition
        lbm::Config c=base_cfg(g); c.particle_diam=b.d_phys; c.max_warmup=0; c.release_time=8.0f;
        lbm::Solver s(c); s.load_geometry(g.type.data(),g.perm.data(),g.inh.data(),g.dep_vel.data());
        s.load_flow_checkpoint("warm.ckpt");
        auto res=s.run();
        char df[64],cf[64]; snprintf(df,64,"dep_bin%zu.bin",i); snprintf(cf,64,"conc_bin%zu.bin",i);
        s.export_deposition(df,1); s.export_concentration(cf,1);
        int bx,by; auto depf=read_field(df,bx,by); auto cof=read_field(cf,bx,by);
        for(size_t k=0;k<dep_tot.size();++k){ dep_tot[k]+=b.frac*depf[k]; conc_tot[k]+=b.frac*cof[k]; }
        double depfrac=100.0*res.total_deposited/std::max(1e-30,res.total_emitted);
        double airfrac=100.0*res.total_airborne/std::max(1e-30,res.total_emitted);
        printf(" %zu   %6.2f   %.3f   %8.2f    %8.2f\n",i,b.d_ae*1e6,b.frac,depfrac,airfrac);
    }
    // write mass-weighted totals
    FILE*f=fopen("dep_total.bin","wb"); int hdr[2]={nx,ny}; fwrite(hdr,4,2,f);
    std::vector<float> df(dep_tot.begin(),dep_tot.end()); fwrite(df.data(),4,df.size(),f); fclose(f);
    f=fopen("conc_total.bin","wb"); fwrite(hdr,4,2,f);
    std::vector<float> cf(conc_tot.begin(),conc_tot.end()); fwrite(cf.data(),4,cf.size(),f); fclose(f);
    printf("\n[poly] mass-weighted deposited-surface exposure field -> dep_total.bin, air field -> conc_total.bin\n");
    return 0;
}
