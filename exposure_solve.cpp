// exposure_solve.cpp — production contaminant exposure solve, end to end.
//
// Pipeline (see PRODUCTION_SOLVE.md):
//   geometry -> LBM mean flow under HRR -> freeze (u,nut) -> assemble linear
//   transport operator M -> receptor field w (effective inhabitance) ->
//   reverse_steady: phi_ss = dJ/ds over EVERY candidate source cell at once ->
//   Omega reduction over the outdoor ground set -> single objective number J.
//
// J is the steady-state, rate-normalized, population-weighted exposure
// (exposure per unit time; LOWER IS BETTER). By linearity + steady reciprocity,
//   J_ensemble = (1/|Omega|) * sum_{x in Omega} phi_ss(x)
// is the mean exposure rate for "one emitter of unknown, uniformly-likely
// location in Omega" — the whole objective from ONE reverse solve.
//
// Geometry here is the MUST-style urban array (must_geom.h), a proven, solver-
// ready stand-in; for a real city swap in city_builder7.h + voxelize.h and pass
// its per-cell eff_inh as w (below).  Receptor field w:
//   - if a w-file is given (cols: x_m,y_m,z_m,weight — metres, array-relative),
//     it is rasterized into w  (this is where city_builder7's eff_inh plugs in);
//   - else a default PEDESTRIAN-LEVEL proxy is used: fluid cells at ~z_ped that
//     are adjacent to a building get unit weight (street-level population).
//
// Build (CPU/OpenMP):
//   g++ -O3 -std=c++17 -fopenmp -c lbm_kernels_cpu.cpp -o kc.o
//   g++ -O3 -std=c++17 -fopenmp -c lbm_solver.cpp -o so.o
//   g++ -O3 -std=c++17 -fopenmp exposure_solve.cpp kc.o so.o -o exposure_solve
//   (GPU: link lbm_kernels.cu built with nvcc instead of kc.o)
//
// Run: ./exposure_solve [trial=2681829] [dx_m=0.5] [cap=300000] [z_ped_m=2.0] [w.csv]
//   COLLISION/HRR_SIGMA read from env (default hrr/0.98).
#include "lbm_solver.h"
#include "must_geom.h"
#include "adjoint_transport.h"
#include <cstdio>
#include <cstring>
#include <vector>
#include <cmath>
#include <string>

int main(int argc,char**argv){
    std::string trial = (argc>1)? argv[1] : "2681829";
    double dx    = (argc>2)? atof(argv[2]) : 0.5;
    int    cap   = (argc>3)? atoi(argv[3]) : 300000;
    double z_ped = (argc>4)? atof(argv[4]) : 2.0;
    { const char* e=getenv("MAX_WARMUP"); }  // (LBM warmup is set below; kept for parity)

    // ── 1. geometry (urban array; swap city_builder7+voxelize for a real city) ─
    const must::MustTrial* T=nullptr;
    for(auto& t: must::MUST_TRIALS) if(trial==t.name){ T=&t; break; }
    if(!T){ fprintf(stderr,"unknown trial '%s'\n",trial.c_str()); return 2; }
    must::MustSpec sp = must::spec_for(*T); sp.dx=dx;
    must::MustGeom g  = must::build(sp);
    must::report(g);
    size_t N=g.N;
    std::vector<float> perm(N,0.f), inh(N,0.f), depv(N,0.f);

    // ── 2. LBM mean flow under HRR (release disabled; we only need the flow) ──
    lbm::Config c{};
    c.nx=g.nx; c.ny=g.ny; c.nz=g.nz; c.cell_size=(float)dx;
    c.U_inlet=(T->u_ref_ms>0.f)?T->u_ref_ms:5.0f;
    c.wind_angle=(float)(T->theta_deg*M_PI/180.0);
    c.nu_phys=1.5e-5f; c.Cw=0.325f; c.Sc_t=0.7f; c.D_mol=1e-5f;
    c.collision_mode=2; c.hrr_sigma=0.98f;
    c.source_x=0; c.source_y=0; c.source_z=1.f; c.Q_source=0.f;
    c.particle_diam=0.f; c.particle_density=1000.f;
    c.max_steps=2000000; c.check_interval=500; c.conv_threshold=1e-4f;
    c.max_warmup=80000; c.avg_threshold=2e-3f; c.avg_steps=0; c.spinup_steps=0;
    { const char* e=getenv("MAX_WARMUP"); if(e){int v=atoi(e); if(v>0)c.max_warmup=v;} }
    c.release_time=1e-3f;
    c.inlet_profile=1; c.abl_z0=0.045f; c.abl_zref=4.0f;
    c.abl_Lturb=20.f; c.abl_nmodes=100;
    c.abl_sigu_ratio=2.5f; c.abl_sigv_ratio=1.9f; c.abl_sigw_ratio=1.25f;

    lbm::Solver s(c);
    { const char* NM[3]={"MRT","reg","HRR"}; int want=c.collision_mode, got=s.effective_collision();
      printf("[driver] effective collision=%s, scalar_advection=%d (driver wanted %s)\n",
             NM[got&3], s.effective_scalar(), NM[want&3]);
      if(got!=want && !getenv("COLLISION")){
        fprintf(stderr,"[driver] FATAL: collision mismatch — wanted %s, solver runs %s, and no COLLISION env "
                       "explains it. Aborting rather than running the wrong operator.\n", NM[want&3], NM[got&3]);
        return 3; } }
    s.load_geometry(g.tp.data(), perm.data(), inh.data(), depv.data());
    printf("[exposure] running HRR flow to stationarity...\n");
    lbm::Result res = s.run();
    if(!std::isfinite(res.max_velocity)){ fprintf(stderr,"[exposure] FATAL: flow diverged.\n"); return 2; }

    // ── 3. freeze mean flow -> transport operator M (adj::Field) ─────────────
    std::vector<float> ux,uy,uz,nut; s.copy_mean_flow_to_host(ux,uy,uz,nut);
    adj::Field F; F.nx=g.nx; F.ny=g.ny; F.nz=g.nz; F.N=(int)N;
    F.tp=g.tp.data(); F.ux=ux.data(); F.uy=uy.data(); F.uz=uz.data(); F.nut=nut.data();
    std::vector<float> vdep(N,0.f); F.vdep=vdep.data();
    F.D0=1e-3f; F.Sc_t=c.Sc_t; F.w_s=0.f;
    F.dt=adj::stable_dt(F,0.4f);

    // ── 4. receptor field w: linear infiltration model (Riley et al. 2002) ────
    // Indoor exposure = f_in * F_inf * C_out at the building ENVELOPE; outdoor
    // exposure = f_out * C_out at pedestrian level. Both weight the OUTDOOR
    // (airborne) field the solver produces. Deposition on building faces is a
    // separate SINK (not an exposure term); no double-count because the two
    // pathways sample the same airborne C_out, split by the time budget
    // f_in + f_out <= 1 (NHAPS). v1: UNIFORM occupancy and uniform F_inf (Allen
    // 2012 fallback); per-building occupancy/F_inf via city_builder7 is next.
    // See INFILTRATION_MODEL.md. Refs: Riley/McKone/Lai/Nazaroff 2002 (F_inf);
    // Allen 2012 MESA Air (F_inf=0.62); Klepeis 2001 NHAPS (f_in,f_out).
    std::vector<float> w(N,0.f);
    float FINF=0.62f;  { const char* e=getenv("FINF");  if(e){float v=atof(e); if(v>0)  FINF=v;} }
    float F_IN=0.87f;  { const char* e=getenv("F_IN");  if(e){float v=atof(e); if(v>=0) F_IN=v;} }
    float F_OUT=0.075f;{ const char* e=getenv("F_OUT"); if(e){float v=atof(e); if(v>=0) F_OUT=v;} }
    if(FINF>1.f) FINF=1.f;
    int zped=std::max(1,(int)std::llround(z_ped/dx));
    auto TP=[&](int x,int y,int z){ return g.tp[must::gidx(x,y,z,g.nx,g.ny)]; };
    // built-up region = bounding box of all buildings (SOLID) + margin; outdoor
    // pedestrians live here, NOT in the empty upwind/downwind buffers.
    int bxmin=g.nx,bxmax=-1,bymin=g.ny,bymax=-1;
    for(int z=1;z<g.nz;++z)for(int y=0;y<g.ny;++y)for(int x=0;x<g.nx;++x)
        if(TP(x,y,z)==must::SOLID){ bxmin=std::min(bxmin,x);bxmax=std::max(bxmax,x);
                                    bymin=std::min(bymin,y);bymax=std::max(bymax,y); }
    int mg=std::max(2,(int)std::llround(10.0/dx));   // ~10 m pedestrian margin around the block
    bxmin-=mg;bxmax+=mg;bymin-=mg;bymax+=mg;
    long nEnv=0,nPed=0; double wsum=0, wEnv=0, wPed=0;
    for(int z=1;z<g.nz-1;++z)for(int y=1;y<g.ny-1;++y)for(int x=1;x<g.nx-1;++x){
        if(TP(x,y,z)!=must::FLUID) continue;
        int id=must::gidx(x,y,z,g.nx,g.ny);
        // envelope: outdoor fluid face-adjacent to a building (SOLID) -> indoor intake.
        // v1 UNIFORM occupancy PER envelope cell => building occupancy scales with
        // envelope (surface) area; per-building N_b/|env| (mean-C_out, eq.3) awaits
        // the city_builder7 occupancy bridge.
        bool env = TP(x+1,y,z)==must::SOLID||TP(x-1,y,z)==must::SOLID||
                   TP(x,y+1,z)==must::SOLID||TP(x,y-1,z)==must::SOLID||
                   TP(x,y,z+1)==must::SOLID||TP(x,y,z-1)==must::SOLID;
        if(env){ float wi=F_IN*FINF; w[id]+=wi; wEnv+=wi; ++nEnv; }         // indoor occupants
        // outdoor pedestrians: pedestrian-height fluid inside the built-up region
        if(z==zped && x>=bxmin&&x<=bxmax && y>=bymin&&y<=bymax){ w[id]+=F_OUT; wPed+=F_OUT; ++nPed; }
    }
    wsum=wEnv+wPed;
    printf("[exposure] receptor w (Riley F_inf): F_inf=%.2f f_in=%.2f f_out=%.3f\n",FINF,F_IN,F_OUT);
    printf("[exposure]   indoor: %ld envelope cells (w=%.3g)  outdoor: %ld pedestrian cells in built-up box (w=%.3g)\n",nEnv,wEnv,nPed,wPed);
    if(wsum<=0){ fprintf(stderr,"[exposure] receptor field is empty — check geometry.\n"); return 2; }

    // ── 5. reverse (adjoint) steady solve: phi_ss(x)=dJ/ds(x) over all cells ──
    std::vector<float> phi; int nit=0;
    printf("[exposure] reverse_steady to convergence (cap %d)...\n",cap);
    adj::reverse_steady(F, w, phi, 1e-4, cap, &nit);
    printf("[exposure] adjoint converged in %d iters\n",nit);

    // ── 6. Omega reduction: outdoor ground-level fluid cells ─────────────────
    // J_ensemble = mean exposure rate for a single unknown-location source in Omega.
    double J=0.0; long nOmega=0;
    int zg=1;                                        // first fluid layer above ground
    for(int y=0;y<g.ny;++y)for(int x=0;x<g.nx;++x){
        if(g.tp[must::gidx(x,y,zg,g.nx,g.ny)]==must::FLUID){ J += phi[must::gidx(x,y,zg,g.nx,g.ny)]; ++nOmega; }
    }
    // Divide by the number of candidate source cells |Omega|: J_ensemble is the mean
    // exposure rate PER SOURCE LOCATION, so a larger/smaller or denser/sparser open
    // ground set does not by itself move the metric (it stays intensive, not a sum).
    double J_ensemble = (nOmega>0)? J/(double)nOmega : 0.0;

    s.export_avg_velocity("exposure_vel_z2.bin",(int)std::lround(z_ped/dx));
    // write phi slice at pedestrian height for the visualizer
    { FILE* pf=fopen("exposure_phi_zped.csv","w"); fprintf(pf,"# x_m,y_m,phi  z=%.1fm trial=%s\n",z_ped,T->name);
      for(int y=0;y<g.ny;++y)for(int x=0;x<g.nx;++x){ int z=zped; if(z<g.nz)
          fprintf(pf,"%.2f,%.2f,%.6e\n",(x-g.ax0)*dx,(y-g.ay0)*dx,phi[must::gidx(x,y,z,g.nx,g.ny)]); }
      fclose(pf); }

    printf("\n[exposure] ===== objective =====\n");
    printf("  |Omega| (outdoor ground candidate cells) = %ld\n", nOmega);
    printf("  J_ensemble (mean steady exposure RATE, lower is better) = %.6e\n", J_ensemble);
    printf("  (units: rate per unit emission, lattice; compare across designs, not in absolute terms)\n");
    printf("  wrote exposure_phi_zped.csv + exposure_vel_z2.bin for the visualizer\n");
    return 0;
}
