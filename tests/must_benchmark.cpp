// must_benchmark.cpp — MUST dispersion benchmark driver.
//
// Pipeline:  MUST container array (must_geom.h) -> LBM mean flow under HRR
// (lbm_solver) -> freeze mean velocity + eddy viscosity -> LINEAR QUICK scalar
// transport from the point source (adjoint_transport.h, the PRODUCTION scheme)
// -> sample the non-dimensional concentration K = C·U·H²/Q at the receptor
// towers -> write predicted.csv for must_score.
//
// The LBM run only supplies the converged mean flow (release disabled); the
// dispersion is done by the linear QUICK operator whose exact transpose is the
// adjoint/reverse solver. Refs: Yee & Biltoft (2004); Leonard (1979, QUICK);
// Jacob/Malaspinas/Sagaut (2018, HRR); Chang & Hanna (2004, scoring).
//
// Build (CPU/OpenMP):
//   g++ -O3 -std=c++17 -fopenmp -c lbm_kernels_cpu.cpp -o kc.o
//   g++ -O3 -std=c++17 -fopenmp -c lbm_solver.cpp -o so.o
//   g++ -O3 -std=c++17 -fopenmp must_benchmark.cpp kc.o so.o -o must_benchmark
//   (GPU: link lbm_kernels.cu built with nvcc instead of kc.o)
//
// Run:   ./must_benchmark [trial=2681829] [dx_m=0.5] [transport_steps=12000] [receptors.csv]
//   receptors.csv rows: id,x_m,y_m,z_m  (metres, relative to the array origin)
//   COLLISION/HRR_SIGMA are read from the env by the solver (default hrr/0.98).
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
    double dx      = (argc>2)? atof(argv[2]) : 0.5;
    int    tsteps  = (argc>3)? atoi(argv[3]) : 12000;
    const char* recf = (argc>4)? argv[4] : nullptr;

    // ── 1. pick the trial preset ────────────────────────────────────────────
    const must::MustTrial* T=nullptr;
    for(auto& t : must::MUST_TRIALS) if(trial==t.name){ T=&t; break; }
    if(!T){ fprintf(stderr,"unknown trial '%s' (have 2681829, 2681849, 2640246)\n",trial.c_str()); return 2; }
    printf("[MUST] trial %s  theta=%+.1f deg  u_ref=%.2f m/s  h_rel=%.2f m\n",
           T->name, T->theta_deg, T->u_ref_ms, T->release_h_m);
    if(T->u_ref_ms<=0.f) fprintf(stderr,"[MUST] WARNING: u_ref=0 (a [DATA] field) — set it from the trial file.\n");

    // ── 2. geometry ─────────────────────────────────────────────────────────
    must::MustSpec sp = must::spec_for(*T); sp.dx=dx;
    must::MustGeom g  = must::build(sp);
    must::report(g);
    size_t N=g.N;
    std::vector<float> perm(N,0.f), inh(N,0.f), depv(N,0.f);   // solid containers, passive gas

    // ── 3. LBM mean flow under HRR (release disabled: we only need the flow) ─
    lbm::Config c{};
    c.nx=g.nx; c.ny=g.ny; c.nz=g.nz; c.cell_size=(float)dx;
    c.U_inlet=(T->u_ref_ms>0.f)?T->u_ref_ms:5.0f;
    c.wind_angle=(float)(T->theta_deg*M_PI/180.0);            // oblique inflow, array axis-aligned
    c.nu_phys=1.5e-5f; c.Cw=0.325f; c.Sc_t=0.7f; c.D_mol=1e-5f;
    c.collision_mode=2; c.hrr_sigma=0.98f;                    // HRR (env can override)
    c.source_x=0; c.source_y=0; c.source_z=(float)T->release_h_m; c.Q_source=0.f; // no solver scalar
    c.particle_diam=0.f; c.particle_density=1000.f;
    c.max_steps=2000000; c.check_interval=500; c.conv_threshold=1e-4f;
    c.max_warmup=80000; c.avg_threshold=2e-3f; c.avg_steps=0; c.spinup_steps=0;
    { const char* e=getenv("MAX_WARMUP"); if(e){ int v=atoi(e); if(v>0) c.max_warmup=v; } }
    { const char* e=getenv("TSTEPS");     if(e){ int v=atoi(e); if(v>0) tsteps=v; } }
    c.release_time=1e-3f;                                      // ~0 Phase-B steps (we use QUICK transport, not the solver scalar)
    c.inlet_profile=1; c.abl_z0=0.045f; c.abl_zref=4.0f;      // MUST z0=0.045 m, ref at 4 m
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
    printf("[MUST] running HRR flow to stationarity (this is the expensive step)...\n");
    lbm::Result res = s.run();
    if(!std::isfinite(res.max_velocity)){ fprintf(stderr,"[MUST] FATAL: flow diverged.\n"); return 2; }

    std::vector<float> ux,uy,uz,nut;
    s.copy_mean_flow_to_host(ux,uy,uz,nut);
    s.export_avg_velocity("must_vel_z2.bin", (int)std::lround(1.6/dx));  // for the visualizer
    float U_lb = c.U_inlet * s.velocity_phys_to_lattice();
    int   Hc   = std::max(1,(int)std::llround(sp.cH/dx));

    // ── 4. linear QUICK transport on the frozen mean flow ───────────────────
    adj::Field F; F.nx=g.nx; F.ny=g.ny; F.nz=g.nz; F.N=(int)N;
    F.tp=g.tp.data(); F.ux=ux.data(); F.uy=uy.data(); F.uz=uz.data(); F.nut=nut.data();
    std::vector<float> vdep(N,0.f); F.vdep=vdep.data();
    F.D0=1e-3f; F.Sc_t=c.Sc_t; F.w_s=0.f;                     // passive gas: no deposition/settling
    F.dt=adj::stable_dt(F,0.4f);

    // point source (unit strength => K independent of Q by linearity)
    int sx=g.ax0+(int)std::llround(sp.src_x/dx), sy=g.ay0+(int)std::llround(sp.src_y/dx),
        sz=std::max(1,(int)std::llround(T->release_h_m/dx));
    if(sx<1||sx>=g.nx-1||sy<1||sy>=g.ny-1||sz<1||sz>=g.nz-1){ fprintf(stderr,"[MUST] source cell out of range\n"); return 2; }
    std::vector<float> src(N,0.f), wprobe(N,0.f), Css;
    src[F.idx(sx,sy,sz)]=1.0f;
    for(size_t i=0;i<N;++i) if(g.tp[i]==must::FLUID) wprobe[i]=1.0f;   // total-airborne steady-state probe
    printf("[MUST] QUICK transport to steady state (cap %d steps), dt=%.3f, source cell (%d,%d,%d)\n",tsteps,F.dt,sx,sy,sz);
    int nconv=0;
    adj::forward_steady(F, src, wprobe, 1e-4, tsteps, &Css, &nconv);   // steady concentration field C_ss
    printf("[MUST] steady field reached in %d steps\n", nconv);
    // non-dimensional K = C_ss·U·H²/Q  (Q=1, lattice units => dimensionless), from the STEADY field
    std::vector<double> K(N);
    for(size_t i=0;i<N;++i) K[i] = (double)Css[i] * U_lb * (double)Hc*Hc;

    // ── 5. sample receptors -> predicted.csv ────────────────────────────────
    FILE* out=fopen("must_predicted.csv","w"); fprintf(out,"# id,K_nondim  trial=%s dx=%.2f\n",T->name,dx);
    int nrec=0;
    auto sample=[&](const char* id,double xm,double ym,double zm){
        int x=g.ax0+(int)std::llround(xm/dx), y=g.ay0+(int)std::llround(ym/dx), z=(int)std::llround(zm/dx);
        if(x<0||x>=g.nx||y<0||y>=g.ny||z<0||z>=g.nz){ fprintf(out,"%s,nan\n",id); return; }
        fprintf(out,"%s,%.6e\n", id, K[F.idx(x,y,z)]); ++nrec;
    };
    if(recf){
        FILE* rf=fopen(recf,"r"); if(!rf){ fprintf(stderr,"[MUST] cannot open receptors %s\n",recf); return 2; }
        char line[256];
        while(fgets(line,sizeof line,rf)){ if(line[0]=='#'||line[0]=='\n') continue;
            char id[64]; double x,y,z; if(sscanf(line,"%63[^,],%lf,%lf,%lf",id,&x,&y,&z)==4) sample(id,x,y,z); }
        fclose(rf);
    } else {
        // no receptor file: default downwind centreline sampling at z=1.6 m so the
        // pipeline runs end-to-end (replace with the trial's tower coordinates).
        fprintf(stderr,"[MUST] no receptors file: sampling a default downwind line (z=1.6 m).\n");
        double arrEnd = (g.ax0 + ( (sp.cols-1)*std::max(1,(int)std::llround(sp.pitchX/dx)) ))*dx;
        for(int k=0;k<12;++k){ char id[32]; snprintf(id,sizeof id,"L%02d",k);
            sample(id, arrEnd + 10*k, 0.0, 1.6); }
    }
    fclose(out);
    printf("[MUST] wrote must_predicted.csv (%d receptors) + must_vel_z2.bin\n",nrec);
    printf("[MUST] score against measured data:  ./must_score observed.csv must_predicted.csv\n");
    return 0;
}
