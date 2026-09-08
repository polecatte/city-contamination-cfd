// forward_live.cpp — accidental-release forward dose, live turbulent flow.
//
// Regime (FORWARD_LIVE.md): a BURST source (impulse at t=0) is transported on the
// LIVE flow with the linear QUICK scalar scheme; Phase B runs to 99% clearance
// (mass deposits/advects out), so the cumulative dose Θ=∫C dt is self-terminating.
// Because one live burst is a single turbulent realisation, we run an ENSEMBLE of
// N bursts, each with a different RFG turbulence seed, and report the population
// dose J=⟨w,Θ⟩ as mean ± spread. w is the F_inf-weighted inhabitance (indoor via
// envelope, outdoor via pedestrians), reused from the exposure model.
//
// Build:  g++ -O3 -std=c++17 -fopenmp -c lbm_kernels_cpu.cpp -o kc.o
//         g++ -O3 -std=c++17 -fopenmp -c lbm_solver.cpp -o so.o
//         g++ -O3 -std=c++17 -fopenmp forward_live.cpp kc.o so.o -o forward_live
// Run:    COLL=hrr SCALAR=quick ./forward_live [trial=2681829] [dx=2.0] [N=8]
//
// NOTE: this drives a SINGLE-cell burst per member (the verified mechanism). A
// simultaneous source DISTRIBUTION over Ω is the sum of per-cell bursts by
// linearity (QUICK is linear — verified); doing it in one run needs the per-cell
// source MASK (the one remaining kernel addition, see FORWARD_LIVE.md).
#include "lbm_solver.h"
#include "must_geom.h"
#include <cstdio>
#include <vector>
#include <cmath>
#include <string>
#include <algorithm>

int main(int argc,char**argv){
    std::string trial = (argc>1)? argv[1] : "2681829";
    double dx  = (argc>2)? atof(argv[2]) : 2.0;
    int    N   = (argc>3)? atoi(argv[3]) : 8;     // ensemble members
    double z_ped = 2.0;

    const must::MustTrial* T=nullptr;
    for(auto& t: must::MUST_TRIALS) if(trial==t.name){ T=&t; break; }
    if(!T){ fprintf(stderr,"unknown trial %s\n",trial.c_str()); return 2; }
    must::MustSpec sp = must::spec_for(*T); sp.dx=dx;
    must::MustGeom g  = must::build(sp); must::report(g);
    size_t Ncell=g.N;
    std::vector<float> perm(Ncell,0.f), inh(Ncell,0.f), depv(Ncell,0.f);

    // receptor weight w (F_inf infiltration model; see INFILTRATION_MODEL.md)
    float FINF=0.62f, F_IN=0.87f, F_OUT=0.075f;
    { const char* e; if((e=getenv("FINF")))FINF=atof(e); if((e=getenv("F_IN")))F_IN=atof(e); if((e=getenv("F_OUT")))F_OUT=atof(e); }
    int zped=std::max(1,(int)std::llround(z_ped/dx));
    auto TP=[&](int x,int y,int z){ return g.tp[must::gidx(x,y,z,g.nx,g.ny)]; };
    int bxmin=g.nx,bxmax=-1,bymin=g.ny,bymax=-1;
    for(int z=1;z<g.nz;++z)for(int y=0;y<g.ny;++y)for(int x=0;x<g.nx;++x)
        if(TP(x,y,z)==must::SOLID){ bxmin=std::min(bxmin,x);bxmax=std::max(bxmax,x);bymin=std::min(bymin,y);bymax=std::max(bymax,y); }
    int mg=std::max(2,(int)std::llround(10.0/dx)); bxmin-=mg;bxmax+=mg;bymin-=mg;bymax+=mg;
    std::vector<float> w(Ncell,0.f); double wsum=0;
    for(int z=1;z<g.nz-1;++z)for(int y=1;y<g.ny-1;++y)for(int x=1;x<g.nx-1;++x){
        if(TP(x,y,z)!=must::FLUID) continue; int id=must::gidx(x,y,z,g.nx,g.ny);
        bool env = TP(x+1,y,z)==must::SOLID||TP(x-1,y,z)==must::SOLID||TP(x,y+1,z)==must::SOLID||
                   TP(x,y-1,z)==must::SOLID||TP(x,y,z+1)==must::SOLID||TP(x,y,z-1)==must::SOLID;
        if(env){ w[id]+=F_IN*FINF; wsum+=F_IN*FINF; }
        if(z==zped && x>=bxmin&&x<=bxmax && y>=bymin&&y<=bymax){ w[id]+=F_OUT; wsum+=F_OUT; }
    }
    if(wsum<=0){ fprintf(stderr,"empty receptor field\n"); return 2; }

    // Number of SOURCE cells that make up one release. This driver emits a single-
    // cell burst (below), so n_source=1. When a source DISTRIBUTION over Omega is
    // used instead — the sum of per-cell bursts by linearity (see the file header) —
    // set this to |Omega|. The reported dose is divided by n_source so it is the
    // mean dose PER SOURCE LOCATION: adding or removing candidate release cells does
    // not move the metric. Matches exposure_solve.cpp (J/|Omega|).
    long n_source = 1;
    printf("[forward_live] receptor w assembled (sum=%.3g); ensemble N=%d; n_source=%ld\n",wsum,N,n_source);

    // ── ensemble of bursts, each a different turbulent realisation ───────────
    std::vector<double> J(N,0.0);
    std::vector<float> Theta_sum(Ncell,0.f), tiac;
    for(int k=0;k<N;++k){
        lbm::Config c{};
        c.nx=g.nx; c.ny=g.ny; c.nz=g.nz; c.cell_size=(float)dx;
        c.U_inlet=(T->u_ref_ms>0.f)?T->u_ref_ms:5.0f; c.wind_angle=(float)(T->theta_deg*M_PI/180.0);
        c.nu_phys=1.5e-5f; c.Cw=0.325f; c.Sc_t=0.7f; c.D_mol=1e-5f;
        c.collision_mode=2; c.hrr_sigma=0.98f; c.scalar_advection=3;      // HRR + QUICK
        c.source_x=0; c.source_y=0; c.source_z=(float)T->release_h_m;
        c.Q_source=1.0f;                                                  // burst TOTAL MASS (reused field)
        c.burst_release=true; c.clearance_frac=0.01f;                     // impulse, 99% clearance
        c.abl_seed=1000u + 137u*(unsigned)k;                             // decorrelated turbulence per member
        c.particle_diam=0.f; c.particle_density=1000.f;
        c.max_steps=300000; c.check_interval=500; c.conv_threshold=1e-4f;
        c.max_warmup=60000; c.avg_threshold=2e-3f; c.avg_steps=0; c.spinup_steps=0;
        c.spinup_ft=3;      // 3 flow-throughs of spin-up before release (production; justify per-case)
        c.release_time=0.f; c.inlet_profile=1; c.abl_z0=0.045f; c.abl_zref=4.0f;
        c.abl_Lturb=20.f; c.abl_nmodes=100; c.abl_sigu_ratio=2.5f; c.abl_sigv_ratio=1.9f; c.abl_sigw_ratio=1.25f;

        printf("[forward_live] === member %d/%d (seed=%u) ===\n",k+1,N,c.abl_seed);
        lbm::Solver s(c);
        if(k==0){ const char* NM[3]={"MRT","reg","HRR"}; int want=c.collision_mode, got=s.effective_collision();
          printf("[forward_live] effective collision=%s, scalar_advection=%d (wanted %s+QUICK)\n",
                 NM[got&3], s.effective_scalar(), NM[want&3]);
          if((got!=want || s.effective_scalar()!=3) && !getenv("COLLISION") && !getenv("SCALAR")){
            fprintf(stderr,"[forward_live] FATAL: operator mismatch (wanted %s+QUICK, got %s+scalar%d) — aborting\n",
                    NM[want&3], NM[got&3], s.effective_scalar()); return 3; } }
        s.load_geometry(g.tp.data(), perm.data(), inh.data(), depv.data());
        lbm::Result r = s.run();
        if(!std::isfinite(r.max_velocity)){ fprintf(stderr,"[forward_live] member %d diverged\n",k); return 2; }
        s.copy_tiac_to_host(tiac);                                        // Θ = ∫C dt (full field)
        double Jk=0.0; for(size_t i=0;i<Ncell;++i){ Jk += (double)w[i]*tiac[i]; Theta_sum[i]+=tiac[i]; }
        Jk /= (double)std::max(1L, n_source);                             // dose per source cell (invariant to # release sites)
        J[k]=Jk;
        printf("[forward_live]   member dose J=%.6e (deposited %.1f%%, airborne %.1f%%)\n",
               Jk, 100.0*r.total_deposited/std::max(1e-30,r.total_emitted),
               100.0*r.total_airborne /std::max(1e-30,r.total_emitted));
    }
    double mean=0; for(double v:J) mean+=v; mean/=N;
    double var=0;  for(double v:J) var+=(v-mean)*(v-mean); double sd=(N>1)?std::sqrt(var/(N-1)):0.0;

    printf("\n[forward_live] ===== accidental-release dose (N=%d bursts) =====\n",N);
    printf("  mean J = %.6e\n", mean);
    printf("  std  J = %.6e  (%.1f%% of mean — turbulent scatter across release realisations)\n",
           sd, 100.0*sd/std::max(1e-30,mean));
    printf("  min/max J = %.6e / %.6e\n",
           *std::min_element(J.begin(),J.end()), *std::max_element(J.begin(),J.end()));
    printf("  (dose is per unit released mass AND per source cell (n_source=%ld), lattice units; compare across designs)\n", n_source);
    return 0;
}
