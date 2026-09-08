// ranking_stability.cpp — does the cheap reverse-solver objective rank city designs
// the same as the production "truth"?
//
// CORRECTION (after inspecting the solver): the two objectives below differ in TWO
// ways, not one. The production scalar advects on the LIVE, resolved-turbulent LBM
// flow (lbm_solver.cpp line 359: "advection-diffusion on the live flow"), whereas
// the linear-upwind objective runs on the FROZEN TIME-MEAN flow (what the reverse
// solver uses). So this compares the reverse-solver objective against the production
// truth — conflating the numerical scheme (van-Leer vs upwind, the ~49× diffusion
// penalty from diffusion_compare) WITH the flow (mean vs resolved-turbulent). It is
// the right END-TO-END test for "can the reverse objective be trusted to rank
// designs," but it does NOT isolate numerical diffusion alone; a shift could be
// either cause. Isolating the scheme needs both schemes on the SAME frozen mean flow
// (a van-Leer forward on the mean) — a separate test, not built here.
//
// For each design d:
//   • build the city, run the LBM flow once (passive tracer, point source);
//   • J_vanleer(d) = Σ_x w(x)·TIAC_vanleer(x)   — production scalar, LIVE flow;
//   • capture the frozen time-mean flow; run the linear-upwind forward from the SAME
//     point source on it; J_upwind(d) = Σ_x w(x)·TIAC_upwind(x).
//   Same receptor w (effective inhabitance), same source cell; flow & scheme differ.
//
// Then report Spearman ρ, Kendall τ, Pearson(log J), and top-k overlap of the
// BEST (lowest-J) designs. If the rankings agree, the reverse objective is usable
// for optimization; if they disagree, isolate the cause (scheme vs flow) before
// trusting it — e.g. the frozen-limiter adjoint (Option 2) for the scheme part.
//
// CAVEAT (stated honestly): this uses a POINT source, because the production
// van-Leer cannot cheaply produce the uniform-open-space-source objective the
// optimizer actually uses (van-Leer is nonlinear, so it does not superpose). A
// point source is a CONSERVATIVE proxy — the uniform source is a spatial average
// over release points and is expected to rank at least as stably. If the
// point-source rankings are stable, the uniform-source rankings almost certainly
// are too; if they are not, escalate to the uniform-source test.
//
// Modes:
//   selftest        : verify the statistics on synthetic data (no LBM; fast).
//   run [N] [m] [s] [w] : full comparison over N designs (needs WITH_LBM build).

#include <set>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>

// ── rank statistics ──────────────────────────────────────────────────────────
static void average_ranks(const std::vector<double>& x, std::vector<double>& rk){
    int n=x.size(); std::vector<int> idx(n); for(int i=0;i<n;++i) idx[i]=i;
    std::sort(idx.begin(),idx.end(),[&](int a,int b){return x[a]<x[b];});
    rk.assign(n,0);
    int i=0;
    while(i<n){ int j=i; while(j+1<n && x[idx[j+1]]==x[idx[i]]) ++j;
        double r=0.5*((i+1)+(j+1)); for(int k=i;k<=j;++k) rk[idx[k]]=r; i=j+1; }
}
static double pearson(const std::vector<double>& a,const std::vector<double>& b){
    int n=a.size(); double ma=0,mb=0; for(int i=0;i<n;++i){ma+=a[i];mb+=b[i];} ma/=n;mb/=n;
    double sab=0,sa=0,sb=0;
    for(int i=0;i<n;++i){ double da=a[i]-ma,db=b[i]-mb; sab+=da*db; sa+=da*da; sb+=db*db; }
    return (sa>0&&sb>0)? sab/std::sqrt(sa*sb) : 0.0;
}
static double spearman(const std::vector<double>& a,const std::vector<double>& b){
    std::vector<double> ra,rb; average_ranks(a,ra); average_ranks(b,rb); return pearson(ra,rb);
}
static double kendall(const std::vector<double>& a,const std::vector<double>& b){
    int n=a.size(); long con=0,dis=0;
    for(int i=0;i<n;++i)for(int j=i+1;j<n;++j){
        double da=a[i]-a[j], db=b[i]-b[j];
        if(da*db>0) ++con; else if(da*db<0) ++dis;   // ties ignored
    }
    long tot=con+dis; return tot? (double)(con-dis)/tot : 0.0;
}
// fraction of the best-k (lowest J) designs that both rankings agree on
static double topk_overlap(const std::vector<double>& a,const std::vector<double>& b,int k){
    int n=a.size(); k=std::min(k,n);
    auto bestk=[&](const std::vector<double>& x){
        std::vector<int> idx(n); for(int i=0;i<n;++i)idx[i]=i;
        std::partial_sort(idx.begin(),idx.begin()+k,idx.end(),[&](int p,int q){return x[p]<x[q];});
        return std::vector<int>(idx.begin(),idx.begin()+k); };
    auto A=bestk(a), B=bestk(b); int hit=0;
    for(int x:A) if(std::find(B.begin(),B.end(),x)!=B.end()) ++hit;
    return (double)hit/k;
}

static void report(const std::vector<double>& Jvl,const std::vector<double>& Jup){
    int n=Jvl.size();
    std::vector<double> lvl(n),lup(n);
    for(int i=0;i<n;++i){ lvl[i]=std::log(std::max(Jvl[i],1e-300)); lup[i]=std::log(std::max(Jup[i],1e-300)); }
    double rho=spearman(Jvl,Jup), tau=kendall(Jvl,Jup), rp=pearson(lvl,lup);
    int k=std::max(1,n/3); double ov=topk_overlap(Jvl,Jup,k);
    std::vector<double> rvl,rup; average_ranks(Jvl,rvl); average_ranks(Jup,rup);
    printf("\n design   J_vanleer      J_upwind     rank_vl  rank_up\n");
    for(int i=0;i<n;++i)
        printf("  %3d   %.4e   %.4e    %5.1f   %5.1f\n",i,Jvl[i],Jup[i],rvl[i],rup[i]);
    printf("\n  Spearman rho      = %+.3f   (rank correlation; 1 = identical order)\n",rho);
    printf("  Kendall  tau      = %+.3f\n",tau);
    printf("  Pearson(log J)    = %+.3f\n",rp);
    printf("  top-%d overlap     = %.0f%%   (best designs both schemes agree on)\n",k,100*ov);
    bool stable = (rho>0.9 && ov>0.99);
    printf("\n  VERDICT: %s\n", stable
        ? "RANKINGS STABLE — the cheap reverse objective ranks designs like the\n"
          "           production truth; usable for optimization. (Does NOT isolate cause;\n"
          "           agreement holds despite both scheme and mean-vs-turbulent-flow differences.)"
        : "RANKINGS SHIFT — the reverse objective and production truth disagree on\n"
          "           design order. Isolate the cause (numerical scheme vs mean-vs-turbulent\n"
          "           flow) with a same-flow test before trusting the reverse objective.");
}

static int selftest(){
    // synthetic: an underlying 'truth' J, plus a diffusion-perturbed version that
    // preserves order (should be STABLE) and a noisy version that scrambles it.
    std::vector<double> truth={1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0};
    std::vector<double> mono; for(double v:truth) mono.push_back(v*1.3+0.05*v*v); // monotone
    std::vector<double> noisy={3.0,1.0,2.0,8.0,4.0,7.0,5.0,6.0};
    printf("=== selftest: statistics sanity ===\n");
    printf("monotone perturbation (expect rho=+1, overlap=100%%):\n");
    report(truth,mono);
    printf("\nscrambled (expect low rho/overlap):\n");
    report(truth,noisy);
    // exact checks
    double r1=spearman(truth,mono); double k1=kendall(truth,truth);
    printf("\n  checks: spearman(monotone)=%.3f (want 1.000), kendall(identical)=%.3f (want 1.000)\n",r1,k1);
    return (std::fabs(r1-1.0)<1e-9 && std::fabs(k1-1.0)<1e-9)?0:1;
}

#ifdef WITH_LBM
#include "adjoint_transport.h"
#include "reverse_objective.h"
#include "city_builder7.h"
#include "voxelize.h"
#include "lbm_solver.h"
#include "occupancy.h"
using namespace city;

struct Design { double street_width, block_w, block_d, roughness, cbd_peak, park_centrality; };

// a small, deterministic, diverse set spanning the ventilation-relevant axes.
// Column 1 is street width (m): with the base heights it spans the Oke (1988)
// canyon regimes (skimming H/W>~0.65 → isolated-roughness H/W<~0.3).
static std::vector<Design> make_designs(int N){
    std::vector<Design> base = {
        {10, 36, 20, 0.10,  20, 0.2},
        {16, 60, 32, 0.60, 100, 0.8},
        {24, 48, 24, 0.30,  40, 0.5},
        {32, 72, 40, 0.80,  60, 0.1},
        {12, 28, 18, 0.20,  30, 0.9},
        {20, 52, 28, 0.50,  90, 0.4},
        {28, 40, 22, 0.40,  50, 0.6},
        {40, 64, 36, 0.70,  70, 0.3},
    };
    if(N<(int)base.size()) base.resize(N);
    return base;
}

static Params design_params(const Design& d, double city_m){
    Params p{};
    p.city_w=city_m; p.city_h=city_m;
    p.block_w=d.block_w; p.block_d=d.block_d;   // base_height is fixed/ignored by the builder
    p.cbd_peak=d.cbd_peak; p.cbd_decay=8e-6; p.cbd_aspect=1; p.cbd_angle=0;
    p.biz_inner_frac=0.05; p.biz_aspect=1; p.park_centrality=d.park_centrality; p.park_fraction=0.10;
    p.roughness=d.roughness;   // roughness is now a design axis (was the dead base_height)
    p.road_w_x=d.street_width; p.road_w_y=d.street_width;   // isotropic streets per design
    p.population_total=20000;
    p.wind_direction=0;
    default_buffers(p); compute_domain(p);
    p.buf_xn=(p.Sx-p.city_w)*0.5; p.buf_yn=(p.Sy-p.city_h)*0.5;
    p.source_x=p.buf_xn-100.0; p.source_y=p.buf_yn+p.city_h*0.5;
    ensure_source_buffer(p);
    double cx=p.buf_xn+p.city_w*0.5, cy=p.buf_yn+p.city_h*0.5;
    p.cbd_x=cx; p.cbd_y=cy; p.biz_center_x=cx; p.biz_center_y=cy;
    return p;
}

static int run_full(int argc,char**argv){
    int N      =(argc>2)?atoi(argv[2]):6;
    double cm  =(argc>3)?atof(argv[3]):512.0;
    double rel =(argc>4)?atof(argv[4]):60.0;
    int    warm=(argc>5)?atoi(argv[5]):4000;
    auto designs=make_designs(N); N=designs.size();
    // v8: size ONE fixed minimal directional domain for the whole sweep, for the
    // tallest design — every design shares an identical grid (never compare
    // different-sized domains). COST-732 / Tominaga et al. (2008): 5H inflow
    // (−x,−y), 15H outflow (+x,+y), 5H top headroom; power-of-2 NOT enforced.
    // (Asymmetric ⇒ the city sits offset toward the inflow corner, extra room
    // downstream — the intended CFD layout, not centred.)
    double Hmax=0; for(auto&d:designs){ Params pp=design_params(d,cm); Hmax=std::max(Hmax,generate(pp).max_height); }
    int NZ_FIX=nz_cost732(Hmax);
    printf("[rank] fixed minimal domain: Hmax=%.0fm  buffers 5H/15H (in/out)  nz=%d  (no pow2)\n", Hmax, NZ_FIX);
    std::vector<double> Jvl(N,0.0), Jup(N,0.0);
    // ── Resume support: if ranking_stability.csv already holds finished designs,
    // append and skip them, so a crash / re-run doesn't redo completed solves.
    std::set<int> done;
    { FILE* rd=fopen("ranking_stability.csv","r"); if(rd){ char line[512];
        bool first=true; while(fgets(line,sizeof line,rd)){ if(first){first=false;continue;}
            int di; if(sscanf(line,"%d,",&di)==1) done.insert(di); } fclose(rd); } }
    bool resume = !done.empty();
    FILE* csv=fopen("ranking_stability.csv", resume?"a":"w");
    if(!resume) fprintf(csv,"design,street_width,block_w,roughness,cbd_peak,park_centrality,J_vanleer,J_upwind\n");
    if(resume) printf("[rank] RESUME: %zu design(s) already done, skipping them\n", done.size());

    for(int d=0; d<N; ++d){
        if(done.count(d)){ printf("[rank] design %d/%d already done — skipped\n",d+1,N); continue; }
        Params p=design_params(designs[d], cm);
        static const double IN_H = getenv("INFLOW_H") ? atof(getenv("INFLOW_H")) : 5.0;   // INFLOW_H env
        static const double OUT_H= getenv("OUTFLOW_H")? atof(getenv("OUTFLOW_H")):15.0;   // OUTFLOW_H env (trim for memory)
        minimal_domain(p, Hmax, IN_H, OUT_H);          // COST-732 default 5H inflow / 15H outflow (+x)
        p.source_x=p.buf_xn-100.0;                     // 100 m upstream of city, inside the inflow buffer
        p.source_y=p.buf_yn+p.city_h*0.5;              // on the city centreline (y)
        Result r=generate(p);
        int NX=r.nx_cells,NY=r.ny_cells;
        VoxelGrid g=voxelize(p,r,true,0.0,1800.0,0.5,NZ_FIX);   // dep_dp=0 → passive tracer; nz fixed across sweep
        int NZ=g.nz, N3=NX*NY*NZ;
        lbm::Config c{}; c.nx=g.nx;c.ny=g.ny;c.nz=g.nz; c.cell_size=(float)CELL;
        c.U_inlet=5.f;c.wind_angle=0.f;c.nu_phys=1.5e-5f;c.Cw=0.325f;c.Sc_t=0.7f;c.D_mol=1e-5f;
        c.source_x=p.source_x;c.source_y=p.source_y;c.source_z=4.f;c.Q_source=1.f;
        c.particle_diam=0.f;c.particle_density=1800.f;          // passive tracer (no settling)
        c.max_steps=2000000;c.check_interval=500;c.conv_threshold=1e-4f;
        c.max_warmup=warm;c.avg_threshold=0.f;c.release_time=(float)rel;   /* 0 ⇒ solver default stationarity tol (5e-3); the old 2e-3 was the retired RMS-warmup value */
        c.inlet_profile=1;c.abl_z0=0.7f;c.abl_zref=40.f;c.abl_Lturb=40.f;c.abl_nmodes=100;
        c.abl_sigu_ratio=2.5f;c.abl_sigv_ratio=1.9f;c.abl_sigw_ratio=1.25f;

        lbm::Solver s(c);
        s.load_geometry(g.type.data(),g.perm.data(),g.inh.data(),g.dep_vel.data());
        printf("\n[rank] design %d/%d  sw=%.0fm rn=%.2f cbd=%.0f  (NX=%d NY=%d NZ=%d)\n",
               d+1,N,designs[d].street_width,designs[d].roughness,designs[d].cbd_peak,NX,NY,NZ);
        auto res=s.run();

        // receptor field w (effective inhabitance), z=1 — same for both schemes
        double street_pop=street_fraction(nhaps_average(),OUTDOOR_PARK_SHARE)*r.population;
        auto sf=build_street_occupancy(p,r,g,street_pop);
        std::vector<float> w; adj::build_receptor_field(r,g.type.data(),NX,NY,NZ,sf,w);

        // J_vanleer = Σ w·TIAC_vanleer  (production scalar; z=1 slice)
        s.export_tiac("rank_tiac_vl.bin",1);
        std::vector<float> tvl(NX*NY);
        { FILE* f=fopen("rank_tiac_vl.bin","rb"); int h[2]; size_t rd=fread(h,4,2,f);(void)rd;
          rd=fread(tvl.data(),4,NX*NY,f); fclose(f); }
        double jvl=0.0; for(int yx=0;yx<NX*NY;++yx) jvl += (double)w[(size_t)1*NY*NX+yx]*tvl[yx];

        // J_upwind = Σ w·TIAC_upwind  (linear-upwind forward, SAME point source)
        std::vector<float> ux,uy,uz,nut; s.copy_mean_flow_to_host(ux,uy,uz,nut);
        std::vector<float> vdep(N3,0.f);
        adj::Field F{NX,NY,NZ,N3,g.type.data(),ux.data(),uy.data(),uz.data(),nut.data(),
                     vdep.data(),1e-4f,0.7f,0.f,0.f}; F.dt=adj::stable_dt(F);  // w_s=0 passive
        std::vector<float> sp(N3,0.f);
        int sx=(int)std::lround(p.source_x/CELL), sy=NY/2, sz=1;
        if(sx<0)sx=0; if(sx>=NX)sx=NX-1;
        sp[(size_t)(sz*NY+sy)*NX+sx]=1.f;
        float umax=1e-3f; for(int i=0;i<N3;++i){ float spd=std::fabs(ux[i])+std::fabs(uy[i])+std::fabs(uz[i]);
            if(g.type[i]==adj::FLUID && spd>umax) umax=spd; }
        int nstep=(int)std::min(4000.0,std::max(400.0,3.0*NX/umax/F.dt));
        double jup = adj::forward_objective(F, sp, w, nstep);

        Jvl[d]=jvl; Jup[d]=jup;
        printf("[rank]   J_vanleer=%.4e  J_upwind=%.4e\n", jvl, jup);
        fprintf(csv,"%d,%.3f,%.1f,%.2f,%.1f,%.2f,%.6e,%.6e\n",d,designs[d].street_width,
                designs[d].block_w,designs[d].roughness,designs[d].cbd_peak,
                designs[d].park_centrality,jvl,jup);
        fflush(csv);
    }
    fclose(csv);
    report(Jvl,Jup);
    printf("\n[rank] wrote ranking_stability.csv\n");
    return 0;
}
#endif

int main(int argc,char**argv){
    std::string mode=(argc>1)?argv[1]:"selftest";
    if(mode=="selftest") return selftest();
#ifdef WITH_LBM
    if(mode=="run") return run_full(argc,argv);
#endif
    fprintf(stderr,"usage: %s selftest                  (verify statistics; fast)\n",argv[0]);
    fprintf(stderr,"       %s run [N] [city_m] [rel_s] [warmup]   (needs WITH_LBM build)\n",argv[0]);
    return 1;
}
