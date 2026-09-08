// robustness_overnight.cpp — intensive overnight diagnostic attacking the
// program's main UNRESOLVED uncertainties in one long run.
//
// It scores N diverse city designs under a 2×2 matrix of conditions and asks
// whether the design RANKING (what the optimizer actually relies on) survives:
//
//   axis A — production truth vs reverse-solver objective:
//            J_vanleer = van-Leer transport on the LIVE (resolved-turbulent) LBM
//                        flow — the production "truth".
//            J_upwind  = linear-upwind transport on the FROZEN TIME-MEAN flow —
//                        exactly what the reverse solver optimizes against.
//            NOTE: these differ in TWO ways at once — the numerical scheme
//            (van-Leer vs upwind, the ~49× diffusion penalty from diffusion_compare)
//            AND the flow (resolved-turbulent vs time-mean). So axis A tests the
//            END-TO-END question "does the cheap reverse objective rank designs like
//            the expensive production truth?" — which is what matters for using the
//            reverse solver. It does NOT isolate numerical diffusion alone; a shift
//            could come from either factor. To isolate the scheme, run both schemes
//            on the SAME frozen mean flow (a van-Leer forward on the mean) — a
//            separate test, not built here.
//   axis B — Reynolds floor:    tau=0.530 (NU_FLOOR=0.010)  vs  tau=0.515 (0.005)
//            (the Re-dependence the re_independence test flagged, FAC2=0.194).
//
// Simultaneously it performs two integrity audits on real geometry:
//   • reverse-solver reciprocity on REAL city flows: forward Σw·C vs reverse ΣF·s.
//     adj_step is the exact transpose of fwd_step by construction, so this is a
//     wiring / NaN / dimension regression check, reported as reciprocity_rel.
//   • conservation sanity + finiteness across many diverse geometries and both
//     floors (catches blowups / gross mass creation; see mass_sane below).
//
// Cost is 2N LBM flow solves (the two scalar evaluations per flow are cheap), so
// it scales to fill a night: defaults to N=48, 768 m, warmup 10000, release 120 s.
// Progress is written incrementally to robustness_overnight.csv (fflush per row),
// so a crash or timeout still leaves usable partial results.
//
// Modes:
//   selftest                          : verify the statistics (no LBM; fast).
//   run [N] [city_m] [rel_s] [warmup] : the full overnight diagnostic (WITH_LBM).

#include <set>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include "rank_stats.h"
using namespace rstats;

// ── reporting ────────────────────────────────────────────────────────────────
[[maybe_unused]] static void rank_line(const char* name,const std::vector<double>& J){
    std::vector<double> rk; average_ranks(J,rk);
    printf("  %-22s ranks:", name);
    for(double r:rk) printf(" %4.1f",r);
    printf("\n");
}
static void compare(const char* what,const std::vector<double>& a,const std::vector<double>& b,int k){
    printf("  %-34s rho=%+.3f  tau=%+.3f  top%d=%3.0f%%\n",
           what, spearman(a,b), kendall(a,b), k, 100*topk_overlap(a,b,k));
}

static int selftest(){
    std::vector<double> t={1,2,3,4,5,6,7,8};
    std::vector<double> mono; for(double v:t) mono.push_back(v*1.3+0.05*v*v);
    std::vector<double> scr={3,1,2,8,4,7,5,6};
    printf("=== selftest ===\n");
    printf("monotone (expect rho=+1.000, top=100%%):\n");
    compare("monotone vs truth",t,mono,3);
    printf("scrambled (expect lower):\n");
    compare("scrambled vs truth",t,scr,3);
    double r1=spearman(t,mono), k1=kendall(t,t);
    printf("checks: spearman(mono)=%.3f kendall(identity)=%.3f\n",r1,k1);
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

// N diverse designs via a deterministic low-discrepancy (golden-ratio) sweep of
// the ventilation-relevant axes — good coverage for any N, reproducible.
static std::vector<Design> make_designs(int N){
    auto frac=[&](int i,double g){ double v=i*g; return v-(long)v; };
    std::vector<Design> d;
    for(int i=0;i<N;++i){
        double sw=8  +32 *frac(i,0.6180339887);     // street_width 8–40 m
        double bw=28 +44 *frac(i,0.7548776662);     // block_w 28–72
        double bd=16 +24 *frac(i,0.5698402910);     // block_d 16–40
        double rn=0.0+0.8 *frac(i,0.3247179572);     // roughness 0–0.8 (was dead base_height)
        double cp=20 +100*frac(i,0.8191725134);     // cbd_peak 20–120
        double pc=0.0+1.0 *frac(i,0.4358974359);    // park_centrality 0–1
        d.push_back({sw,bw,bd,rn,cp,pc});
    }
    return d;
}
static Params design_params(const Design& d,double city_m){
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

// one (design, floor) evaluation → fills J_vanleer, J_upwind, reciprocity, integrity
struct Eval { double Jvl, Jup, Jens, recip; double emitted, dep, air; bool finite; };

static Eval evaluate(const Params& p, const char* nu_floor,
                     double rel_s, int warm, int nz_fixed){
    setenv("NU_FLOOR", nu_floor, 1);          // set the Reynolds floor for this run
    Result r=generate(p);
    int NX=r.nx_cells,NY=r.ny_cells;
    VoxelGrid g=voxelize(p,r,true,0.0,1800.0,0.5,nz_fixed);  // dep_dp=0 → passive tracer; nz fixed across sweep
    int NZ=g.nz, N3=NX*NY*NZ;
    lbm::Config c{}; c.nx=g.nx;c.ny=g.ny;c.nz=g.nz; c.cell_size=(float)CELL;
    c.U_inlet=5.f;c.wind_angle=0.f;c.nu_phys=1.5e-5f;c.Cw=0.325f;c.Sc_t=0.7f;c.D_mol=1e-5f;
    c.source_x=p.source_x;c.source_y=p.source_y;c.source_z=4.f;c.Q_source=1.f;
    c.particle_diam=0.f;c.particle_density=1800.f;
    c.max_steps=2000000;c.check_interval=500;c.conv_threshold=1e-4f;
    c.max_warmup=warm;c.avg_threshold=0.f;c.release_time=(float)rel_s;   /* 0 ⇒ solver default stationarity tol (5e-3) */
    c.inlet_profile=1;c.abl_z0=0.7f;c.abl_zref=40.f;c.abl_Lturb=40.f;c.abl_nmodes=100;
    c.abl_sigu_ratio=2.5f;c.abl_sigv_ratio=1.9f;c.abl_sigw_ratio=1.25f;

    lbm::Solver s(c);
    s.load_geometry(g.type.data(),g.perm.data(),g.inh.data(),g.dep_vel.data());
    auto res=s.run();
    Eval e{}; e.emitted=res.total_emitted; e.dep=res.total_deposited; e.air=res.total_airborne;
    e.finite=std::isfinite(res.total_airborne)&&std::isfinite(res.total_deposited);

    // receptor field w (effective inhabitance) — common to both schemes
    double street_pop=street_fraction(nhaps_average(),OUTDOOR_PARK_SHARE)*r.population;
    auto sf=build_street_occupancy(p,r,g,street_pop);
    std::vector<float> w; adj::build_receptor_field(r,g.type.data(),NX,NY,NZ,sf,w);

    // J_vanleer = Σ w·TIAC (production scalar, z=1)
    s.export_tiac("rob_tiac.bin",1);
    std::vector<float> tvl(NX*NY);
    { FILE* f=fopen("rob_tiac.bin","rb"); int h[2]; size_t rd=fread(h,4,2,f);(void)rd;
      rd=fread(tvl.data(),4,NX*NY,f); fclose(f); }
    double jvl=0.0; for(int yx=0;yx<NX*NY;++yx) jvl+=(double)w[(size_t)1*NY*NX+yx]*tvl[yx];

    // linear-upwind on the frozen mean flow, SAME point source
    std::vector<float> ux,uy,uz,nut; s.copy_mean_flow_to_host(ux,uy,uz,nut);
    std::vector<float> vdep(N3,0.f);
    adj::Field F{NX,NY,NZ,N3,g.type.data(),ux.data(),uy.data(),uz.data(),nut.data(),
                 vdep.data(),1e-4f,0.7f,0.f,0.f}; F.dt=adj::stable_dt(F);
    std::vector<float> sp(N3,0.f);
    // IDENTICAL cell to the LBM source_cell() (lbm_solver.cpp): truncation (not
    // round) with the same clamps, so van-Leer and linear-upwind disperse from the
    // exact same source — otherwise the J comparison would not be apples-to-apples.
    int sx=std::clamp((int)(p.source_x/CELL),0,NX-1);
    int sy=std::clamp((int)(p.source_y/CELL),0,NY-1);
    int sz=std::clamp(std::max(1,(int)(4.0/CELL)),1,NZ-1);   // source_z = 4 m, as in Config
    sp[(size_t)(sz*NY+sy)*NX+sx]=1.f;
    float umax=1e-3f; for(int i=0;i<N3;++i){ float spd=std::fabs(ux[i])+std::fabs(uy[i])+std::fabs(uz[i]);
        if(g.type[i]==adj::FLUID && spd>umax) umax=spd; }
    int nstep=(int)std::min(6000.0,std::max(600.0,3.0*NX/umax/F.dt));
    // Evaluate the linear-upwind objective two ways on the SAME flow & source:
    //   forward  J_fwd = Σ w·C   (the objective used for ranking)
    //   reverse  J_rev = Σ F·s   (adjoint footprint contracted with the source)
    // adj_step is the exact transpose of fwd_step *by construction for any flow*, so
    // J_fwd == J_rev to float precision regardless of the field. reciprocity_rel is
    // therefore a wiring / NaN / dimension regression check on the production path
    // (flow extraction, receptor & source builders) — not a new math property.
    std::vector<float> Foot;
    double j_rev = adj::effective_exposure_reverse(F, w, sp, nstep, /*Jfwd*/nullptr, &Foot);
    double j_fwd = adj::forward_objective(F, sp, w, nstep);
    e.Jvl=jvl; e.Jup=j_fwd;          // forward value is the objective (matches J_vanleer form)
    e.recip=std::fabs(j_fwd-j_rev)/std::max({std::fabs(j_fwd),std::fabs(j_rev),1e-300});

    // ── Ensemble exposure objective ⟨F, p⟩ ──────────────────────────────────
    // Instead of one point source, contract the SAME footprint F (already solved
    // above — it is source-independent) with a UNIFORM source prior p(x) over all
    // open city ground (roads + parks, z=1). This is E_p[J]: the population's
    // expected exposure averaged over an equally-likely spread of release
    // locations. One extra dot product per city — no extra solve. Averaging over
    // sources also smooths the per-plume numerical-diffusion artefacts that make
    // the single-point objective scheme-sensitive.
    std::vector<float> s_uni;
    adj::build_source_mask(r, p, g.type.data(), NX, NY, NZ, /*Q_total=*/1.0, s_uni);
    e.Jens = adj::dot(Foot, s_uni);
    return e;
}

static int run_full(int argc,char**argv){
    int N      =(argc>2)?atoi(argv[2]):48;
    double cm  =(argc>3)?atof(argv[3]):768.0;
    double rel =(argc>4)?atof(argv[4]):120.0;
    int    warm=(argc>5)?atoi(argv[5]):10000;
    auto designs=make_designs(N);
    // v8: size ONE fixed minimal directional domain for the whole sweep, for the
    // tallest design (every design shares an identical grid — never compare
    // different-sized domains). COST-732 / Tominaga et al. (2008): 5H inflow
    // (−x,−y), 15H outflow (+x,+y), 5H top headroom; power-of-2 NOT enforced.
    // Hmax is the ACTUAL tallest building any design generates (deterministic).
    double Hmax=0.0;
    for(int d=0; d<N; ++d){
        Params pp=design_params(designs[d],cm); Hmax=std::max(Hmax,generate(pp).max_height);
    }
    int NZ_FIX=nz_cost732(Hmax);
    printf("[robust] fixed minimal domain: Hmax=%.0fm  buffers 5H/15H (in/out)  nz=%d  (no pow2)\n",
           Hmax, NZ_FIX);
    std::vector<double> vl_hi(N),up_hi(N),vl_lo(N),up_lo(N);
    double worst_recip=0.0; int n_badmass=0, n_nonfinite=0;

    std::set<int> done;
    { FILE* rd=fopen("robustness_overnight.csv","r"); if(rd){ char line[512];
        bool first=true; while(fgets(line,sizeof line,rd)){ if(first){first=false;continue;}
            int di; if(sscanf(line,"%d,",&di)==1) done.insert(di); } fclose(rd); } }
    bool resume = !done.empty();
    FILE* csv=fopen("robustness_overnight.csv", resume?"a":"w");
    if(!resume){ fprintf(csv,"design,street_width,block_w,roughness,cbd_peak,park_centrality,"
                "Jvl_tau530,Jup_tau530,Jvl_tau515,Jup_tau515,Jens_tau530,Jens_tau515,"
                "recip_530,recip_515,outflow_frac_530,outflow_frac_515,mass_ok,finite\n");
        fflush(csv); }
    else printf("[robust] RESUME: %zu design(s) already done, skipping them\n", done.size());
    printf("[robust] N=%d  city=%.0fm  release=%.0fs  warmup=%d  (2N=%d flow solves)\n",
           N,cm,rel,warm,2*N);

    for(int d=0; d<N; ++d){
        if(done.count(d)){ printf("[robust] design %d/%d already done — skipped\n",d+1,N); continue; }
        Params p=design_params(designs[d],cm);
        static const double IN_H = getenv("INFLOW_H") ? atof(getenv("INFLOW_H")) : 5.0;   // INFLOW_H env
        static const double OUT_H= getenv("OUTFLOW_H")? atof(getenv("OUTFLOW_H")):15.0;   // OUTFLOW_H env (trim for memory)
        minimal_domain(p, Hmax, IN_H, OUT_H);          // COST-732 default 5H inflow / 15H outflow (+x)
        p.source_x=p.buf_xn-100.0;                      // 100 m upstream of city, inside the inflow buffer
        p.source_y=p.buf_yn+p.city_h*0.5;               // on the city centreline (y)
        printf("\n[robust] === design %d/%d  sw=%.0fm rn=%.2f cbd=%.0f park=%.2f ===\n",
               d+1,N,designs[d].street_width,designs[d].roughness,designs[d].cbd_peak,
               designs[d].park_centrality);

        printf("[robust]  tau=0.530 (NU_FLOOR=0.010)…\n");
        Eval hi=evaluate(p,"0.010",rel,warm,NZ_FIX);
        printf("[robust]  tau=0.515 (NU_FLOOR=0.005)…\n");
        Eval lo=evaluate(p,"0.005",rel,warm,NZ_FIX);

        vl_hi[d]=hi.Jvl; up_hi[d]=hi.Jup; vl_lo[d]=lo.Jvl; up_lo[d]=lo.Jup;
        // Conservation SANITY (not a closure test): emitted = dep + air + outflow,
        // and outflow is computed by difference and NOT exposed, so |emitted-(dep+air)|
        // is identically the outflow — large by design (open downwind boundary) and
        // untestable for closure. What IS testable: dep & air finite and ~non-negative,
        // and no GROSS mass creation. Healthy van-Leer runs show ~1% imbalance from
        // accounting (airborne sampled at the final instant vs emitted=Q·dt·n_release)
        // — benign; only an order-of-magnitude blowup signals a real bug. (Observed in
        // a real run: outflow=-0.6%, well within tolerance.) outflow_frac logged for info.
        auto mass_sane=[](const Eval& e)->bool{
            if(!std::isfinite(e.dep)||!std::isfinite(e.air)) return false;
            double E=std::fabs(e.emitted); if(E<=0) return true;
            if(e.dep < -0.02*E) return false;                  // no sizeable negative deposition
            if(e.air < -0.02*E) return false;                  // no sizeable negative airborne
            double outflow=e.emitted-e.dep-e.air;
            return outflow >= -0.05*E;                         // no gross (>5%) mass creation
        };
        bool sane_hi=mass_sane(hi), sane_lo=mass_sane(lo);
        double of_hi = hi.emitted>0.0 ? (hi.emitted-hi.dep-hi.air)/hi.emitted : 0.0;
        double of_lo = lo.emitted>0.0 ? (lo.emitted-lo.dep-lo.air)/lo.emitted : 0.0;
        worst_recip=std::max({worst_recip,hi.recip,lo.recip});
        if(!sane_hi||!sane_lo) ++n_badmass;
        if(!hi.finite||!lo.finite) ++n_nonfinite;

        printf("[robust]  Jvl530=%.3e Jup530=%.3e | Jvl515=%.3e Jup515=%.3e | recip=%.1e/%.1e | outflow=%.0f%%/%.0f%%\n",
               hi.Jvl,hi.Jup,lo.Jvl,lo.Jup,hi.recip,lo.recip,100*of_hi,100*of_lo);
        fprintf(csv,"%d,%.3f,%.1f,%.2f,%.1f,%.2f,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.3e,%.3e,%.3e,%.3e,%d,%d\n",
                d,designs[d].street_width,designs[d].block_w,designs[d].roughness,
                designs[d].cbd_peak,designs[d].park_centrality,
                hi.Jvl,hi.Jup,lo.Jvl,lo.Jup,hi.Jens,lo.Jens,hi.recip,lo.recip,of_hi,of_lo,
                (sane_hi&&sane_lo)?1:0,(hi.finite&&lo.finite)?1:0);
        fflush(csv);
    }
    fclose(csv);

    int k=std::max(1,N/3);
    printf("\n================ ROBUSTNESS SUMMARY (N=%d, top-k=%d) ================\n",N,k);
    rank_line("van-Leer  tau=0.530",vl_hi);
    rank_line("upwind    tau=0.530",up_hi);
    rank_line("van-Leer  tau=0.515",vl_lo);
    rank_line("upwind    tau=0.515",up_lo);
    printf("\n  -- axis A: reverse-solver objective (mean+upwind) vs production truth (turbulent+vanLeer) --\n");
    printf("     (conflates numerical scheme AND mean-vs-turbulent flow; tests end-to-end usability)\n");
    compare("truth vs reverse-objective @ tau=0.530", vl_hi, up_hi, k);
    compare("truth vs reverse-objective @ tau=0.515", vl_lo, up_lo, k);
    printf("\n  -- axis B: Reynolds floor --\n");
    compare("tau=0.530 vs 0.515 (van-Leer truth)", vl_hi, vl_lo, k);
    compare("tau=0.530 vs 0.515 (upwind objective)", up_hi, up_lo, k);

    // overall: does the SAME design win across all four conditions?
    auto argmin=[&](const std::vector<double>& v){ int b=0; for(int i=1;i<N;++i) if(v[i]<v[b])b=i; return b; };
    int b1=argmin(vl_hi),b2=argmin(up_hi),b3=argmin(vl_lo),b4=argmin(up_lo);
    printf("\n  best design per condition: vl530=%d up530=%d vl515=%d up515=%d %s\n",
           b1,b2,b3,b4,(b1==b2&&b1==b3&&b1==b4)?"(unanimous)":"(differ)");
    bool scheme_ok = spearman(vl_hi,up_hi)>0.9 && spearman(vl_lo,up_lo)>0.9;
    bool floor_ok  = spearman(vl_hi,vl_lo)>0.9 && spearman(up_hi,up_lo)>0.9;
    printf("\n  INTEGRITY: worst reciprocity_rel=%.2e %s\n",
           worst_recip,(worst_recip<1e-3)?"[production wiring OK: transpose exact, no NaN]"
                                          :"[CHECK adjoint wiring / NaN in flow]");
    printf("             mass-sanity failures (neg or created)=%d/%d | non-finite J=%d/%d\n",
           n_badmass,N,n_nonfinite,N);
    printf("\n  VERDICT:\n");
    printf("   reverse-objective vs truth: %s\n", scheme_ok?
           "STABLE — the cheap reverse objective ranks designs like the production\n"
           "                               truth; usable for optimization. (A shift would NOT isolate\n"
           "                               cause — could be scheme or mean-vs-turbulent flow.)"
          :"SHIFT — the reverse objective and production truth disagree on design\n"
           "                               order. Isolate cause with a same-flow scheme test before trusting it.");
    printf("   Reynolds-floor ranking    : %s\n", floor_ok?
           "STABLE — the floor is an absolute bias, not a ranking threat."
          :"SHIFT — the floor changes design preference; finer cells / better collision needed.");
    printf("\n[robust] wrote robustness_overnight.csv\n");
    return 0;
}
#endif

int main(int argc,char**argv){
    std::string mode=(argc>1)?argv[1]:"selftest";
    if(mode=="selftest") return selftest();
#ifdef WITH_LBM
    if(mode=="run") return run_full(argc,argv);
#endif
    fprintf(stderr,"usage: %s selftest\n",argv[0]);
    fprintf(stderr,"       %s run [N=24] [city_m=768] [rel_s=120] [warmup=8000]  (needs WITH_LBM)\n",argv[0]);
    return 1;
}
