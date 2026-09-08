// linearity_guard.cpp — the OPENLB_MIGRATION_PLAN.md §6.2 / audit Gate 7b acceptance test.
//
// WHY THIS EXISTS. The exposure metric J = (1/|Ω|) Σ_x w(x)·Θ(x) is defined on a burst
// released over EVERY outdoor ground cell at once, and treated as the superposition of
// per-cell releases: J(s₁+…+sₙ) = Σ J(sᵢ). That is not a convenience — it is what makes a
// single solve equivalent to |Ω| solves, and it is the property EXPOSURE_METRIC.md's
// definition of J rests on. The advection–diffusion lattice and an Eulerian
// settling+sink deposition are both linear, so superposition SHOULD hold exactly. But
// "should" is not "does": a nonlinearity introduced by a limiter, a clamp (e.g. the α
// saturation in the deposition sink), a source term applied to a nonlinear quantity, or a
// solver instability would break it — and would do so SILENTLY, still producing a finite,
// plausible-looking J that no longer means what the metric says it means. Given the whole
// output of this project is a scalar ranking across city designs, that is the single most
// dangerous failure mode in the port. Hence a gate.
//
// WHAT IT CHECKS. The cheap, decisive version from the audit: release from cell a alone
// (Θ_a), from cell b alone (Θ_b), and from {a,b} together (Θ_ab). Then
//
//        Θ_ab(x)  ==  Θ_a(x) + Θ_b(x)      for every x, to solver tolerance.
//
// Two independent statistics are reported, because they fail differently:
//   * relative L2 over the whole field — catches a diffuse, everywhere-small bias
//     (e.g. an α clamp that only bites where the two plumes overlap);
//   * max pointwise relative error over SIGNIFICANT cells — catches a localised break.
// Cells far below the noise floor are excluded from the pointwise statistic (relative
// error on ~0 is meaningless), but they still count in the L2 norm.
//
// MODES
//   --selftest                      verify the CHECKER itself on synthetic fields
//                                   (runs anywhere; needs no OpenLB and no solver)
//   --dirs <A> <B> <AB>             compare three existing urban_flow output dirs
//   --run <binary> <geom> <work>    pick two Ω cells, drive the solver 3×, then compare
//
// Build:  g++ -O2 -std=c++17 -I. tests/linearity_guard.cpp -o linearity_guard
// Gate:   exit 0 = PASS. Anything else = do NOT trust J.

#include "geometry_loader.h"      // the dependency-free 5-int reader (part A)

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <array>
#include <algorithm>

// ── tolerances ───────────────────────────────────────────────────────────────
// 0.1% is the audit's number. It is loose for an exactly-linear operator in double
// precision, and deliberately so: the solver runs to a clearance-fraction stopping
// criterion, so the three runs need not stop at the identical step, which perturbs Θ
// slightly. A real nonlinearity shows up percent-scale, not at 1e-3.
static double TOL_L2   = 1e-3;
static double TOL_PT   = 1e-3;
// Cells below this fraction of max|Θ_ab| are excluded from the POINTWISE statistic only.
static double SIG_FRAC = 1e-6;

struct Stats { double l2rel, maxrel; long nsig; int worst[3]; };

// Compare Θ_ab against Θ_a + Θ_b.
static Stats compare(const std::vector<float>& A, const std::vector<float>& B,
                     const std::vector<float>& AB, int nx, int ny, int nz)
{
    Stats s{0,0,0,{0,0,0}};
    double num=0, den=0, peak=0;
    for (size_t i=0;i<AB.size();++i) peak = std::max(peak, (double)std::fabs(AB[i]));
    const double sigCut = peak * SIG_FRAC;

    for (size_t i=0;i<AB.size();++i) {
        const double sum = (double)A[i] + (double)B[i];
        const double d   = (double)AB[i] - sum;
        num += d*d; den += (double)AB[i]*(double)AB[i];
        if (std::fabs(sum) > sigCut) {
            ++s.nsig;
            const double rel = std::fabs(d) / std::fabs(sum);
            if (rel > s.maxrel) {
                s.maxrel = rel;
                const int z = (int)(i/((size_t)nx*ny));
                const int y = (int)((i - (size_t)z*nx*ny)/nx);
                const int x = (int)(i - (size_t)z*nx*ny - (size_t)y*nx);
                s.worst[0]=x; s.worst[1]=y; s.worst[2]=z;
            }
        }
    }
    s.l2rel = (den>0) ? std::sqrt(num/den) : (num>0 ? 1.0 : 0.0);
    return s;
}

static bool loadTheta(const std::string& dir, bridge::GridField<float>& g) {
    if (!bridge::read_grid(dir + "/theta.f32", g)) {
        fprintf(stderr, "[linearity_guard] cannot read %s/theta.f32\n", dir.c_str());
        return false;
    }
    return true;
}

// Gate 7a companion: the mass budget, if the solver wrote one.
static void reportBudget(const std::string& dir) {
    FILE* f = fopen((dir+"/meta_flow.txt").c_str(), "r");
    if (!f) return;
    char line[512]; double closure=-1, emit=0, dep=0, air=0, drained=0;
    while (fgets(line,sizeof line,f)) {
        sscanf(line,"budget_closure %lf",&closure);
        sscanf(line,"mass_emitted %lf",&emit);
        sscanf(line,"mass_deposited %lf",&dep);
        sscanf(line,"mass_airborne %lf",&air);
        sscanf(line,"mass_drained %lf",&drained);
    }
    fclose(f);
    if (emit>0) {
        const double resid = std::fabs(emit-(dep+air+drained))/emit;
        printf("   budget %s: emit=%.6e dep=%.6e air=%.6e drained=%.6e -> residual %.3e [%s]\n",
               dir.c_str(), emit, dep, air, drained, resid,
               resid < 0.01 ? "PASS <1%" : "FAIL");
        if (closure>=0 && std::fabs(closure-resid) > 1e-9)
            printf("   (solver reported budget_closure=%.3e)\n", closure);
    }
}

// ── self-test of the checker ─────────────────────────────────────────────────
// Fabricates a linear triple (must PASS) and three ways of breaking linearity that a real
// solver bug would produce (each must FAIL). Without this, a checker that always says
// PASS would be indistinguishable from a solver that is genuinely linear.
static int selftest() {
    const int nx=12, ny=10, nz=8; const size_t N=(size_t)nx*ny*nz;
    std::vector<float> A(N), B(N), AB(N);
    uint64_t s=0x9E3779B97F4A7C15ull;
    auto rnd=[&]{ s^=s<<13; s^=s>>7; s^=s<<17; return (double)(s>>11)/9007199254740992.0; };
    for (size_t i=0;i<N;++i) { A[i]=(float)rnd(); B[i]=(float)rnd(); }

    int failures=0;
    auto expect=[&](const char* name, const Stats& st, bool wantPass){
        const bool pass = (st.l2rel<=TOL_L2 && st.maxrel<=TOL_PT);
        const bool ok   = (pass==wantPass);
        printf("   %-34s L2=%.3e maxpt=%.3e -> %-4s (want %-4s) [%s]\n",
               name, st.l2rel, st.maxrel, pass?"PASS":"FAIL", wantPass?"PASS":"FAIL",
               ok?"ok":"CHECKER BROKEN");
        if(!ok) ++failures;
    };

    printf("[selftest] checker behaviour on synthetic fields\n");

    for (size_t i=0;i<N;++i) AB[i]=A[i]+B[i];
    expect("exactly linear", compare(A,B,AB,nx,ny,nz), true);

    // float round-off only — must still pass
    for (size_t i=0;i<N;++i) AB[i]=(float)((double)A[i]+(double)B[i]);
    expect("linear + float round-off", compare(A,B,AB,nx,ny,nz), true);

    // a 2% global gain error (e.g. a mis-scaled source term)
    for (size_t i=0;i<N;++i) AB[i]=(float)(1.02*((double)A[i]+(double)B[i]));
    expect("2% global gain error", compare(A,B,AB,nx,ny,nz), false);

    // saturation where the plumes overlap — the α-clamp failure mode
    for (size_t i=0;i<N;++i) { double v=(double)A[i]+(double)B[i]; AB[i]=(float)std::min(v,1.2); }
    expect("saturation on overlap (alpha clamp)", compare(A,B,AB,nx,ny,nz), false);

    // a single badly-wrong cell — must be caught by the pointwise statistic even though
    // it is invisible in L2 (this is why both statistics are reported)
    for (size_t i=0;i<N;++i) AB[i]=A[i]+B[i];
    AB[N/2] = (float)(1.5*((double)A[N/2]+(double)B[N/2]));
    { Stats st=compare(A,B,AB,nx,ny,nz);
      printf("   %-34s L2=%.3e maxpt=%.3e\n","one cell 50%% wrong",st.l2rel,st.maxrel);
      if (st.maxrel < TOL_PT) { printf("      [CHECKER BROKEN] pointwise stat missed it\n"); ++failures; }
      else printf("      [ok] caught pointwise at (%d,%d,%d)\n",st.worst[0],st.worst[1],st.worst[2]); }

    printf("[selftest] %s\n", failures? "FAILED — the checker itself is wrong":"OK — checker discriminates linear from nonlinear");
    return failures ? 1 : 0;
}

// ── pick two well-separated Ω cells from the source mask ─────────────────────
static bool pickCells(const std::string& geom, int a[3], int b[3]) {
    bridge::GridField<uint8_t> sm;
    if (!bridge::load_source_mask(geom+"/source_mask.u8", sm)) return false;
    std::vector<std::array<int,3>> cells;
    for (int z=0;z<sm.nz;++z) for (int y=0;y<sm.ny;++y) for (int x=0;x<sm.nx;++x)
        if (sm.data[sm.idx(x,y,z)]) cells.push_back(std::array<int,3>{x,y,z});
    if (cells.size() < 2) { fprintf(stderr,"[linearity_guard] need >=2 Omega cells, found %zu\n", cells.size()); return false; }
    // Two cells about a third of the way in from each end of the Ω list: far enough apart
    // that their plumes overlap only downstream (which is exactly where a nonlinearity in
    // the deposition sink would show up), but both well inside the domain.
    const auto& c0 = cells[cells.size()/3];
    const auto& c1 = cells[(2*cells.size())/3];
    for (int i=0;i<3;++i) { a[i]=c0[i]; b[i]=c1[i]; }
    return true;
}

static int runSolver(const std::string& bin, const std::string& geom,
                     const std::string& out, const std::string& srcCells) {
    std::string cmd = "GEOM_DIR='"+geom+"' OUT_DIR='"+out+"' SRC_CELLS='"+srcCells+"' '"+bin+"'";
    printf("   $ %s\n", cmd.c_str());
    const int rc = system(cmd.c_str());
    if (rc != 0) fprintf(stderr,"[linearity_guard] solver exited %d for %s\n", rc, srcCells.c_str());
    return rc;
}

int main(int argc, char** argv) {
    if (const char* e=getenv("LIN_TOL"))     { TOL_L2=TOL_PT=atof(e); }
    if (const char* e=getenv("LIN_SIGFRAC")) { SIG_FRAC=atof(e); }

    if (argc>=2 && !strcmp(argv[1],"--selftest")) return selftest();

    std::string dA,dB,dAB;

    if (argc>=5 && !strcmp(argv[1],"--dirs")) {
        dA=argv[2]; dB=argv[3]; dAB=argv[4];
    }
    else if (argc>=5 && !strcmp(argv[1],"--run")) {
        const std::string bin=argv[2], geom=argv[3], work=argv[4];
        int a[3],b[3];
        if (!pickCells(geom,a,b)) return 2;
        char sa[64],sb[64],sab[160];
        snprintf(sa ,sizeof sa ,"%d,%d,%d",a[0],a[1],a[2]);
        snprintf(sb ,sizeof sb ,"%d,%d,%d",b[0],b[1],b[2]);
        snprintf(sab,sizeof sab,"%d,%d,%d;%d,%d,%d",a[0],a[1],a[2],b[0],b[1],b[2]);
        printf("[linearity_guard] release cells: a=(%s)  b=(%s)\n", sa, sb);
        dA=work+"/lin_a"; dB=work+"/lin_b"; dAB=work+"/lin_ab";
        // Same inlet seed for all three runs: the LIVE flow must be identical across them,
        // or the comparison measures turbulence decorrelation instead of linearity.
        setenv("ABL_SEED", getenv("ABL_SEED") ? getenv("ABL_SEED") : "1000", 1);
        if (runSolver(bin,geom,dA ,sa )) return 3;
        if (runSolver(bin,geom,dB ,sb )) return 3;
        if (runSolver(bin,geom,dAB,sab)) return 3;
    }
    else {
        fprintf(stderr,
          "usage:\n"
          "  %s --selftest\n"
          "  %s --dirs <outA> <outB> <outAB>\n"
          "  %s --run <urban_flow_binary> <geom_dir> <work_dir>\n", argv[0],argv[0],argv[0]);
        return 64;
    }

    bridge::GridField<float> A,B,AB;
    if (!loadTheta(dA,A) || !loadTheta(dB,B) || !loadTheta(dAB,AB)) return 2;
    if (A.size()!=B.size() || A.size()!=AB.size()) {
        fprintf(stderr,"[linearity_guard] grid size mismatch: %zu / %zu / %zu\n",
                A.size(),B.size(),AB.size());
        return 2;
    }

    printf("\n=== LINEARITY GUARD (plan §6.2 / Gate 7b) ===\n");
    printf("   grid %dx%dx%d dx=%.3f m\n", AB.nx,AB.ny,AB.nz,AB.dx);
    reportBudget(dA); reportBudget(dB); reportBudget(dAB);

    const Stats st = compare(A.data,B.data,AB.data,AB.nx,AB.ny,AB.nz);
    printf("\n   Theta_ab  vs  Theta_a + Theta_b\n");
    printf("   relative L2 over field   : %.3e   (tol %.1e)\n", st.l2rel, TOL_L2);
    printf("   max pointwise rel. error : %.3e   (tol %.1e) at (%d,%d,%d), over %ld significant cells\n",
           st.maxrel, TOL_PT, st.worst[0],st.worst[1],st.worst[2], st.nsig);

    const bool pass = (st.l2rel<=TOL_L2 && st.maxrel<=TOL_PT);
    if (pass) {
        printf("\n   GATE 7b: PASS — superposition holds; J is the quantity EXPOSURE_METRIC.md defines.\n");
        return 0;
    }
    printf("\n   GATE 7b: FAIL — superposition is broken. J from a full-Omega burst is NOT the\n"
           "   sum of per-cell releases, so it does not mean what the exposure metric says.\n"
           "   Do not rank designs on it. Look first at: the deposition alpha clamp\n"
           "   (min(1, 8*v_d_lb) saturating), any limiter in the AD collision, and whether the\n"
           "   three runs really saw the identical flow (same ABL_SEED, same step count).\n");
    return 1;
}
