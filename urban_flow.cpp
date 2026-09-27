// urban_flow.cpp — STAGE B (airflow, LIVE flow): OpenLB 1.8 NSE + WALE LES engine for the
// urban_openlbm migration. Consumes the geometry-bridge outputs (material_map.dat +
// source_mask.u8), develops a LIVE turbulent wind field on the city, and hands it to the
// scalar burst (Step 4). No time-averaging — this is the FORWARD_LIVE regime: the burst
// rides the instantaneous, evolving flow (matching forward_city.cpp), so the airflow stage
// only needs to reach a statistically-developed turbulent state, then snapshot it.
//
// ┌─ SCOPE ──────────────────────────────────────────────────────────────────────┐
// │ Step 3 of OPENLB_MIGRATION_PLAN.md §7: airflow only — D3Q19 + WALE LES,        │
// │ ABL/RFG velocity inlet, pressure outlet, free-slip sides/top, rough-wall floor,│
// │ developed LIVE (no averaging). Step 4 (AD scalar + deposition) is STUBBED below.│
// └──────────────────────────────────────────────────────────────────────────────┘
//
// CORRECTIVE / STABILISATION METHODS wired here (the numerical safety net the custom
// engine had — see TECHNICAL_STATUS notes on ABL drift and low-Re stability):
//   (C1) Mach / CFL / τ preflight — refuse to run outside the stable low-Ma regime.
//   (C2) Rough-wall function on the GROUND (z0) — preserves ABL horizontal homogeneity;
//        buildings stay smooth no-slip bounce-back (COST 732). Needs the ground split
//        (MAT_GROUND) added to the geometry bridge.
//   (C3) Inlet startup RAMP (smoothstep over 1 flow-through) — no pressure shock.
//   (C4) DIVERGENCE / NaN guard — abort cleanly on non-finite or super-Mach state
//        (mirrors forward_city.cpp's guard) instead of silently producing garbage.
//   (C5) Outlet SPONGE layer — a graded high-viscosity fringe before the outlet damps
//        turbulent structures so they don't reflect off the pressure boundary.
//   (C6) LES stabilisation — selectable collision (WALE default / consistent-Smagorinsky
//        / regularized) + optional ADM (approximate-deconvolution) filtering.
//
// ⚠ VERIFICATION STATUS: written against the OpenLB app idiom + the 1.8 User Guide, NOT
// compiled here (no OpenLB in the authoring env). Compiles inside YOUR 1.8 tree. Every
// version-sensitive call is marked "CONFIRM 1.8". Structural starting point, not verified.
//
// ── v1.3→1.8 API deltas to apply while compiling ──
//   SuperLattice3D→SuperLattice ; SuperGeometry3D<T>→SuperGeometry<T,3> ; virtual
//   Dynamics*→dynamics tuples ; createInterpBoundaryCondition3D→setInterpolatedVelocity/
//   PressureBoundary ; WALE descriptor/dynamics names ; add Platform::GPU_CUDA for the A4000.
//
// Build (drop in examples/urban/urban_flow/, then `make`).  Run: GEOM_DIR=./geom_out ./urban_flow

#define URBAN_FLOW_WITH_OPENLB 1

#include "olb3D.h"
#ifndef OLB_PRECOMPILED
#include "olb3D.hh"
#endif

#include "geometry_loader.h"   // Stage-A material map + source mask reader/stamper
#include "abl_inlet_olb.h"     // verified ABL/RFG inlet as AnalyticalF3D (with startup ramp)

#include <cstdlib>
#include <string>
#include <vector>
#include <cmath>

using namespace olb;
using namespace olb::descriptors;
typedef double T;

// ── B5 (corrected): THETA already ships in 1.8 as FIELD_BASE<1,0,0>
// (src/descriptor/fields.h:423), so defining it here was a redefinition. Only DEPOSIT is
// genuinely ours. One fewer field to justify.
namespace olb::descriptors {
struct DEPOSIT : public FIELD_BASE<1> {};   // deposited mass at wall-adjacent cells
}

// ── LES collision selection (C6) ──  -DCOLLISION_MODEL=0|1|2  (default 0 = WALE)
#ifndef COLLISION_MODEL
#define COLLISION_MODEL 0
#endif
// B6: WALED3Q19Descriptor is D3Q19<EFFECTIVE_OMEGA,VELO_GRAD> and carries NO POROSITY
// field, so PorousBGKdynamics cannot instantiate on it. Spell the descriptor out with
// POROSITY added rather than using the alias.
using UrbanD3Q19Descriptor = D3Q19<EFFECTIVE_OMEGA,VELO_GRAD,POROSITY>;
#define DESCRIPTOR UrbanD3Q19Descriptor
#if   COLLISION_MODEL==0
  #define BULK_DYNAMICS WALEBGKdynamics                 // CONFIRM 1.8
#elif COLLISION_MODEL==1
  #define BULK_DYNAMICS ConStrainSmagorinskyBGKdynamics // consistent-strain Smagorinsky (more dissipative/stable)
#else
  #define BULK_DYNAMICS RLBdynamics                     // regularized LB (most robust, more diffusive)
#endif

// ── Step 4: advection–diffusion scalar lattice (D3Q7) ──
// CONFIRM 1.8: AD descriptor carries a VELOCITY field (the coupled NSE velocity) and, here,
// two extra scalar fields we accumulate on-device (GPU-correct, §6.3):
//   THETA  — running Θ = ∫C dt   ;   DEPOSIT — deposited mass at wall-adjacent cells.
#define AD_DESCRIPTOR D3Q7<VELOCITY,THETA,DEPOSIT>      // CONFIRM 1.8 field names
#define AD_DYNAMICS   AdvectionDiffusionBGKdynamics     // CONFIRM 1.8

static double envd(const char* k,double d){const char* e=getenv(k);return e?atof(e):d;}
static int    envi(const char* k,int d){const char* e=getenv(k);return e?atoi(e):d;}

// Material numbers — MUST match openlb_geometry.h (note MAT_GROUND split from MAT_WALL,
// and MAT_SPONGE is a SOLVER-LOCAL material carved out of MAT_FLUID near the outlet (C5)).
enum { MAT_VOID=0, MAT_FLUID=1, MAT_WALL=2, MAT_INLET=3, MAT_OUTLET=4,
       MAT_SLIP=5, MAT_POROUS=6, MAT_GROUND=7, MAT_SPONGE=8 };

static abl::ABLInlet gInlet;   // built in main, referenced in setBoundaryValues

// ─────────────────────────────────────────────────────────────────────────────
// (C5) Carve a sponge band out of the fluid cells adjacent to the outlet face, on the
// HOST material array before stamping. Wind-aligned: default +x → band at
// x∈[nx-buf-n, nx-buf), i.e. SPONGE_BUFFER fluid cells short of the outlet plane.
// Solver-local (MAT_SPONGE); the geometry bridge stays pure geometry.
// ─────────────────────────────────────────────────────────────────────────────
static void carve_sponge(bridge::GridField<int32_t>& m, double wind_deg, int nSponge) {
    if (nSponge <= 0) return;
    const int nx=m.nx, ny=m.ny, nz=m.nz;
    double wd = std::fmod(wind_deg,360.0); if (wd<0) wd+=360.0;
    auto mark=[&](int x,int y,int z){ size_t i=m.idx(x,y,z); if(m.data[i]==MAT_FLUID) m.data[i]=MAT_SPONGE; };
    // SPONGE_BUFFER cells of FLUID are left between the band and the outlet plane. OpenLB
    // derives each boundary cell's inward normal from its material neighbourhood; if the
    // sponge abuts the outlet, every outlet cell's only inward neighbour is MAT_SPONGE and
    // the normal may not be derivable. One cell of MAT_FLUID keeps that unambiguous at
    // negligible cost to the absorbing layer. SPONGE_BUFFER=0 restores the old placement.
    const int buf = envi("SPONGE_BUFFER", 1);
    if (wd < 45 || wd >= 315) for(int z=0;z<nz;++z)for(int y=0;y<ny;++y)for(int x=nx-buf-nSponge;x<nx-buf;++x) mark(x,y,z); // +x
    else if (wd < 135)        for(int z=0;z<nz;++z)for(int x=0;x<nx;++x)for(int y=ny-buf-nSponge;y<ny-buf;++y) mark(x,y,z); // +y
    else if (wd < 225)        for(int z=0;z<nz;++z)for(int y=0;y<ny;++y)for(int x=buf;x<buf+nSponge;++x)       mark(x,y,z); // -x
    else                      for(int z=0;z<nz;++z)for(int x=0;x<nx;++x)for(int y=buf;y<buf+nSponge;++y)      mark(x,y,z); // -y
}

// ─────────────────────────────────────────────────────────────────────────────
// (C1) Mach / CFL / relaxation preflight — pure converter arithmetic, no lattice yet.
// ─────────────────────────────────────────────────────────────────────────────
static bool preflight(UnitConverter<T,DESCRIPTOR> const& c) {
    OstreamManager clout(std::cout,"preflight");
    const T uLB  = c.getCharLatticeVelocity();                 // CONFIRM 1.8 accessor
    const T tau  = c.getLatticeRelaxationTime();               // CONFIRM 1.8
    const T Ma   = uLB * std::sqrt(3.0);
    bool ok=true; auto need=[&](bool cnd,const char* msg){ clout<<"["<<(cnd?"PASS":"FAIL")<<"] "<<msg<<std::endl; if(!cnd) ok=false; };
    clout << "uLB=" << uLB << "  tau=" << tau << "  Ma=" << Ma << std::endl;
    need(std::isfinite(uLB)&&uLB>0, "lattice velocity finite and positive");
    need(tau > 0.5,                 "relaxation tau > 0.5 (positive effective viscosity)");
    need(Ma  < 0.1,                 "inlet Mach < 0.1 (safe low-compressibility regime)");
    need(uLB < 0.1,                 "lattice velocity < 0.1 (CFL / stability margin)");
    clout << "preflight " << (ok?"PASS — cleared to run":"FAIL — refusing to run (set FORCE=1 to override)") << std::endl;
    return ok;
}

// ─────────────────────────────────────────────────────────────────────────────
// prepareLattice — dynamics + boundary conditions + correctives by material number.
// ─────────────────────────────────────────────────────────────────────────────
void prepareLattice(SuperLattice<T,DESCRIPTOR>& sLattice,
                    UnitConverter<T,DESCRIPTOR> const& converter,
                    SuperGeometry<T,3>& superGeometry, int nSponge) {   // CONFIRM 1.8 types
    OstreamManager clout(std::cout, "prepareLattice");
    const T omega = converter.getLatticeRelaxationFrequency(); (void)omega;  // B4: unused now

    sLattice.defineDynamics<NoDynamics>(superGeometry, MAT_VOID);                    // B3

    // bulk fluid + inlet + outlet carry the selected LES dynamics (C6)
    auto bulkInd = superGeometry.getMaterialIndicator({MAT_FLUID, MAT_INLET, MAT_OUTLET});
    sLattice.template defineDynamics<BULK_DYNAMICS<T,DESCRIPTOR>>(bulkInd);          // CONFIRM 1.8

    // buildings: smooth no-slip bounce-back
    sLattice.defineDynamics<BounceBack>(superGeometry, MAT_WALL);                    // B3

    // (C2) GROUND: rough-wall FUNCTION (z0), NOT plain bounce-back — this is what holds the
    // ABL profile horizontally homogeneous over the fetch. CONFIRM 1.8: exact API varies by
    // release; typical form is a wallFunction boundary taking the converter + a param struct
    // (wall profile Musker/power-law, roughness z0, rhoMethod, van-Driest). If your build has
    // no wall-function BC, fall back to bounce-back and accept the drift (set GROUND_BOUNCEBACK=1).
    // B8/G1: OpenLB 1.8 has NO aerodynamic-roughness (z0) wall function. wallFunctionParam
    // has no z0 member, and setTurbulentWallModel/WallModelParameters is a smooth-wall
    // Musker model. So the z0 branch cannot be written against 1.8 at all. Use the
    // unresolved no-slip floor for now; Gate 6a decides G1 on measurement (free-slip
    // approach floor / this / setTurbulentWallModel), not on argument.
    sLattice.defineDynamics<BounceBack>(superGeometry, MAT_GROUND);                  // B3+B8

    // porous park canopy (PorousBGK — set porosity field to the calibrated C_d/LAD)
    sLattice.template defineDynamics<PorousBGKdynamics<T,DESCRIPTOR>>(
        superGeometry.getMaterialIndicator({MAT_POROUS}));                           // CONFIRM 1.8
    // CONFIRM 1.8: set the POROSITY external field on MAT_POROUS cells, e.g.
    //   AnalyticalConst3D<T,T> por(envd("PARK_POROSITY",0.8));
    //   sLattice.defineField<POROSITY>(superGeometry.getMaterialIndicator({MAT_POROUS}), por);

    // (C5) SPONGE fringe: strongly-dissipative Smagorinsky just before the outlet.
    if (nSponge > 0) {
        sLattice.template defineDynamics<SmagorinskyBGKdynamics<T,DESCRIPTOR>>(
            superGeometry.getMaterialIndicator({MAT_SPONGE}));                       // CONFIRM 1.8
        // CONFIRM 1.8: raise the Smagorinsky constant on the sponge (e.g. Cs≈0.5) so it acts
        // as an absorbing layer; set via the dynamics' Cs field / an AnalyticalConst.
    }

    // ── domain boundaries ──
    // B4: the set*Boundary free functions are gone; 1.8 uses the declarative
    // boundary::set<> API, which takes omega from the cell dynamics rather than an argument.
    //
    // Each call is announced and individually caught so a "Could not set Boundary" throw
    // names the material that failed and how many cells it had, instead of aborting
    // anonymously. OpenLB derives a discrete inward normal per boundary cell from its
    // material neighbourhood, so this failing is a statement about geometry adjacency, and
    // the material number plus cell count is the minimum needed to chase it.
    auto& gstat = superGeometry.getStatistics();
    auto trySet = [&](const char* what, int matNo, auto&& fn) {
        clout << "boundary " << what << " on MAT " << matNo
              << " (" << gstat.getNvoxel(matNo) << " cells) ... " << std::flush;
        try { fn(); clout << "ok" << std::endl; }
        catch (std::exception const& e) {
            clout << "FAILED: " << e.what() << std::endl;
            clout << "  MAT " << matNo << " has no derivable inward normal for at least one "
                  << "cell. Check that its neighbours include MAT_FLUID (1) specifically, "
                  << "not merely some other fluid-carrying material." << std::endl;
            throw;
        }
    };
    trySet("InterpolatedVelocity", MAT_INLET, [&]{
        boundary::set<boundary::InterpolatedVelocity<T,DESCRIPTOR>>(sLattice, superGeometry, MAT_INLET); });
    trySet("InterpolatedPressure", MAT_OUTLET, [&]{
        boundary::set<boundary::InterpolatedPressure<T,DESCRIPTOR>>(sLattice, superGeometry, MAT_OUTLET); });
    trySet("FullSlip", MAT_SLIP, [&]{
        boundary::set<boundary::FullSlip<T,DESCRIPTOR>>(sLattice, superGeometry, MAT_SLIP); });

    // initial condition: rest; the inlet ramps in over the first flow-through.
    // 1.8 takes AnalyticalF arguments, not a scalar and a std::vector, and the indicator
    // parameter is FunctorPtr&& -- so the indicator must be a fresh temporary at each call
    // rather than one named lvalue reused.
    AnalyticalConst3D<T,T> rhoOne(T(1));
    AnalyticalConst3D<T,T> uZero(T(0),T(0),T(0));
    sLattice.defineRhoU(
        superGeometry.getMaterialIndicator({MAT_FLUID,MAT_INLET,MAT_OUTLET,MAT_POROUS,MAT_SPONGE}),
        rhoOne, uZero);
    sLattice.iniEquilibrium(
        superGeometry.getMaterialIndicator({MAT_FLUID,MAT_INLET,MAT_OUTLET,MAT_POROUS,MAT_SPONGE}),
        rhoOne, uZero);
    sLattice.initialize();
    clout << "prepareLattice done (collision model " << COLLISION_MODEL
          << ", sponge " << nSponge << " cells)" << std::endl;
}

// ─────────────────────────────────────────────────────────────────────────────
// setBoundaryValues — refresh the ABL/RFG inlet every step at physical time, with the
// (C3) smoothstep startup ramp applied through the functor.
// ─────────────────────────────────────────────────────────────────────────────
void setBoundaryValues(SuperLattice<T,DESCRIPTOR>& sLattice,
                       UnitConverter<T,DESCRIPTOR> const& converter,
                       SuperGeometry<T,3>& superGeometry, int iT, int rampSteps) {
    const T physT = converter.getPhysTime(iT);                                       // CONFIRM 1.8
    const T ramp  = bridge::AblVelocityF3D<T,DESCRIPTOR,UnitConverter<T,DESCRIPTOR>>::smoothstep(
                        rampSteps>0 ? (T)iT/(T)rampSteps : T(1));
    bridge::AblVelocityF3D<T,DESCRIPTOR,UnitConverter<T,DESCRIPTOR>> ablU(converter, gInlet, physT, ramp);
    sLattice.defineU(superGeometry.getMaterialIndicator({MAT_INLET}), ablU);         // CONFIRM 1.8
    // CONFIRM 1.8 (GPU): push host-side defineU changes to device, e.g.
    //   sLattice.setProcessingContext(ProcessingContext::Simulation);
}

// ─────────────────────────────────────────────────────────────────────────────
// (C4) divergence / NaN guard — true = healthy, false = diverged (abort the run).
// ─────────────────────────────────────────────────────────────────────────────
static bool healthy(SuperLattice<T,DESCRIPTOR>& sLattice) {
    const T maxU = sLattice.getStatistics().getMaxU();          // CONFIRM 1.8 accessor
    if (!std::isfinite(maxU) || maxU > 0.4) {                   // >0.4 LU ⇒ super-Mach / blow-up
        OstreamManager clout(std::cout,"DIVERGED");
        clout << "*** FATAL: non-finite or super-Mach state (maxU=" << maxU
              << ") — coarsen dx / lower U / raise Cs / enable sponge, then retry ***" << std::endl;
        return false;
    }
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// exportLiveFlow — write the INSTANTANEOUS live field (NO averaging) in the project's
// 5-int format so Stage C (visualize_forward.py) reads it unchanged:
//   umean_full.f32 : ncomp=4 (ux,uy,uz,nut). "mean" here is a filename kept for viz
//   compatibility; the payload is the live snapshot the burst rides.
// ─────────────────────────────────────────────────────────────────────────────
void exportLiveFlow(SuperLattice<T,DESCRIPTOR>& sLattice,
                    UnitConverter<T,DESCRIPTOR> const& converter,
                    SuperGeometry<T,3>& superGeometry,
                    int nx,int ny,int nz,double dx,const std::string& outdir) {
    OstreamManager clout(std::cout,"exportLiveFlow");
    sLattice.setProcessingContext(ProcessingContext::Evaluation);                    // CONFIRM 1.8 (GPU→host)

    const size_t Ncell=(size_t)nx*ny*nz;
    std::vector<float> uf(4*Ncell, 0.f);
    auto IDX=[&](int x,int y,int z){ return (size_t)z*ny*nx + (size_t)y*nx + x; };

    // Gather per-cell velocity (and eddy viscosity) by iterating the local block lattices —
    // the low-level idiom that is most stable across OpenLB versions.
    auto& load = sLattice.getLoadBalancer();                                         // CONFIRM 1.8
    for (int iC=0; iC<load.size(); ++iC) {                                           // CONFIRM 1.8
        auto& block = sLattice.getBlock(iC);                                         // CONFIRM 1.8
        auto& bgeo  = superGeometry.getBlockGeometry(iC);
        const auto off=bridge::blockOffset(superGeometry,iC);                         // S1
        const int gx0=off[0], gy0=off[1], gz0=off[2];
        const int bnx=bgeo.getNx(), bny=bgeo.getNy(), bnz=bgeo.getNz();
        for (int x=0;x<bnx;++x) for(int y=0;y<bny;++y) for(int z=0;z<bnz;++z) {
            int X=gx0+x, Y=gy0+y, Z=gz0+z;
            if (X<0||X>=nx||Y<0||Y>=ny||Z<0||Z>=nz) continue;
            T u[3]={0,0,0};
            block.get(x,y,z).computeU(u);                                            // CONFIRM 1.8: lattice u
            const T cv = converter.getConversionFactorVelocity();                    // CONFIRM 1.8
            size_t id=IDX(X,Y,Z);
            uf[id]         = (float)(u[0]*cv);
            uf[Ncell+id]   = (float)(u[1]*cv);
            uf[2*Ncell+id] = (float)(u[2]*cv);
            // CONFIRM 1.8: eddy viscosity ν_t from the WALE effective-omega field, e.g.
            //   T omEff = block.get(x,y,z).template getField<descriptors::OMEGA>();
            //   uf[3*Ncell+id] = (float)converter.getPhysViscosity((1/omEff-0.5)/3 - nuLB_base);
            uf[3*Ncell+id] = 0.f;   // left 0 until the effective-omega field name is confirmed
        }
    }
    std::string fn = outdir + "/umean_full.f32";
    FILE* f=fopen(fn.c_str(),"wb");
    if(f){ int h[5]={nx,ny,nz,(int)std::lround(dx*1000.0),4};
           fwrite(h,sizeof(int),5,f); fwrite(uf.data(),sizeof(float),uf.size(),f); fclose(f);
           clout << "wrote live snapshot " << fn << std::endl; }
}

// ══════════════════ STEP 4: advection–diffusion transport + deposition ══════════════════
// Physics (CONTAMINANT_BC.md §1): the AD scalar rides the LIVE NSE flow (not a frozen mean);
// solid walls carry zero advective flux + a dry-deposition sink (α = 8·v_d_lb half-way
// bounce-back closure); the outlet does advective outflow with zero inflow; top/lateral are
// zero-flux; Ω emits an accidental burst over a ~2 s pulse. Gravitational settling w_s adds a
// −w_s ẑ advection offset. The budget M_emit = M_out + M_dep + M_air closes, so Θ = ∫C dt
// converges and the burst self-terminates at ~99% clearance.
//
// LINEARITY (§6.2): the AD lattice + Eulerian deposition are linear, so J(Σsᵢ)=ΣJ(sᵢ). The
// full-Ω burst == superposition of per-cell bursts; the acceptance gate is a single-cell vs
// Ω-slice release check plus a mass budget closing ~99%.
//
// §6.3 (GPU): the three custom operators below are written as HOST block loops for clarity and
// CPU-gate testing; for A4000 production reimplement each as an on-device post-processor
// (structure maps 1:1 — same per-cell arithmetic).

// Gate 4 (airflow first): Step 4 is compiled only with -DENABLE_STEP4. Phase 6 turns it on
// after S2 (deposition velocity needs lattice units) and S3 (AD omega is a discarded
// placeholder) are fixed and the 40^3 box gates 7a/7b/7c are written.
#ifdef ENABLE_STEP4

// per-cell Stage-A inputs mirrored to the host, grid-indexed
struct ScalarInputs {
    int nx,ny,nz; double dx;
    std::vector<uint8_t> src;   // Ω source mask (source_mask.u8)
    std::vector<float>   vd;    // surface dry-deposition velocity m/s (dep_vel.f32)
    std::vector<int32_t> mat;   // material map (for wall-adjacency / outflow tests)
    inline size_t idx(int x,int y,int z)const{return (size_t)z*ny*nx+(size_t)y*nx+x;}
};

// prepareScalarLattice — AD dynamics + BCs + NSE→AD velocity coupling.
void prepareScalarLattice(SuperLattice<T,AD_DESCRIPTOR>& adLattice,
                          SuperLattice<T,DESCRIPTOR>& nsLattice,
                          UnitConverter<T,DESCRIPTOR> const& converter,
                          SuperGeometry<T,3>& superGeometry, T D_mol, T Sc_t) {
    OstreamManager clout(std::cout,"prepareScalarLattice");
    // AD relaxation from the molecular + turbulent diffusivity D_eff = D_mol + ν_t/Sc_t.
    // CONFIRM 1.8: build a second UnitConverter (AdeUnitConverter) for the scalar, or set the
    // AD omega from D_eff directly. ν_t is spatially varying (WALE) — couple it per cell.
    // TODO(S3, Phase 6): this placeholder is discarded below, never applied. At omega=1 the
    // D3Q7 diffusivity is D=(1/omega-0.5)/4=0.125 lu ~ 40 m^2/s, ~100x the turbulent
    // diffusivity, making the plume pure diffusion. Derive from D_eff = D_mol + nu_t/Sc_t;
    // nu_t varies per cell under WALE, so this must become a per-cell effective omega.
    const T omegaAD = 1.0;  // placeholder — NOT APPLIED

    auto fluidish = superGeometry.getMaterialIndicator({MAT_FLUID,MAT_INLET,MAT_OUTLET,MAT_POROUS,MAT_SPONGE});
    adLattice.template defineDynamics<AD_DYNAMICS<T,AD_DESCRIPTOR>>(fluidish);          // CONFIRM 1.8
    adLattice.defineDynamics<NoDynamics>(superGeometry, MAT_VOID);                      // B3
    // zero-flux walls: bounce-back on buildings + ground (deposition handled by the sink op)
    adLattice.defineDynamics<BounceBack>(superGeometry, MAT_WALL);                      // B3
    adLattice.defineDynamics<BounceBack>(superGeometry, MAT_GROUND);                    // B3
    // outlet: advective outflow / zero inflow  (CONFIRM 1.8: setZeroGradientBoundary or a
    // convective/anti-bounce-back AD outflow on MAT_OUTLET)
    boundary::set<boundary::ZeroDistribution<T,AD_DESCRIPTOR>>(adLattice, superGeometry, MAT_OUTLET); // B4
    // top/lateral: zero-flux (bounce-back / mirror on MAT_SLIP)
    adLattice.defineDynamics<BounceBack>(superGeometry, MAT_SLIP);                      // B3

    // NSE → AD velocity coupling (advection on the LIVE flow).
    // CONFIRM 1.8: attach a NavierStokesAdvectionDiffusionCoupling generator between the two
    // lattices so each AD step advects on the current NSE velocity; the settling offset −w_s ẑ
    // is added to the coupled velocity in the coupling (or in the per-step velocity copy).
    AnalyticalConst3D<T,T> rhoZero(T(0));
    AnalyticalConst3D<T,T> uZeroAD(T(0),T(0),T(0));
    adLattice.defineRhoU(
        superGeometry.getMaterialIndicator({MAT_FLUID,MAT_INLET,MAT_OUTLET,MAT_POROUS,MAT_SPONGE}),
        rhoZero, uZeroAD);
    adLattice.initialize();
    (void)omegaAD;(void)converter;(void)D_mol;(void)Sc_t;(void)nsLattice;
    clout << "prepareScalarLattice done" << std::endl;
}

// per-step transport operators (host block loops; see §6.3 note above)
namespace step4 {
using ADLat = SuperLattice<T,AD_DESCRIPTOR>;

// copy live NSE velocity (+ settling offset) into the AD VELOCITY field
inline void couple(ADLat& ad, SuperLattice<T,DESCRIPTOR>& ns, SuperGeometry<T,3>& sg,
                   UnitConverter<T,DESCRIPTOR> const& conv, T w_s_lb) {
    auto& load = ad.getLoadBalancer();
    for (int iC=0; iC<load.size(); ++iC) {
        auto& ab=ad.getBlock(iC); auto& nb=ns.getBlock(iC);
        auto& bg=sg.getBlockGeometry(iC); const int bnx=bg.getNx(),bny=bg.getNy(),bnz=bg.getNz();
        for(int x=0;x<bnx;++x)for(int y=0;y<bny;++y)for(int z=0;z<bnz;++z){
            T u[3]; nb.get(x,y,z).computeU(u); u[2]-=w_s_lb;                            // CONFIRM 1.8: −w_s ẑ settling
            ab.get(x,y,z).template setField<descriptors::VELOCITY>(u);                 // CONFIRM 1.8
        }
    }
    (void)conv;
}

// inject the burst at Ω cells during the pulse; returns mass emitted this step
inline double injectBurst(ADLat& ad, SuperGeometry<T,3>& sg, const ScalarInputs& in,
                          T rate_lb, bool on) {
    if(!on) return 0.0; double emitted=0;
    auto& load=ad.getLoadBalancer();
    for(int iC=0;iC<load.size();++iC){ auto& ab=ad.getBlock(iC);
        auto& bg=sg.getBlockGeometry(iC); const auto off=bridge::blockOffset(sg,iC);          // S1
        const int gx=off[0],gy=off[1],gz=off[2];
        const int bnx=bg.getNx(),bny=bg.getNy(),bnz=bg.getNz();
        for(int x=0;x<bnx;++x)for(int y=0;y<bny;++y)for(int z=0;z<bnz;++z){
            int X=gx+x,Y=gy+y,Z=gz+z; if(X<0||X>=in.nx||Y<0||Y>=in.ny||Z<0||Z>=in.nz)continue;
            if(in.src[in.idx(X,Y,Z)]){ auto c=ab.get(x,y,z); T rho=c.computeRho();      // CONFIRM 1.8
                c.defineRho(rho+rate_lb); emitted+=rate_lb; } }
    }
    return emitted;
}

// deposition sink at fluid cells adjacent to a wall/ground/park face; returns mass deposited
inline double deposit(ADLat& ad, SuperGeometry<T,3>& sg, const ScalarInputs& in) {
    static const int dxn[6]={1,-1,0,0,0,0},dyn[6]={0,0,1,-1,0,0},dzn[6]={0,0,0,0,1,-1};
    double dep=0; auto& load=ad.getLoadBalancer();
    for(int iC=0;iC<load.size();++iC){ auto& ab=ad.getBlock(iC);
        auto& bg=sg.getBlockGeometry(iC); const auto off=bridge::blockOffset(sg,iC);          // S1
        const int gx=off[0],gy=off[1],gz=off[2];
        const int bnx=bg.getNx(),bny=bg.getNy(),bnz=bg.getNz();
        for(int x=0;x<bnx;++x)for(int y=0;y<bny;++y)for(int z=0;z<bnz;++z){
            int X=gx+x,Y=gy+y,Z=gz+z; if(X<0||X>=in.nx||Y<0||Y>=in.ny||Z<0||Z>=in.nz)continue;
            int32_t mt=in.mat[in.idx(X,Y,Z)]; if(mt!=MAT_FLUID&&mt!=MAT_POROUS&&mt!=MAT_SPONGE)continue;
            // find a solid/ground/park face neighbour and its v_d
            float vd=0.f; for(int d=0;d<6;++d){int xx=X+dxn[d],yy=Y+dyn[d],zz=Z+dzn[d];
                if(xx<0||xx>=in.nx||yy<0||yy>=in.ny||zz<0||zz>=in.nz)continue;
                int32_t nm=in.mat[in.idx(xx,yy,zz)];
                if(nm==MAT_WALL||nm==MAT_GROUND||nm==MAT_POROUS) vd=std::max(vd,in.vd[in.idx(xx,yy,zz)]); }
            if(vd<=0.f)continue;
            auto c=ab.get(x,y,z); T rho=c.computeRho();                                 // CONFIRM 1.8
            // TODO(S2, Phase 6): vd is PHYSICAL m/s (voxelize.h:65) but the closure needs
            // lattice units: alpha = 8*v_d_lb, v_d_lb = vd*dt/dx. At the current operating
            // point dt/dx = 0.0125, so this over-deposits by 80x and inverts the mass budget.
            T alpha=std::min<T>(1.0, 8.0*vd);   // alpha = 8*v_d_lb (half-way bounce-back closure)
            T removed=(alpha/8.0)*rho; c.defineRho(rho-removed);
            T acc=c.template getField<descriptors::DEPOSIT>(); c.template setField<descriptors::DEPOSIT>(acc+removed);
            dep+=removed; }
    }
    return dep;
}

// accumulate Θ += C·dt (on the THETA field) and return current airborne mass ΣC
inline double accumulateTheta(ADLat& ad, SuperGeometry<T,3>& sg, const ScalarInputs& in, T dt_lb) {
    double airborne=0; auto& load=ad.getLoadBalancer();
    for(int iC=0;iC<load.size();++iC){ auto& ab=ad.getBlock(iC);
        auto& bg=sg.getBlockGeometry(iC); const auto off=bridge::blockOffset(sg,iC);          // S1
        const int gx=off[0],gy=off[1],gz=off[2];
        const int bnx=bg.getNx(),bny=bg.getNy(),bnz=bg.getNz();
        for(int x=0;x<bnx;++x)for(int y=0;y<bny;++y)for(int z=0;z<bnz;++z){
            int X=gx+x,Y=gy+y,Z=gz+z; if(X<0||X>=in.nx||Y<0||Y>=in.ny||Z<0||Z>=in.nz)continue;
            int32_t mt=in.mat[in.idx(X,Y,Z)]; if(mt==MAT_VOID||mt==MAT_WALL||mt==MAT_GROUND)continue;
            auto c=ab.get(x,y,z); T C=c.computeRho();                                    // CONFIRM 1.8
            T th=c.template getField<descriptors::THETA>(); c.template setField<descriptors::THETA>(th+C*dt_lb);
            airborne+=C; }
    }
    return airborne;
}

// gather a THETA or DEPOSIT field to a host float array (grid-indexed) for export
inline void gatherField(ADLat& ad, SuperGeometry<T,3>& sg, const ScalarInputs& in,
                        int which /*0=THETA 1=DEPOSIT*/, std::vector<float>& out) {
    out.assign((size_t)in.nx*in.ny*in.nz, 0.f); auto& load=ad.getLoadBalancer();
    for(int iC=0;iC<load.size();++iC){ auto& ab=ad.getBlock(iC);
        auto& bg=sg.getBlockGeometry(iC); const auto off=bridge::blockOffset(sg,iC);          // S1
        const int gx=off[0],gy=off[1],gz=off[2];
        const int bnx=bg.getNx(),bny=bg.getNy(),bnz=bg.getNz();
        for(int x=0;x<bnx;++x)for(int y=0;y<bny;++y)for(int z=0;z<bnz;++z){
            int X=gx+x,Y=gy+y,Z=gz+z; if(X<0||X>=in.nx||Y<0||Y>=in.ny||Z<0||Z>=in.nz)continue;
            auto c=ab.get(x,y,z);
            out[in.idx(X,Y,Z)] = which==0 ? (float)c.template getField<descriptors::THETA>()
                                          : (float)c.template getField<descriptors::DEPOSIT>(); }
    }
}
} // namespace step4
#endif // ENABLE_STEP4

// write Θ and deposition fields in the project 5-int format for Stage C (J = ⟨w,Θ⟩/|Ω|)
static void writeField5(const std::string& fn,const std::vector<float>& d,int nx,int ny,int nz,double dx){
    FILE* f=fopen(fn.c_str(),"wb"); if(!f)return; int h[5]={nx,ny,nz,(int)std::lround(dx*1000.0),1};
    fwrite(h,sizeof(int),5,f); fwrite(d.data(),sizeof(float),d.size(),f); fclose(f);
}

// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    olb::initialize(&argc, &argv);                                                   // B1
    OstreamManager clout(std::cout,"urban_flow");
    const std::string GEOM = getenv("GEOM_DIR") ? getenv("GEOM_DIR") : "geom_out";
    const std::string OUT  = getenv("OUT_DIR")  ? getenv("OUT_DIR")  : "urban_flow_out";
    { std::string c="mkdir -p '"+OUT+"'"; if(system(c.c_str())){} }

    // ── load geometry bridge outputs, carve the outlet sponge band (C5) ──
    bridge::GridField<int32_t> mat;
    if (!bridge::load_material_map(GEOM+"/material_map.dat", mat)) return 2;
    const int nx=mat.nx, ny=mat.ny, nz=mat.nz; const double dx=mat.dx;
    const double WIND_DEG = envd("WIND_DEG", 0.0);
    const int nSponge = envi("SPONGE_CELLS", 8);
    carve_sponge(mat, WIND_DEG, nSponge);
    clout << "material_map " << nx<<"x"<<ny<<"x"<<nz<<" dx="<<dx<<"  sponge="<<nSponge<<" cells" << std::endl;

    // ── unit converter (forward_city operating point) ──
    const T U_INLET = envd("U_INLET", 4.0), Z_REF = envd("ABL_ZREF", 4.0), nu_phys = 1.5e-5;
    const int RES = envi("RESOLUTION", 1);
    UnitConverter<T,DESCRIPTOR> converter(                                            // CONFIRM 1.8 ctor variant
        (T)dx/RES, /*physDeltaT*/(T)(0.05*dx/U_INLET),
        /*charL*/(T)(nz*dx), /*charU*/U_INLET, /*nu*/nu_phys, /*rho*/1.2);
    converter.print();

    // (C1) preflight — refuse to run outside the stable regime unless FORCE=1
    if (!preflight(converter) && !envi("FORCE",0)) { clout<<"aborting on preflight"<<std::endl; return 4; }

    // ── OpenLB geometry over an nx×ny×nz cuboid (1:1 with the imported map) ──
    // B2: CuboidGeometry3D was renamed CuboidDecomposition3D in 1.8.
    // Also (2.4): the IndicatorCuboid3D + spacing ctor may yield nx or nx+1 nodes per axis,
    // and everything downstream assumes exact 1:1 with the imported map. The explicit-extent
    // ctor removes that ambiguity. Overlap raised 2 -> 3: the 1.8 default, and what
    // interpolated boundaries plus WALE velocity gradients want.
    Vector<T,3> origin(0,0,0);
#ifdef PARALLEL_MODE_MPI
    const int noOfCuboids = singleton::mpi().getSize();
#else
    const int noOfCuboids = 1;
#endif
    CuboidDecomposition3D<T> cuboidDecomposition(origin, (T)dx, Vector<int,3>{nx,ny,nz}, noOfCuboids);
    HeuristicLoadBalancer<T> loadBalancer(cuboidDecomposition);
    SuperGeometry<T,3> superGeometry(cuboidDecomposition, loadBalancer, 3);
    bridge::stampSuperGeometry(superGeometry, mat);   // geometry bridge + sponge, applied
    superGeometry.getStatistics().print();

    // Gate 5: OpenLB's own per-material voxel counts must equal the Stage-A histogram
    // exactly (modulo the sponge cells carved out of FLUID). This one check catches the
    // getOrigin() unit bug, any nx-vs-nx+1 off-by-one, and any overlap indexing error.
    for (int m=0; m<=MAT_SPONGE; ++m)
      clout << "GATE5 MAT " << m << " olb=" << superGeometry.getStatistics().getNvoxel(m) << std::endl;

    // ── verified ABL/RFG inlet ──
    gInlet.z0 = envd("ABL_Z0", 0.045); gInlet.d = 0.0; gInlet.wind_angle = WIND_DEG*M_PI/180.0;
    gInlet.L_turb = envd("ABL_LTURB", 20.0); gInlet.n_modes = envi("ABL_NMODES", 100);
    gInlet.sigma_u_ratio=2.5; gInlet.sigma_v_ratio=1.9; gInlet.sigma_w_ratio=1.25;
    gInlet.init(U_INLET, Z_REF, /*seed=*/1000u);
    clout << "ABL inlet u*=" << gInlet.u_star << " (verified: mean 0.10%, div 5.2%)" << std::endl;

    // ── lattice ──
    SuperLattice<T,DESCRIPTOR> sLattice(superGeometry);                              // CONFIRM 1.8
    prepareLattice(sLattice, converter, superGeometry, nSponge);

    // ── develop the LIVE turbulent flow (no averaging) ──
    const int SPIN_FT   = envi("SPINUP_FT", 3);
    const int stepsPerFT= (int)std::llround((nx*dx)/U_INLET / converter.getPhysDeltaT());  // CONFIRM 1.8
    const int MAX_STEPS = envi("MAX_STEPS", SPIN_FT*stepsPerFT);
    const int rampSteps = stepsPerFT;                      // (C3) ramp over 1 flow-through
    const int CHECK     = envi("CHECK_EVERY", 200);        // (C4) divergence-guard cadence
    const int ADM_EVERY = envi("ADM_EVERY", 0);           // (C6) 0 = ADM off
    clout << "live spin-up: " << SPIN_FT << " flow-throughs = " << MAX_STEPS
          << " steps (" << stepsPerFT << "/FT), ramp " << rampSteps << ", check " << CHECK << std::endl;

    for (int iT=0; iT<MAX_STEPS; ++iT) {
        setBoundaryValues(sLattice, converter, superGeometry, iT, rampSteps);   // C3
        sLattice.collideAndStream();                                            // CONFIRM 1.8

        if (ADM_EVERY>0 && iT%ADM_EVERY==0) {                                   // C6 optional ADM filter
            // CONFIRM 1.8: SuperLatticeADM3D<T,DESCRIPTOR> admF(sLattice, adm_sigma, adm_order);
            //             admF.execute(superGeometry, MAT_FLUID);   // approximate deconvolution
        }
        if (iT%CHECK==0) {
            if (!healthy(sLattice)) { return 2; }                               // C4 abort on divergence
            clout << "iT=" << iT << " t=" << converter.getPhysTime(iT)
                  << "s ramp=" << bridge::AblVelocityF3D<T,DESCRIPTOR,UnitConverter<T,DESCRIPTOR>>::smoothstep(
                        rampSteps>0?(T)iT/(T)rampSteps:T(1)) << std::endl;
        }
    }
    if (!healthy(sLattice)) return 2;

    // ── snapshot the developed LIVE field for Stage C ──
    exportLiveFlow(sLattice, converter, superGeometry, nx, ny, nz, dx, OUT);

#ifdef ENABLE_STEP4
    // ════════════════════════ STEP 4: live-flow burst transport ════════════════════════
    // Continue the LIVE flow and run the accidental burst over Ω on it (no frozen mean).
    ScalarInputs sin; sin.nx=nx; sin.ny=ny; sin.nz=nz; sin.dx=dx; sin.mat=mat.data;
    { bridge::GridField<uint8_t> sm; if(!bridge::load_source_mask(GEOM+"/source_mask.u8",sm)){clout<<"no source_mask"<<std::endl;return 2;} sin.src=sm.data; }
    { bridge::GridField<float> dv; if(bridge::read_grid(GEOM+"/dep_vel.f32",dv)) sin.vd=dv.data; else sin.vd.assign((size_t)nx*ny*nz,0.f); }
    long nOmega=0; for(auto v:sin.src) if(v) ++nOmega;

    SuperLattice<T,AD_DESCRIPTOR> adLattice(superGeometry);                            // CONFIRM 1.8
    const T D_mol=envd("D_MOL",1e-5), Sc_t=envd("SC_T",0.7);
    prepareScalarLattice(adLattice, sLattice, converter, superGeometry, D_mol, Sc_t);

    const T   dt        = converter.getPhysDeltaT();                                    // CONFIRM 1.8
    const T   PULSE_S   = envd("PULSE_S", 2.0);                                         // burst duration
    const T   rate_lb   = envd("Q_RATE", 1.0);                                          // per-Ω-cell emission/step (lattice)
    const T   w_s_lb    = converter.getLatticeVelocity(envd("W_SETTLE", 0.0));          // settling (0 = gas) CONFIRM 1.8
    const T   CLEAR     = envd("CLEAR_FRAC", 0.01);                                     // stop at 99% clearance
    const int MAXB      = envi("MAX_BURST_STEPS", 20*stepsPerFT);
    const int TS_EVERY  = envi("TS_EVERY", 200);
    double emit=0, dep=0, peakAir=0; int endStep=MAXB;
    FILE* ts=fopen((OUT+"/exposure_timeseries.csv").c_str(),"w");
    if(ts) fprintf(ts,"step,t_s,airborne,deposited,emitted,outflow\n");

    for (int iB=0; iB<MAXB; ++iB) {
        setBoundaryValues(sLattice, converter, superGeometry, MAX_STEPS+iB, 0);   // inlet stays live, no ramp
        sLattice.collideAndStream();                                              // NSE (live)
        step4::couple(adLattice, sLattice, superGeometry, converter, w_s_lb);     // advect on live u −w_sẑ
        adLattice.collideAndStream();                                             // AD (CONFIRM 1.8)
        bool pulseOn = (iB*dt) < PULSE_S;
        emit += step4::injectBurst(adLattice, superGeometry, sin, rate_lb, pulseOn);
        dep  += step4::deposit(adLattice, superGeometry, sin);
        double air = step4::accumulateTheta(adLattice, superGeometry, sin, dt);
        peakAir = std::max(peakAir, air);
        if(!std::isfinite(air)){ OstreamManager c(std::cout,"DIVERGED"); c<<"AD non-finite at burst step "<<iB<<std::endl; if(ts)fclose(ts); return 2; }
        if(iB%TS_EVERY==0 && ts){ double outfl=emit-dep-air; fprintf(ts,"%d,%.4f,%.6e,%.6e,%.6e,%.6e\n",iB,iB*dt,air,dep,emit,outfl); fflush(ts);
            clout<<"burst iB="<<iB<<" t="<<iB*dt<<"s air="<<air<<" dep="<<dep<<" emit="<<emit<<std::endl; }
        if(!pulseOn && peakAir>0 && air < CLEAR*peakAir){ endStep=iB; break; }         // self-terminate at clearance
    }
    if(ts) fclose(ts);

    // export Θ = ∫C dt and the deposition map for Stage C (J = ⟨w,Θ⟩/|Ω|)
    std::vector<float> theta, deposition;
    step4::gatherField(adLattice, superGeometry, sin, 0, theta);
    step4::gatherField(adLattice, superGeometry, sin, 1, deposition);
    writeField5(OUT+"/theta.f32",      theta,      nx,ny,nz,dx);
    writeField5(OUT+"/deposition.f32", deposition, nx,ny,nz,dx);

    double outflow = emit - dep;   // drained by advective outflow (remainder still airborne→Θ)
    { FILE* mf=fopen((OUT+"/meta_flow.txt").c_str(),"w"); if(mf){
        fprintf(mf,"grid %d %d %d\ndx_m %.4f\nomega_cells %ld\n",nx,ny,nz,dx,nOmega);
        fprintf(mf,"burst_steps %d\nmass_emitted %.6e\nmass_deposited %.6e\nmass_drained %.6e\n",endStep,emit,dep,outflow);
        fprintf(mf,"deposited_frac %.4f\ntheta_layout 5xint32[nx,ny,nz,dx*1000,1]\n", emit>0?dep/emit:0.0);
        fprintf(mf,"# Stage C: J = (1/omega_cells) * sum_x receptor_w(x) * theta(x)\n"); fclose(mf);} }

    clout << "urban_flow COMPLETE — live airflow + burst transport. Wrote umean_full.f32, "
          << "theta.f32, deposition.f32, exposure_timeseries.csv to " << OUT << "/  (Omega="
          << nOmega << ", emitted=" << emit << ", deposited=" << dep << ")" << std::endl;
#else
    clout << "urban_flow COMPLETE (airflow only; Step 4 off — rebuild with -DENABLE_STEP4). "
          << "Wrote umean_full.f32 to " << OUT << "/" << std::endl;
#endif // ENABLE_STEP4
    return 0;
}
