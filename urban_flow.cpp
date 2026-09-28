// urban_flow.cpp — STAGE B (airflow, LIVE flow): OpenLB 1.8 NSE + WALE LES engine for the
// urban_openlbm migration. Consumes the geometry-bridge outputs (material_map.dat +
// source_mask.u8), develops a LIVE turbulent wind field on the city, and hands it to the
// scalar burst (Step 4). No time-averaging — this is the FORWARD_LIVE regime: the burst
// rides the instantaneous, evolving flow (matching forward_city.cpp), so the airflow stage
// only needs to reach a statistically-developed turbulent state, then snapshot it.
//
// ┌─ SCOPE ──────────────────────────────────────────────────────────────────────┐
// │ Steps 3-4 of OPENLB_MIGRATION_PLAN.md §7: D3Q19 + WALE LES, ABL/RFG velocity   │
// │ inlet, pressure outlet, free-slip sides + stressed top, rough-wall floor,     │
// │ developed LIVE (no averaging); STEP4=1 adds the D3Q7 burst + deposition.       │
// └──────────────────────────────────────────────────────────────────────────────┘
//
// CORRECTIVE / STABILISATION METHODS wired here (the numerical safety net the custom
// engine had — see TECHNICAL_STATUS notes on ABL drift and low-Re stability):
//   (C1) Mach / CFL / τ preflight — refuse to run outside the stable low-Ma regime.
//   (C2) Rough-wall floor on MAT_GROUND (z0 log-law stress, RoughWall) + Richards-Hoxey
//        shear stress at the top (TopStress) — preserve ABL horizontal homogeneity (Gate
//        6a); buildings stay smooth no-slip bounce-back (COST 732).
//   (C3) Inlet startup RAMP (smoothstep over 1 flow-through) — no pressure shock.
//   (C4) DIVERGENCE / NaN guard — abort cleanly on non-finite or super-Mach state
//        (mirrors forward_city.cpp's guard) instead of silently producing garbage.
//   (C5) Outlet SPONGE layer — a graded high-viscosity fringe before the outlet damps
//        turbulent structures so they don't reflect off the pressure boundary.
//   (C6) LES stabilisation — selectable collision (WALE default / consistent-Smagorinsky
//        / regularized) + optional ADM (approximate-deconvolution) filtering.
//
// VERIFICATION STATUS: compiles and runs against OpenLB 1.8.1 (CPU_SISD, serial and OMP).
// Phase-5/6 gate results are recorded in OPENLB_PHASE5_6_GATES.md. The "CONFIRM 1.8" markers
// predate the first compile; the calls they flag are now exercised by those runs.
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
#include <array>

using namespace olb;
using namespace olb::descriptors;
typedef double T;

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

static double envd(const char* k,double d){const char* e=getenv(k);return e?atof(e):d;}
static int    envi(const char* k,int d){const char* e=getenv(k);return e?atoi(e):d;}

// Material numbers — MUST match openlb_geometry.h (note MAT_GROUND split from MAT_WALL,
// and MAT_SPONGE is a SOLVER-LOCAL material carved out of MAT_FLUID near the outlet (C5)).
enum { MAT_VOID=0, MAT_FLUID=1, MAT_WALL=2, MAT_INLET=3, MAT_OUTLET=4,
       MAT_SLIP=5, MAT_POROUS=6, MAT_GROUND=7, MAT_SPONGE=8, MAT_FRAME=9 };

static abl::ABLInlet gInlet;   // built in main, referenced in setBoundaryValues

// ─────────────────────────────────────────────────────────────────────────────
// (C5) Carve a sponge band out of the fluid cells adjacent to the outlet face, on the
// HOST material array before stamping. Wind-aligned: default +x → band at x∈[nx-n,nx),
// eroded by one cell so it never touches a boundary plane (see the note in the body).
// Solver-local (MAT_SPONGE); the geometry bridge stays pure geometry.
// ─────────────────────────────────────────────────────────────────────────────
static void carve_sponge(bridge::GridField<int32_t>& m, double wind_deg, int nSponge) {
    if (nSponge <= 0) return;
    const int nx=m.nx, ny=m.ny, nz=m.nz;
    double wd = std::fmod(wind_deg,360.0); if (wd<0) wd+=360.0;
    static const int dn[6][3]={{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1}};
    // A cell joins the sponge only if IT AND ALL SIX NEIGHBOURS are MAT_FLUID, which keeps
    // MAT_SPONGE one cell clear of every domain boundary plane. OpenLB's
    // BlockGeometryStatistics3D::getType needs a boundary cell's inward neighbour to be
    // MAT_FLUID (1) specifically -- bulk dynamics on some other material is not enough -- so a
    // sponge cell abutting the outlet or a lateral slip face makes that boundary unsettable
    // ("Could not set Boundary"). See OPENLB_MIGRATION_PLAN.md 6.4.
    //
    // Written as an erosion rather than index arithmetic because the arithmetic version was
    // wrong twice: the outlet plane occupies x=nx-1, so a band ending at nx-buf still reaches
    // it, and the lateral slip faces need the same clearance in y and z. The erosion is
    // correct for any wind direction and any face layout without special cases.
    auto eligible=[&](int x,int y,int z)->bool{
        if (m.data[m.idx(x,y,z)] != MAT_FLUID) return false;
        for (int d=0; d<6; ++d) {
            const int xx=x+dn[d][0], yy=y+dn[d][1], zz=z+dn[d][2];
            if (xx<0||xx>=nx||yy<0||yy>=ny||zz<0||zz>=nz) return false;
            if (m.data[m.idx(xx,yy,zz)] != MAT_FLUID) return false;
        }
        return true;
    };
    // Two passes: eligibility is judged against the ORIGINAL map, so an earlier mark in the
    // same band cannot disqualify its neighbour.
    std::vector<size_t> pick;
    auto mark=[&](int x,int y,int z){ if (eligible(x,y,z)) pick.push_back(m.idx(x,y,z)); };
    if (wd < 45 || wd >= 315) for(int z=0;z<nz;++z)for(int y=0;y<ny;++y)for(int x=nx-nSponge;x<nx;++x)  mark(x,y,z); // +x
    else if (wd < 135)        for(int z=0;z<nz;++z)for(int x=0;x<nx;++x)for(int y=ny-nSponge;y<ny;++y)  mark(x,y,z); // +y
    else if (wd < 225)        for(int z=0;z<nz;++z)for(int y=0;y<ny;++y)for(int x=0;x<nSponge;++x)      mark(x,y,z); // -x
    else                      for(int z=0;z<nz;++z)for(int x=0;x<nx;++x)for(int y=0;y<nSponge;++y)      mark(x,y,z); // -y
    for (size_t k : pick) m.data[k] = MAT_SPONGE;
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
    // tau > 0.5 alone is a false-negative sieve: the molecular-viscosity operating point
    // sat at tau = 0.5000001 and passed it by 1.4e-7 while carrying effectively zero
    // viscosity. BGK/WALE with no resolved SGS contribution (the flow is at rest when the
    // inlet ramp starts, and WALE's nu_t vanishes in pure shear) needs a real floor.
    need(tau >= 0.505,              "relaxation tau >= 0.505 (non-vanishing base viscosity)");
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
    // 1.8 dynamics read OMEGA and the LES constant from LATTICE PARAMETERS, not from
    // constructor arguments. Without these two calls both default to 0: omega=0 is tau=inf
    // and the first collide writes NaN into every bulk cell (observed: 11 390 NaN fluid
    // cells after one step at rest, before this fix).

    sLattice.defineDynamics<NoDynamics>(superGeometry, MAT_VOID);                    // B3

    // bulk fluid + inlet + outlet carry the selected LES dynamics (C6)
    auto bulkInd = superGeometry.getMaterialIndicator({MAT_FLUID, MAT_INLET, MAT_OUTLET});
    sLattice.template defineDynamics<BULK_DYNAMICS<T,DESCRIPTOR>>(bulkInd);          // CONFIRM 1.8

    // buildings: smooth no-slip bounce-back
    sLattice.defineDynamics<BounceBack>(superGeometry, MAT_WALL);                    // B3
    // Domain box edges/corners (OPENLB_MIGRATION_PLAN.md 6.4). These touch no fluid cell,
    // so bounce-back on them is inert; they exist to be neither fluid nor material 0.
    sLattice.defineDynamics<BounceBack>(superGeometry, MAT_FRAME);

    // (C2) GROUND: bounce-back here keeps the floor impermeable and mass-conserving; the
    // rough-wall behaviour (z0 log law) is layered on top by RoughWall after each stream,
    // since OpenLB 1.8 has no z0 wall function (G1). GROUND_MODEL=0 leaves plain no-slip
    // bounce-back, which Gate 6a measured at 42.7 % near-ground drift by the city face.
    sLattice.defineDynamics<BounceBack>(superGeometry, MAT_GROUND);                  // B3+B8

    // porous park canopy (PorousBGK — set porosity field to the calibrated C_d/LAD)
    sLattice.template defineDynamics<PorousBGKdynamics<T,DESCRIPTOR>>(
        superGeometry.getMaterialIndicator({MAT_POROUS}));                           // CONFIRM 1.8
    // POROSITY defaults to 0 = fully solid, so without this every park would be a building.
    // 0.8 matches voxelize.h's PERM_PARK; the C_d/LAD calibration (PARK_POROSITY.md) is open.
    AnalyticalConst3D<T,T> por(envd("PARK_POROSITY", 0.8));
    sLattice.defineField<POROSITY>(superGeometry.getMaterialIndicator({MAT_POROUS}), por);

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
    // Parameters are set AFTER every defineDynamics / boundary::set, as OpenLB's own examples
    // do: boundary::set installs its own BGK mixin dynamics on MAT_INLET/MAT_OUTLET, and a
    // parameter set before a dynamics is registered does not reach it.
    const T omega = converter.getLatticeRelaxationFrequency();
    sLattice.setParameter<descriptors::OMEGA>(omega);
    // One lattice-global constant (G3), read by the bulk model AND the sponge's Smagorinsky.
    // WALE uses it as C_w (collisionLES.h: preFactor = C^2); 0.325 is the project's value
    // (WALE_MODEL.md). For consistent-strain Smagorinsky OpenLB's tgv3d uses 0.033.
    const T lesConst = envd("LES_CONST", COLLISION_MODEL==0 ? 0.325 : COLLISION_MODEL==1 ? 0.033 : 0.0);
    sLattice.setParameter<collision::LES::SMAGORINSKY>(lesConst);
    clout << "omega=" << omega << "  LES constant=" << lesConst << std::endl;

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
//
// The max |u| is taken by scanning the cells directly, NOT from
// sLattice.getStatistics().getMaxU(), for two reasons:
//  (1) LatticeStatistics keeps its running max with `if (uSqr > tmpMax)`, which is false
//      for NaN, so a NaN cell never registers and a statistics-based guard cannot see
//      exactly the failure it exists to catch.
//  (2) Under this build it returned sqrt(numeric_limits<double>::min()) = 1.49e-154 on a
//      flow with |u|_lb ~ 1e-4 -- i.e. no cell contributed at all, so the guard was blind.
// A scan costs about one collide sweep, and it runs every CHECK_EVERY steps. On GPU it needs
// the Evaluation processing context first (Phase 8).
// ─────────────────────────────────────────────────────────────────────────────
static T scanMaxU(SuperLattice<T,DESCRIPTOR>& sLattice, SuperGeometry<T,3>& superGeometry, bool& finite) {
    T maxSq = 0; bool fin = true;
    auto& load = sLattice.getLoadBalancer();
    for (int iC=0; iC<load.size(); ++iC) {
        auto& block = sLattice.getBlock(iC);
        auto& bgeo  = superGeometry.getBlockGeometry(iC);
        const int bnx=bgeo.getNx(), bny=bgeo.getNy(), bnz=bgeo.getNz();
        #ifdef PARALLEL_MODE_OMP
        #pragma omp parallel for schedule(static) reduction(max:maxSq) reduction(&&:fin)
        #endif
        for (int x=0;x<bnx;++x) for(int y=0;y<bny;++y) for(int z=0;z<bnz;++z) {
            const int m = bgeo.get({x,y,z});
            if (m==MAT_VOID||m==MAT_WALL||m==MAT_GROUND||m==MAT_FRAME) continue;
            T u[3]; block.get(x,y,z).computeU(u);
            const T q = u[0]*u[0]+u[1]*u[1]+u[2]*u[2];
            if (!std::isfinite(q)) fin = false; else if (q > maxSq) maxSq = q;
        }
    }
#ifdef PARALLEL_MODE_MPI
    singleton::mpi().reduceAndBcast(maxSq, MPI_MAX);
    int f = fin ? 1 : 0; singleton::mpi().reduceAndBcast(f, MPI_MIN); fin = f==1;
#endif
    finite = fin;
    return std::sqrt(maxSq);
}

static bool healthy(SuperLattice<T,DESCRIPTOR>& sLattice, SuperGeometry<T,3>& superGeometry, T* maxUout=nullptr) {
    bool finite = true;
    const T maxU = scanMaxU(sLattice, superGeometry, finite);
    if (maxUout) *maxUout = maxU;
    if (!finite || maxU > 0.4) {                                // >0.4 LU ⇒ super-Mach / blow-up
        OstreamManager clout(std::cout,"DIVERGED");
        if (!finite) {   // name the materials holding non-finite cells -- the first thing to ask
            long bad[16] = {0}; int ex[16][3];
            auto& load = sLattice.getLoadBalancer();
            for (int iC=0; iC<load.size(); ++iC) {
                auto& block = sLattice.getBlock(iC); auto& bgeo = superGeometry.getBlockGeometry(iC);
                const auto off = bridge::blockOffset(superGeometry, iC);
                for (int x=0;x<bgeo.getNx();++x) for(int y=0;y<bgeo.getNy();++y) for(int z=0;z<bgeo.getNz();++z) {
                    const int m = bgeo.get({x,y,z}); if (m<0||m>15) continue;
                    T u[3]; block.get(x,y,z).computeU(u);
                    if (!std::isfinite(u[0]*u[0]+u[1]*u[1]+u[2]*u[2])) {
                        if (!bad[m]) { ex[m][0]=off[0]+x; ex[m][1]=off[1]+y; ex[m][2]=off[2]+z; }
                        ++bad[m];
                    }
                }
            }
            for (int m=0;m<16;++m) if (bad[m]) clout << "  MAT " << m << ": " << bad[m]
                << " non-finite cells, e.g. (" << ex[m][0] << "," << ex[m][1] << "," << ex[m][2] << ")" << std::endl;
        }
        clout << "*** FATAL: non-finite or super-Mach state (maxU=" << maxU
              << (finite ? "" : ", non-finite cells present")
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

// ─────────────────────────────────────────────────────────────────────────────
// (C2 / G1) RoughWall — aerodynamically rough floor, applied to the first fluid layer above
// MAT_GROUND after every collideAndStream. OpenLB 1.8 has no z0 wall function (G1), and
// Gate 6a measured what the bounce-back floor it falls back to does at dx = 4 m: the mean
// wind at the first cell drops from 3.9 to 1.3 m/s by the city face (42.7 % drift) and to
// 0.7 m/s by mid-fetch. A no-slip wall resolved by one 4 m cell imposes a laminar stress
// ~nu*U1/(dx/2), an order of magnitude above the rough-wall stress rho*u*^2.
//
// The operator replaces that stress with the log-law one, in two mass-conserving steps:
//  (1) SPECULAR FLOOR. OpenLB's BounceBack is full-way: the population a first-layer cell P
//      receives in direction (cx,cy,+1) is P's own (-cx,-cy,-1) from the previous step.
//      Specular reflection instead delivers the (cx,cy,-1) population that P' = P-2(cx,cy)
//      sent, which after the stream sits in P' as its BB-returned (-cx,-cy,+1). So every
//      up-going slot is refilled from its specular partner. Where P' or the cell between
//      is not a layer cell (a building, a boundary face, a block edge) the slot keeps its
//      BB value. The partner relation is symmetric, so the remap is a bijection on the
//      layer's up-going values: mass is conserved exactly and the normal flux stays zero.
//      The floor now exerts no tangential stress of its own.
//  (2) LOG-LAW STRESS. Tangential momentum is removed at the rate rho*u*^2 per unit area,
//      u* = kappa*|U_t(P)| / ln(z_P/z0) from the instantaneous local velocity (Schumann-
//      Groetzbach type), through the exact-difference forcing f_i += feq(rho,u+du)-feq(rho,u),
//      du = -u*^2 * dt/dz along U_t (one cell deep). Clamped so it can slow but not reverse.
// Heights follow the INLET's convention (abl_inlet.h: ground-cell centre is z = 0, so the
// first fluid cell is at z_P = dx), so the floor and the inlet describe the same log law;
// the half-cell ambiguity of where a bounce-back wall "is" is left to a sensitivity run.
// Buildings stay smooth no-slip bounce-back (COST 732) -- only cells over MAT_GROUND get
// this. Parks (MAT_POROUS) are excluded: their canopy drag is the porous model's job.
// Host-side OpenMP loop over ~ nx*ny cells; for GPU it becomes a post-processor (Phase 8).
// ─────────────────────────────────────────────────────────────────────────────
struct RoughWall {
    T z0 = 0.045, zP = 4.0, kappa = 0.41;
    struct Layer {
        std::vector<int> x, y;            // block-local coordinates, z = zLayer
        std::vector<int> partner;         // [4*k + d]: index of P' for dir d, or -1 (keep BB)
        std::vector<T>   save;            // [4*k + d]: snapshot of the up-going slot values
        int z = 1;
    };
    std::vector<Layer> layers;
    int up[4] = {-1,-1,-1,-1}, upOpp[4] = {-1,-1,-1,-1};   // (c_h,+1) and (-c_h,+1) indices
    int dh[4][2] = {{1,0},{-1,0},{0,1},{0,-1}};
    long nCells = 0, nSpecular = 0;

    static bool layerMat(int m) { return m==MAT_FLUID || m==MAT_SPONGE; }

    void init(SuperGeometry<T,3>& sg, T z0_, T zP_) {
        z0 = z0_; zP = zP_;
        for (int d=0; d<4; ++d) for (int i=0; i<DESCRIPTOR::q; ++i) {
            auto c = descriptors::c<DESCRIPTOR>(i);
            if (c[0]==dh[d][0] && c[1]==dh[d][1] && c[2]==1)   up[d]    = i;
            if (c[0]==-dh[d][0] && c[1]==-dh[d][1] && c[2]==1) upOpp[d] = i;
        }
        auto& load = sg.getLoadBalancer();
        layers.assign(load.size(), Layer{});
        for (int iC=0; iC<load.size(); ++iC) {
            auto& bg = sg.getBlockGeometry(iC); auto& L = layers[iC];
            const int nx=bg.getNx(), ny=bg.getNy(), nz=bg.getNz();
            std::vector<int> id((size_t)nx*ny, -1);
            // the layer: cells carrying bulk fluid dynamics with the ground directly below
            for (int z=1; z<nz; ++z) {
                for (int y=0;y<ny;++y) for (int x=0;x<nx;++x)
                    if (layerMat(bg.get({x,y,z})) && bg.get({x,y,z-1})==MAT_GROUND) {
                        id[(size_t)y*nx+x] = (int)L.x.size(); L.x.push_back(x); L.y.push_back(y);
                    }
                if (!L.x.empty()) { L.z = z; break; }   // flat ground: one layer height
            }
            auto at=[&](int x,int y){ return (x<0||y<0||x>=nx||y>=ny) ? -1 : id[(size_t)y*nx+x]; };
            L.partner.assign(4*L.x.size(), -1); L.save.assign(4*L.x.size(), 0);
            for (size_t k=0; k<L.x.size(); ++k) for (int d=0; d<4; ++d) {
                const int mid = at(L.x[k]-dh[d][0],   L.y[k]-dh[d][1]);
                const int src = at(L.x[k]-2*dh[d][0], L.y[k]-2*dh[d][1]);
                if (mid>=0 && src>=0) { L.partner[4*k+d] = src; ++nSpecular; }
            }
            nCells += (long)L.x.size();
        }
    }

    void apply(SuperLattice<T,DESCRIPTOR>& sLattice) {
        const T lnz = std::log(zP/z0);
        for (size_t iC=0; iC<layers.size(); ++iC) {
            auto& block = sLattice.getBlock(iC); auto& L = layers[iC];
            const long n = (long)L.x.size();
            // (1a) snapshot the BB-returned up-going values the remap reads from
            #ifdef PARALLEL_MODE_OMP
            #pragma omp parallel for schedule(static)
            #endif
            for (long k=0; k<n; ++k) {
                auto cell = block.get(L.x[k], L.y[k], L.z);
                for (int d=0; d<4; ++d) L.save[4*k+d] = cell[upOpp[d]];
            }
            #ifdef PARALLEL_MODE_OMP
            #pragma omp parallel for schedule(static)
            #endif
            for (long k=0; k<n; ++k) {
                auto cell = block.get(L.x[k], L.y[k], L.z);
                // (1b) specular remap: slot (c_h,+1) <- partner's BB-returned (-c_h,+1)
                for (int d=0; d<4; ++d) {
                    const int p = L.partner[4*k+d];
                    if (p >= 0) cell[up[d]] = L.save[4*p+d];
                }
                // (2) log-law wall stress by exact-difference forcing
                T rho, u[3]; cell.computeRhoU(rho, u);
                const T ut = std::sqrt(u[0]*u[0] + u[1]*u[1]);
                if (!(ut > 0)) continue;
                const T us = kappa*ut/lnz;
                const T du = std::min(us*us, ut);          // lattice units, dz = 1 cell
                T u2[3] = { u[0] - du*u[0]/ut, u[1] - du*u[1]/ut, u[2] };
                for (int i=0; i<DESCRIPTOR::q; ++i)
                    cell[i] += equilibrium<DESCRIPTOR>::secondOrder(i, rho, u2)
                             - equilibrium<DESCRIPTOR>::secondOrder(i, rho, u);
            }
        }
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// TopStress — the Richards & Hoxey (1993) top condition for a neutral ABL. A log-law profile
// carries the same shear stress rho*u*^2 at every height, delivered to the floor from above.
// A free-slip lid admits none, so with the rough wall taking u*^2 out at the bottom the column
// decelerates from the top down: Gate 6a measured -8..-11 % at the cell under the lid with
// the bounce-back floor AND with the rough wall (identical, so it is not a floor effect).
// This adds the stress back: +u*^2 of streamwise momentum per step to the fluid layer under
// the top slip face, by the same exact-difference forcing as RoughWall. u* is the inlet's
// (the profile being maintained), scaled by the inlet ramp squared during start-up.
// ─────────────────────────────────────────────────────────────────────────────
struct TopStress {
    std::vector<std::vector<std::array<int,3>>> cells;
    long nCells = 0;
    T dirX = 1, dirY = 0;
    void init(SuperGeometry<T,3>& sg, int nzGlobal, T windRad) {
        dirX = std::cos(windRad); dirY = std::sin(windRad);
        auto& load = sg.getLoadBalancer(); cells.assign(load.size(), {});
        for (int iC=0; iC<load.size(); ++iC) {
            auto& bg = sg.getBlockGeometry(iC); const auto off = bridge::blockOffset(sg, iC);
            for (int x=0;x<bg.getNx();++x) for (int y=0;y<bg.getNy();++y) for (int z=0;z<bg.getNz();++z) {
                const int m = bg.get({x,y,z});
                if ((m==MAT_FLUID||m==MAT_SPONGE) && off[2]+z == nzGlobal-2 && bg.get({x,y,z+1})==MAT_SLIP)
                    cells[iC].push_back({x,y,z});
            }
            nCells += (long)cells[iC].size();
        }
    }
    void apply(SuperLattice<T,DESCRIPTOR>& sLattice, T ustarLB) {
        const T du = ustarLB*ustarLB;
        for (size_t iC=0; iC<cells.size(); ++iC) {
            auto& block = sLattice.getBlock(iC); auto& L = cells[iC];
            const long n = (long)L.size();
            #ifdef PARALLEL_MODE_OMP
            #pragma omp parallel for schedule(static)
            #endif
            for (long k=0; k<n; ++k) {
                auto cell = block.get(L[k][0], L[k][1], L[k][2]);
                T rho, u[3]; cell.computeRhoU(rho, u);
                T u2[3] = { u[0] + du*dirX, u[1] + du*dirY, u[2] };
                for (int i=0; i<DESCRIPTOR::q; ++i)
                    cell[i] += equilibrium<DESCRIPTOR>::secondOrder(i, rho, u2)
                             - equilibrium<DESCRIPTOR>::secondOrder(i, rho, u);
            }
        }
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// WALE velocity-gradient refresh (COLLISION_MODEL 0).
//
// OpenLB 1.8's WALE collision does not differentiate the velocity itself: it reads the
// VELO_GRAD field (collisionLES.h, WaleEffectiveOmega), which the APPLICATION must refresh
// before every collide -- examples/turbulence/tgv3d does it with
// defineField<VELO_GRAD>(SuperLatticeVelocityGradientFD3D). Never refreshed, VELO_GRAD stays
// 0, WALE's nu_t is 0 everywhere and the "LES" is silently plain BGK at the base viscosity.
//
// The stock functor heap-allocates std::vectors per cell and runs serially: measured
// < 1.5 MLUPS on the 630k-cell Gate-6a box against 24 MLUPS without it. This is the same
// computation -- lattice-unit gradient, VELO_GRAD[3i+j] = du_i/dx_j (turbulentF3D.hh) --
// as two OpenMP sweeps: velocity into a padded buffer, then differences. The stencil is the
// stock one where it can be: 8th-order central (latticeDerivatives3D.hh, 672/168/32/3 over
// 840) wherever the +-4 cells all carry fluid, so in the bulk this IS OpenLB's WALE input;
// nearer a wall 2nd-order central, then 1st-order one-sided, zero with no fluid neighbour.
// (The stock functor checks only the +-1 materials before taking +-4 differences, i.e. it
// differentiates across thin walls; this version does not.) Written on MAT_FLUID only: inlet/outlet run the boundary's BGK mixin and the
// sponge/porous cells their own dynamics, none of which read VELO_GRAD.
// WALE_GRAD_CHECK=1 compares it once against the stock functor (see main).
// Neighbour reads use the block's overlap layer, so under MPI the halo must be current
// (it is after collideAndStream's communication). On GPU this is a host sweep -- Phase 8
// must move it on-device alongside G2.
// ─────────────────────────────────────────────────────────────────────────────
struct VeloGradRefresh {
    std::vector<std::vector<T>>       ubuf;
    std::vector<std::vector<uint8_t>> fl;
    static bool carriesFluid(int m) {
        return m==MAT_FLUID||m==MAT_INLET||m==MAT_OUTLET||m==MAT_POROUS||m==MAT_SPONGE;
    }
    void operator()(SuperLattice<T,DESCRIPTOR>& sLattice, SuperGeometry<T,3>& superGeometry) {
        auto& load = sLattice.getLoadBalancer();
        ubuf.resize(load.size()); fl.resize(load.size());
        for (int iC=0; iC<load.size(); ++iC) {
            auto& block = sLattice.getBlock(iC);
            auto& bgeo  = superGeometry.getBlockGeometry(iC);
            const int nx=bgeo.getNx(), ny=bgeo.getNy(), nz=bgeo.getNz();
            constexpr int PAD = 1;                             // buffer padding (<= overlap)
            const int px=nx+2*PAD, py=ny+2*PAD, pz=nz+2*PAD;
            auto& u = ubuf[iC]; auto& f = fl[iC];
            u.resize(3*(size_t)px*py*pz); f.resize((size_t)px*py*pz);
            auto P=[&](int x,int y,int z){ return ((size_t)(x+PAD)*py+(y+PAD))*pz+(z+PAD); };
            #ifdef PARALLEL_MODE_OMP
            #pragma omp parallel for schedule(static)
            #endif
            for (int x=-PAD;x<nx+PAD;++x) for (int y=-PAD;y<ny+PAD;++y) for (int z=-PAD;z<nz+PAD;++z) {
                const size_t k=P(x,y,z);
                const int m = bgeo.get({x,y,z});
                f[k] = carriesFluid(m);
                T uu[3]={0,0,0};
                if (f[k]) block.get(x,y,z).computeU(uu);
                u[3*k]=uu[0]; u[3*k+1]=uu[1]; u[3*k+2]=uu[2];
            }
            #ifdef PARALLEL_MODE_OMP
            #pragma omp parallel for schedule(static)
            #endif
            for (int x=0;x<nx;++x) for (int y=0;y<ny;++y) for (int z=0;z<nz;++z) {
                if (bgeo.get({x,y,z}) != MAT_FLUID) continue;
                const size_t k=P(x,y,z);
                const int c[3]={x,y,z}, n[3]={nx,ny,nz};
                Vector<T,9> g;
                for (int j=0;j<3;++j) {                         // derivative direction
                    size_t kp[5], kn[5];                        // kp[s] = +s cells along j
                    bool wide = c[j]-4 >= -PAD && c[j]+4 < n[j]+PAD;
                    for (int st=1; st<=4; ++st) {
                        int a[3]={x,y,z}, b[3]={x,y,z}; a[j]+=st; b[j]-=st;
                        if (st>1 && !wide) break;
                        kp[st]=P(a[0],a[1],a[2]); kn[st]=P(b[0],b[1],b[2]);
                        if (!f[kp[st]] || !f[kn[st]]) wide=false;
                    }
                    const bool fp=f[kp[1]], fn=f[kn[1]];
                    for (int i=0;i<3;++i) {                     // velocity component
                        auto U=[&](size_t kk){ return u[3*kk+i]; };
                        T d = 0;
                        if (wide)
                            d = (T(672)*(U(kp[1])-U(kn[1])) + T(168)*(U(kn[2])-U(kp[2]))
                               + T(32)*(U(kp[3])-U(kn[3])) + T(3)*(U(kn[4])-U(kp[4]))) / T(840);
                        else if (fp && fn) d = T(0.5)*(U(kp[1])-U(kn[1]));
                        else if (fp)       d = U(kp[1])-U(k);
                        else if (fn)       d = U(k)-U(kn[1]);
                        g[3*i+j] = d;
                    }
                }
                block.get(x,y,z).template setField<descriptors::VELO_GRAD>(g);
            }
        }
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// TimeMean — running mean of the resolved velocity, for the Phase-5 airflow gates only.
// The production path stays LIVE (no averaging); this exists because Gate 6a (ABL drift)
// and Gate 6b (cube reattachment Xr/H) are statements about the MEAN flow, and an
// instantaneous snapshot of a turbulent field cannot answer either (HANDOFF.md: the old
// solver's "44-70 % drift" was exactly such a snapshot artefact). Enabled with AVG_FT>0;
// written as uavg.f32 in the 5-int layout, ncomp=6: mean (ux,uy,uz) then variance
// (ux'^2,uy'^2,uz'^2), m/s and m^2/s^2 -- the variances give the turbulence intensity that
// Gate 6b's reattachment length depends on.
// ─────────────────────────────────────────────────────────────────────────────
struct TimeMean {
    int nx=0,ny=0,nz=0; long nSamples=0;
    std::vector<double> sum;
    void init(int X,int Y,int Z){ nx=X;ny=Y;nz=Z; sum.assign(6*(size_t)nx*ny*nz,0.0); nSamples=0; }
    void sample(SuperLattice<T,DESCRIPTOR>& sLattice, SuperGeometry<T,3>& superGeometry, T cv) {
        const size_t N=(size_t)nx*ny*nz;
        auto& load = sLattice.getLoadBalancer();
        for (int iC=0; iC<load.size(); ++iC) {
            auto& block = sLattice.getBlock(iC);
            auto& bgeo  = superGeometry.getBlockGeometry(iC);
            const auto off=bridge::blockOffset(superGeometry,iC);                     // S1
            const int bnx=bgeo.getNx(), bny=bgeo.getNy(), bnz=bgeo.getNz();
            #ifdef PARALLEL_MODE_OMP
            #pragma omp parallel for schedule(static)
            #endif
            for (int x=0;x<bnx;++x) for(int y=0;y<bny;++y) for(int z=0;z<bnz;++z) {
                const int X=off[0]+x, Y=off[1]+y, Z=off[2]+z;
                if (X<0||X>=nx||Y<0||Y>=ny||Z<0||Z>=nz) continue;
                T u[3]={0,0,0};
                block.get(x,y,z).computeU(u);
                const size_t id=(size_t)Z*ny*nx+(size_t)Y*nx+X;
                for (int c=0;c<3;++c) { const double v=u[c]*cv; sum[c*N+id]+=v; sum[(3+c)*N+id]+=v*v; }
            }
        }
        ++nSamples;
    }
    void write(const std::string& fn, double dx) const {
        if (nSamples==0) return;
        const size_t N=(size_t)nx*ny*nz;
        std::vector<float> f(sum.size());
        for (size_t i=0;i<3*N;++i) f[i]=(float)(sum[i]/nSamples);
        for (size_t i=3*N;i<6*N;++i) { const double m=sum[i-3*N]/nSamples; f[i]=(float)std::max(0.0, sum[i]/nSamples - m*m); }
        FILE* fp=fopen(fn.c_str(),"wb"); if(!fp) return;
        int h[5]={nx,ny,nz,(int)std::lround(dx*1000.0),6};
        fwrite(h,sizeof(int),5,fp); fwrite(f.data(),sizeof(float),f.size(),fp); fclose(fp);
    }
};

// ══════════════════ STEP 4: advection–diffusion transport + deposition ══════════════════
// Physics (CONTAMINANT_BC.md §1): the AD scalar rides the LIVE NSE flow (not a frozen mean);
// solid walls carry zero advective flux + a dry-deposition sink; the inlet carries clean air
// (Dirichlet C = 0); the outlet is zero-gradient outflow; top/lateral are zero-flux; Ω emits
// an accidental burst over a ~2 s pulse. Gravitational settling w_s adds a −w_s ẑ advection
// offset. Θ = ∫C dt converges and the burst self-terminates at ~99 % clearance.
//
// LINEARITY (§6.2): the AD lattice + Eulerian deposition are linear, so J(Σsᵢ)=ΣJ(sᵢ). Gate 7b
// (tests/linearity_guard.cpp) checks Θ_{a+b} = Θ_a + Θ_b through SRC_CELLS.
//
// Runtime-enabled (STEP4=1). Host operators are OpenMP block loops (G2): fine for the 40³
// gate box, a full-domain sweep per step on the city, and on GPU they must become
// post-processors (Phase 8). The NSE→AD velocity copy is OpenLB's own coupling operator.
//
// Units. C is the AD lattice density (concentration per cell, lattice units), so mass = Σ C.
// S2: deposition removes the fraction v_d·dt/dx of C per step per deposition face (v_d in
// m/s from dep_vel.f32 → lattice velocity). S3: the AD relaxation is derived from
// D_eff = D_MOL + NU_EFF/SC_T and APPLIED (lattice-constant; per-cell ν_t coupling is later).
//
// Mass budget (Gate 7a) is measured, not inferred: over the control volume x ∈ [1, nx−3]
// (emission at Ω cells outside it is reported separately and kept out of the budget)
// the outflow is the exact lattice flux across its two x-faces, read from the post-stream
// ±x populations; airborne mass sums every CV cell, including mass in transit inside
// bounce-back cells. Every cell and the padding start at C = 0 (see prepare), so no baseline
// is needed; the pre-release CV mass is reported as a check. Emitted = dep + air + out.

#define AD_DESCRIPTOR D3Q7<VELOCITY>
// AD collision. BGK at tau_AD ~ 0.52 and cell Peclet ~10 rings: on the Phase-6 box ~1 % of the
// released mass sits in negative-C undershoots (RLB: 0.9 %). -DAD_TRT_MAGIC=<Lambda> selects
// TRT instead. OpenLB's collision::TRT relaxes the EVEN part with OMEGA and derives the odd rate
// from MAGIC; advection-diffusion takes its diffusivity from the ODD rate, so OMEGA must carry
// tau+ = 1/2 + Lambda/(tau_D - 1/2), not the diffusion tau (step4::prepare does this). The
// boundary mixins read the same lattice-global OMEGA, so under TRT they are TRT as well.
#ifdef AD_TRT_MAGIC
  #define AD_DYNAMICS AdvectionDiffusionTRTdynamics
  #define AD_BC_MIXIN AdvectionDiffusionTRTdynamics<T,AD_DESCRIPTOR>
#else
  #ifndef AD_DYNAMICS
  #define AD_DYNAMICS AdvectionDiffusionBGKdynamics
  #endif
  #define AD_BC_MIXIN AdvectionDiffusionRLBdynamics<T,AD_DESCRIPTOR>
#endif

namespace step4 {
using ADLat = SuperLattice<T,AD_DESCRIPTOR>;

// AD dynamics + BCs. OMEGA is set LAST: the Dirichlet inlet and zero-gradient outlet install
// their own AD-RLB mixin dynamics, which a parameter set earlier would not reach (the same
// trap as the NSE lattice).
inline void prepare(ADLat& ad, SuperGeometry<T,3>& sg, T omegaAD) {
    OstreamManager clout(std::cout,"step4");
    auto fluidish = sg.getMaterialIndicator({MAT_FLUID,MAT_POROUS,MAT_SPONGE});
    ad.template defineDynamics<AD_DYNAMICS>(sg.getMaterialIndicator({MAT_FLUID,MAT_POROUS,MAT_SPONGE,MAT_INLET,MAT_OUTLET}));
    ad.template defineDynamics<NoDynamics>(sg, MAT_VOID);
    // zero-flux solids and faces: buildings, ground, box edges, lateral + top (NSE slip)
    for (int m : {MAT_WALL, MAT_GROUND, MAT_FRAME, MAT_SLIP}) ad.template defineDynamics<BounceBack>(sg, m);
    boundary::set<boundary::AdvectionDiffusionDirichlet<T,AD_DESCRIPTOR,AD_BC_MIXIN>>(ad, sg, MAT_INLET); // clean inflow
    setZeroGradientBoundary<T,AD_DESCRIPTOR,AD_BC_MIXIN>(ad, sg.getMaterialIndicator({MAT_OUTLET}));     // outflow
    // C = 0 on EVERY material, solids included: uninitialised OpenLB populations are 0 in the
    // shifted storage, i.e. an equilibrium at C = 1, which bounce-back cells would hand back to
    // the fluid as phantom concentration.
    AnalyticalConst3D<T,T> zero(T(0)); AnalyticalConst3D<T,T> u0(T(0),T(0),T(0));
    auto all = sg.getMaterialIndicator({MAT_FLUID,MAT_WALL,MAT_INLET,MAT_OUTLET,MAT_SLIP,MAT_POROUS,MAT_GROUND,MAT_SPONGE,MAT_FRAME});
    ad.defineRho(sg.getMaterialIndicator({MAT_INLET}), zero);
    ad.iniEquilibrium(std::move(all), zero, u0);
    // ...and the padding (overlap) cells, which no material indicator reaches. Left at C = 1
    // they cycle through the boundary bounce-back cells with period 2 and the CV mass sum
    // oscillates by a constant-amplitude phantom.
    for (int iC=0; iC<ad.getLoadBalancer().size(); ++iC) {
        auto& b = ad.getBlock(iC); auto& bg = sg.getBlockGeometry(iC);
        const int pad = b.getPadding(), nx=bg.getNx(), ny=bg.getNy(), nz=bg.getNz();
        for (int x=-pad;x<nx+pad;++x) for (int y=-pad;y<ny+pad;++y) for (int z=-pad;z<nz+pad;++z) {
            if (x>=0&&x<nx&&y>=0&&y<ny&&z>=0&&z<nz) continue;
            auto c = b.get(x,y,z);
            for (int i=0;i<AD_DESCRIPTOR::q;++i) c[i] = -descriptors::t<T,AD_DESCRIPTOR>(i);
        }
    }
#ifdef AD_TRT_MAGIC
    const T tauD = 1/omegaAD, tauPlus = T(0.5) + T(AD_TRT_MAGIC)/(tauD - T(0.5));
    ad.template setParameter<descriptors::OMEGA>(1/tauPlus);
    ad.template setParameter<collision::TRT::MAGIC>(T(AD_TRT_MAGIC));
    clout << "AD TRT: Lambda=" << T(AD_TRT_MAGIC) << " tau-(diffusion)=" << tauD << " tau+=" << tauPlus << std::endl;
#else
    ad.template setParameter<descriptors::OMEGA>(omegaAD);
#endif
    ad.initialize();
    (void)fluidish;
    clout << "AD lattice ready: omega_AD=" << omegaAD << " (tau_AD=" << 1/omegaAD << ")" << std::endl;
}

struct Ops {
    int nx=0, ny=0, nz=0; double dx=0;
    int xLo = 1, xHi = 0;                      // control volume in x (inclusive)
    std::vector<double> theta, dep;           // grid-indexed Θ = ∫C dt [lattice C · s], deposit
    std::vector<float>  w;                    // receptor weight (receptor_w.f32), empty if absent
    double wTheta = 0;                        // running <w, Θ>, for J(t) in the time series
    struct Blk {                               // block-local cell lists
        std::vector<int> sx,sy,sz;             // Ω source cells
        std::vector<int> dxv,dyv,dzv; std::vector<T> rate;   // deposition cells, Σ v_d·dt/dx
        std::vector<int> ax,ay,az; std::vector<size_t> ag; std::vector<uint8_t> acv; // fluidish
        int gx0=0,gy0=0,gz0=0, bnx=0,bny=0,bnz=0;
    };
    std::vector<Blk> blk;
    long nSrc=0, nDep=0;
    int iPlusX=-1, iMinusX=-1;

    size_t gidx(int x,int y,int z) const { return (size_t)z*ny*nx+(size_t)y*nx+x; }

    void init(SuperGeometry<T,3>& sg, const bridge::GridField<int32_t>& mat,
              const std::vector<uint8_t>& src, const std::vector<float>& vd, T dtOverDx) {
        nx=mat.nx; ny=mat.ny; nz=mat.nz; dx=mat.dx; xHi = nx-3;
        theta.assign((size_t)nx*ny*nz,0.0); dep.assign((size_t)nx*ny*nz,0.0);
        for (int i=0;i<AD_DESCRIPTOR::q;++i){ auto c=descriptors::c<AD_DESCRIPTOR>(i);
            if (c[0]==1&&c[1]==0&&c[2]==0) iPlusX=i; if (c[0]==-1&&c[1]==0&&c[2]==0) iMinusX=i; }
        static const int dn[6][3]={{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1}};
        auto& load = sg.getLoadBalancer(); blk.assign(load.size(), Blk{});
        for (int iC=0;iC<load.size();++iC){
            auto& bg=sg.getBlockGeometry(iC); auto& B=blk[iC];
            const auto off=bridge::blockOffset(sg,iC);                               // S1
            B.gx0=off[0]; B.gy0=off[1]; B.gz0=off[2]; B.bnx=bg.getNx(); B.bny=bg.getNy(); B.bnz=bg.getNz();
            for(int x=0;x<B.bnx;++x)for(int y=0;y<B.bny;++y)for(int z=0;z<B.bnz;++z){
                const int X=B.gx0+x,Y=B.gy0+y,Z=B.gz0+z;
                if(X<0||X>=nx||Y<0||Y>=ny||Z<0||Z>=nz)continue;
                const int m=mat.data[mat.idx(X,Y,Z)];
                if (m!=MAT_VOID) { B.ax.push_back(x);B.ay.push_back(y);B.az.push_back(z);
                    B.ag.push_back(gidx(X,Y,Z)); B.acv.push_back(X>=xLo&&X<=xHi); }
                const bool fl = (m==MAT_FLUID||m==MAT_POROUS||m==MAT_SPONGE);
                if (!fl) continue;
                if (src[mat.idx(X,Y,Z)]) { B.sx.push_back(x);B.sy.push_back(y);B.sz.push_back(z); ++nSrc; }
                T r=0;
                for(int d=0;d<6;++d){ const int xx=X+dn[d][0],yy=Y+dn[d][1],zz=Z+dn[d][2];
                    if(xx<0||xx>=nx||yy<0||yy>=ny||zz<0||zz>=nz)continue;
                    const int nm=mat.data[mat.idx(xx,yy,zz)];
                    if(nm==MAT_WALL||nm==MAT_GROUND||nm==MAT_POROUS) r += (T)vd[mat.idx(xx,yy,zz)]*dtOverDx; }  // S2
                if (r>0){ B.dxv.push_back(x);B.dyv.push_back(y);B.dzv.push_back(z);B.rate.push_back(std::min<T>(r,1)); ++nDep; }
            }
        }
    }
    // shifted storage: f = f~ + t_i, so C = Σf~ + 1
    template <class CELL> static T conc(CELL c){ T s=1; for(int i=0;i<AD_DESCRIPTOR::q;++i) s+=c[i]; return s; }

    // exact lattice mass flux across the CV's two x-faces during the last stream
    void faceFlux(ADLat& ad, double& outDown, double& outUp) {
        double dn=0, up=0;
        for (size_t iC=0;iC<blk.size();++iC){ auto& b=ad.getBlock(iC); auto& B=blk[iC];
            for (int x=0;x<B.bnx;++x){ const int X=B.gx0+x;
                if (X!=xHi && X!=xHi+1 && X!=0 && X!=1) continue;
                for(int y=0;y<B.bny;++y)for(int z=0;z<B.bnz;++z){
                    auto c=b.get(x,y,z);
                    const T fp=c[iPlusX]+descriptors::t<T,AD_DESCRIPTOR>(iPlusX);
                    const T fm=c[iMinusX]+descriptors::t<T,AD_DESCRIPTOR>(iMinusX);
                    if (X==xHi+1) dn += fp;                           // +x across xHi|xHi+1
                    if (X==xHi)   dn -= fm;
                    if (X==0)     up += fm;                           // -x across 0|1
                    if (X==1)     up -= fp;
                } } }
        outDown=dn; outUp=up;
    }
    // ΔC = q at each Ω cell. Returns the mass emitted inside the control volume; emission at
    // Ω cells beyond it (the city's Ω reaches x = nx-2, past the CV's downstream face) goes to
    // emitOutsideCV, since it leaves through the outlet without ever crossing a CV face.
    double inject(ADLat& ad, T q, double& emitOutsideCV) {
        double e=0, eo=0;
        for (size_t iC=0;iC<blk.size();++iC){ auto& b=ad.getBlock(iC); auto& B=blk[iC];
            for (size_t k=0;k<B.sx.size();++k){ auto c=b.get(B.sx[k],B.sy[k],B.sz[k]);
                for(int i=0;i<AD_DESCRIPTOR::q;++i) c[i]+=descriptors::t<T,AD_DESCRIPTOR>(i)*q;
                const int X=B.gx0+B.sx[k];
                if (X>=xLo && X<=xHi) e+=q; else eo+=q; } }
        emitOutsideCV += eo; return e;
    }
    // remove the fraction `rate` of C (scaling the true populations keeps u-structure)
    double deposit(ADLat& ad, double& depOutsideCV) {
        double dIn=0, dOut=0;
        for (size_t iC=0;iC<blk.size();++iC){ auto& b=ad.getBlock(iC); auto& B=blk[iC];
            const long n=(long)B.rate.size();
            #ifdef PARALLEL_MODE_OMP
            #pragma omp parallel for schedule(static) reduction(+:dIn,dOut)
            #endif
            for (long k=0;k<n;++k){ auto c=b.get(B.dxv[k],B.dyv[k],B.dzv[k]);
                const T C=conc(c), r=B.rate[k], removed=r*C;
                for(int i=0;i<AD_DESCRIPTOR::q;++i){ const T ti=descriptors::t<T,AD_DESCRIPTOR>(i); c[i]=(1-r)*(c[i]+ti)-ti; }
                const int X=B.gx0+B.dxv[k];
                dep[gidx(X,B.gy0+B.dyv[k],B.gz0+B.dzv[k])] += removed;
                if (X>=xLo && X<=xHi) dIn+=removed; else dOut+=removed; } }
        depOutsideCV += dOut; return dIn;
    }
    // Θ += C·dt on fluid cells; returns CV mass (all non-void cells) and domain-wide airborne.
    // Also advances <w,Θ> so J(t) can be logged without exporting Θ.
    double accumulate(ADLat& ad, SuperGeometry<T,3>& sg, T dt, double& airAll) {
        double cv=0, all=0, wc=0; const bool hasW = !w.empty();
        for (size_t iC=0;iC<blk.size();++iC){ auto& b=ad.getBlock(iC); auto& B=blk[iC]; auto& bg=sg.getBlockGeometry(iC);
            const long n=(long)B.ag.size();
            #ifdef PARALLEL_MODE_OMP
            #pragma omp parallel for schedule(static) reduction(+:cv,all,wc)
            #endif
            for (long k=0;k<n;++k){ const T C=conc(b.get(B.ax[k],B.ay[k],B.az[k]));
                if (B.acv[k]) cv+=C;
                const int m=bg.get({B.ax[k],B.ay[k],B.az[k]});
                if (m==MAT_FLUID||m==MAT_POROUS||m==MAT_SPONGE){ theta[B.ag[k]]+=C*dt; all+=C;
                    if (hasW) wc += (double)w[B.ag[k]]*C; } } }
        wTheta += wc*dt;
        airAll=all; return cv;
    }
    // settling: subtract w_s from the coupled vertical velocity (after the coupling ran)
    void settle(ADLat& ad, T wsLB) {
        if (wsLB<=0) return;
        for (size_t iC=0;iC<blk.size();++iC){ auto& b=ad.getBlock(iC); auto& B=blk[iC];
            const long n=(long)B.ag.size();
            #ifdef PARALLEL_MODE_OMP
            #pragma omp parallel for schedule(static)
            #endif
            for (long k=0;k<n;++k){ auto c=b.get(B.ax[k],B.ay[k],B.az[k]);
                auto u=c.template getField<descriptors::VELOCITY>(); u[2]-=wsLB;
                c.template setField<descriptors::VELOCITY>(u); } }
    }
    void uniformVelocity(ADLat& ad, T uxLB) {                    // frozen-flow gate mode
        Vector<T,3> u(uxLB,0,0);
        for (size_t iC=0;iC<blk.size();++iC){ auto& b=ad.getBlock(iC); auto& B=blk[iC];
            for (size_t k=0;k<B.ag.size();++k) b.get(B.ax[k],B.ay[k],B.az[k]).template setField<descriptors::VELOCITY>(u); }
    }
};
} // namespace step4

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

    // ── unit converter ──
    // The relaxation time is the primary stability knob, so it is an INPUT and dt is DERIVED
    // (UnitConverterFromResolutionAndRelaxationTime: dt = (tau-0.5)/3 * dx^2 / nu).
    //
    // nu here cannot be molecular air (1.5e-5 m^2/s). At dx = 4 m that gives dt ~ 3.6e3 s and
    // a lattice velocity ~ 3.6e3 for any tau >= 0.505 -- the Mach gate would (rightly) refuse.
    // The previous converter fixed dt from uLB = 0.05 instead, which with molecular nu put tau
    // at 0.5000001. At grid scale the resolved flow has no molecular viscosity to speak of;
    // what the lattice carries is a BACKGROUND eddy viscosity, on top of which WALE adds its
    // local nu_t. So nu is NU_EFF, derived by default from the target lattice velocity:
    //     nu_eff = (tau-0.5)/3 * dx * U / uLB      (dx=4, U=4, tau=0.505, uLB=0.032 -> 0.83)
    // which is the same order as the neutral-ABL eddy viscosity kappa*u*z at z ~ 10 m. Set
    // NU_EFF to fix it directly instead; the preflight then reports the uLB it implies.
    //
    // Defaults (OPENLB_PHASE5_6_GATES.md §6): tau at the 0.505 floor for the least added
    // viscosity, and uLB_ref = 0.032 because the measured PEAK lattice speed is ~3.1x uLB_ref
    // (log-law top + resolved fluctuation; 0.153 on the city at 0.05) and must stay < 0.1.
    // The earlier 0.51 / 0.05 ran at peak Ma ~ 0.26 with 30 % more viscosity.
    const T U_INLET = envd("U_INLET", 4.0), Z_REF = envd("ABL_ZREF", 4.0);
    const T TAU     = envd("TAU", 0.505);
    const T U_LB    = envd("LATTICE_U", 0.032);
    const T NU_EFF  = envd("NU_EFF", (TAU-0.5)/3.0 * dx * U_INLET / U_LB);
    const T charL   = (T)(nz*dx);
    UnitConverterFromResolutionAndRelaxationTime<T,DESCRIPTOR> converter(
        /*resolution = cells per charL*/ (size_t)nz, TAU,
        charL, /*charU*/U_INLET, /*nu*/NU_EFF, /*rho*/(T)1.2);
    converter.print();
    clout << "operating point: tau=" << TAU << "  nu_eff=" << NU_EFF << " m^2/s"
          << "  dt=" << converter.getPhysDeltaT() << " s  uLB=" << converter.getCharLatticeVelocity()
          << "  Re_eff(charL)=" << U_INLET*charL/NU_EFF << std::endl;
    if (std::fabs(converter.getPhysDeltaX() - (T)dx) > 1e-9*dx) {
        clout << "converter dx " << converter.getPhysDeltaX() << " != map dx " << dx
              << " -- the lattice would not be 1:1 with the material map" << std::endl;
        return 4;
    }

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
    for (int m=0; m<=MAT_FRAME; ++m)
      clout << "GATE5 MAT " << m << " olb=" << superGeometry.getStatistics().getNvoxel(m) << std::endl;

    // ── verified ABL/RFG inlet ──
    gInlet.z0 = envd("ABL_Z0", 0.045); gInlet.d = 0.0; gInlet.wind_angle = WIND_DEG*M_PI/180.0;
    gInlet.L_turb = envd("ABL_LTURB", 20.0); gInlet.n_modes = envi("ABL_NMODES", 100);
    gInlet.sigma_u_ratio=2.5; gInlet.sigma_v_ratio=1.9; gInlet.sigma_w_ratio=1.25;
    gInlet.init(U_INLET, Z_REF, /*seed=*/(unsigned)envi("ABL_SEED", 1000));   // Gate 8 seed pairs
    clout << "ABL inlet u*=" << gInlet.u_star << " (verified: mean 0.10%, div 5.2%)" << std::endl;
    {   // (C1, continued) the converter's charU is U at Z_REF, but the log-law inlet is fastest
        // at the domain top, and the resolved fluctuation rides on top of that. The Mach
        // preflight above therefore under-states the real peak; report the real one. Gate 6a
        // then gates on the MEASURED peak |u| (the "maxU < 0.1 lu throughout" condition).
        const double zTop = (nz-1)*dx, uTop = gInlet.mean_speed(zTop);
        const double uPk  = uTop + 3.0*gInlet.sigma_u_ratio*gInlet.u_star;
        const T lbTop = converter.getLatticeVelocity((T)uTop), lbPk = converter.getLatticeVelocity((T)uPk);
        clout << "inlet peak: mean U(z_top=" << zTop << " m)=" << uTop << " m/s -> uLB " << lbTop
              << "; +3 sigma_u -> uLB " << lbPk << (lbPk < 0.1 ? "  [ok]" : "  [WARN: above 0.1 lu]") << std::endl;
    }

    // ── lattice ──
    SuperLattice<T,DESCRIPTOR> sLattice(superGeometry);                              // CONFIRM 1.8
    prepareLattice(sLattice, converter, superGeometry, nSponge);

    // ── develop the LIVE turbulent flow (no averaging) ──
    const int SPIN_FT   = envi("SPINUP_FT", 3);
    const int stepsPerFT= (int)std::llround((nx*dx)/U_INLET / converter.getPhysDeltaT());  // CONFIRM 1.8
    const int SPIN_STEPS= SPIN_FT*stepsPerFT;
    // STEP4_UNIFORM_U (Phase-6 gate mode) needs no developed flow, so it skips the spin-up.
    const int MAX_STEPS = envi("MAX_STEPS", envd("STEP4_UNIFORM_U",0.0) > 0 ? 0 : SPIN_STEPS);
    const int rampSteps = stepsPerFT;                      // (C3) ramp over 1 flow-through
    const int CHECK     = envi("CHECK_EVERY", 200);        // (C4) divergence-guard cadence
    const int ADM_EVERY = envi("ADM_EVERY", 0);           // (C6) 0 = ADM off
    clout << "live spin-up: " << SPIN_FT << " flow-throughs = " << SPIN_STEPS
          << " steps (" << stepsPerFT << "/FT), ramp " << rampSteps << ", check " << CHECK << std::endl;
    if (MAX_STEPS != SPIN_STEPS)
        clout << "MAX_STEPS=" << MAX_STEPS << " overrides the spin-up: running " << MAX_STEPS
              << " steps (" << (T)MAX_STEPS/stepsPerFT << " flow-throughs)" << std::endl;

    // Phase-5 gates: time-average the last AVG_FT flow-throughs of the run (0 = off).
    const T   AVG_FT    = envd("AVG_FT", 0.0);
    const int AVG_EVERY = envi("AVG_EVERY", 10);
    const int avgStart  = AVG_FT>0 ? std::max(0, MAX_STEPS - (int)std::llround(AVG_FT*stepsPerFT)) : MAX_STEPS;
    TimeMean tmean;
    if (AVG_FT>0) {
        tmean.init(nx,ny,nz);
        clout << "time mean: steps " << avgStart << ".." << MAX_STEPS << " every " << AVG_EVERY
              << " -> " << OUT << "/uavg.f32" << std::endl;
    }
    // (C2/G1) floor treatment: GROUND_MODEL=1 rough wall (default), 0 = plain bounce-back.
    const int GROUND_MODEL = envi("GROUND_MODEL", 1);
    RoughWall roughWall;
    if (GROUND_MODEL==1) {
        roughWall.init(superGeometry, envd("ABL_Z0", 0.045), envd("WALL_ZP", dx));
        clout << "ground: rough wall, z0=" << roughWall.z0 << " m, z_P=" << roughWall.zP << " m, "
              << roughWall.nCells << " layer cells (" << roughWall.nSpecular << " specular slots)" << std::endl;
    } else clout << "ground: bounce-back (no-slip at the first cell)" << std::endl;
    // Richards & Hoxey top stress, TOP_STRESS=1 (default) with the rough wall; 0 = plain slip lid.
    const int TOP_STRESS = envi("TOP_STRESS", GROUND_MODEL==1 ? 1 : 0);
    TopStress topStress;
    const T ustarLB = converter.getLatticeVelocity((T)gInlet.u_star);
    if (TOP_STRESS) { topStress.init(superGeometry, nz, WIND_DEG*M_PI/180.0);
        clout << "top: shear stress u*^2 (u*=" << gInlet.u_star << " m/s) on " << topStress.nCells << " cells" << std::endl; }
    int floorIT = 0;                                         // step counter for the ramp
    auto floorStep = [&]{
        if (GROUND_MODEL==1) roughWall.apply(sLattice);
        if (TOP_STRESS) {
            const T r = bridge::AblVelocityF3D<T,DESCRIPTOR,UnitConverter<T,DESCRIPTOR>>::smoothstep(
                            rampSteps>0 ? (T)floorIT/(T)rampSteps : T(1));
            topStress.apply(sLattice, ustarLB*r);
        }
        ++floorIT;
    };

    const T cvel = converter.getConversionFactorVelocity();
    T peakU = 0;
    util::Timer<T> timer(MAX_STEPS, superGeometry.getStatistics().getNvoxel());
    timer.start();

#if COLLISION_MODEL==0
    VeloGradRefresh veloGrad;                             // WALE needs VELO_GRAD every step
    auto refreshWALE = [&]{ veloGrad(sLattice, superGeometry); };
    const int GRAD_CHECK_AT = envi("WALE_GRAD_CHECK",0) ? std::min(MAX_STEPS-1, stepsPerFT/2) : -1;
#else
    auto refreshWALE = []{};
    const int GRAD_CHECK_AT = -1;
#endif

    for (int iT=0; iT<MAX_STEPS; ++iT) {
        setBoundaryValues(sLattice, converter, superGeometry, iT, rampSteps);   // C3
        refreshWALE();
#if COLLISION_MODEL==0
        if (iT == GRAD_CHECK_AT) {
            // One-shot check of VeloGradRefresh against OpenLB's own functor, on MAT_FLUID
            // cells whose +-4 neighbourhood carries fluid on every axis, where both take the
            // 8th-order central branch. Must agree to round-off; else an indexing/layout bug.
            std::list<int> mats{MAT_FLUID, MAT_INLET, MAT_OUTLET, MAT_POROUS, MAT_SPONGE};
            SuperLatticeVelocityGradientFD3D<T,DESCRIPTOR> ref(superGeometry, sLattice, mats);
            auto& bl = sLattice.getBlock(0); auto& bg = superGeometry.getBlockGeometry(0);
            T maxAbs=0, maxDiff=0; long n=0;
            for (int x=4;x<bg.getNx()-4;x+=3) for (int y=4;y<bg.getNy()-4;y+=3) for (int z=4;z<bg.getNz()-4;z+=3) {
                if (bg.get({x,y,z}) != MAT_FLUID) continue;
                bool inner=true;
                for (int d=-4; d<=4 && inner; ++d)
                    inner = VeloGradRefresh::carriesFluid(bg.get({x+d,y,z}))
                         && VeloGradRefresh::carriesFluid(bg.get({x,y+d,z}))
                         && VeloGradRefresh::carriesFluid(bg.get({x,y,z+d}));
                if (!inner) continue;
                T r[9]; int in[4]={0,x,y,z}; ref(r, in);
                auto g = bl.get(x,y,z).template getField<descriptors::VELO_GRAD>();
                for (int q=0;q<9;++q){ maxAbs=std::max(maxAbs,std::fabs(r[q])); maxDiff=std::max(maxDiff,std::fabs(r[q]-g[q])); }
                ++n;
            }
            clout << "WALE_GRAD_CHECK at iT=" << iT << ": " << n << " cells, max|grad|=" << maxAbs
                  << " max|diff vs OpenLB functor|=" << maxDiff
                  << (maxDiff <= 1e-9*std::max(maxAbs,T(1e-30)) + 1e-15 ? "  [PASS]" : "  [FAIL]") << std::endl;
        }
#endif
        sLattice.collideAndStream();                                            // CONFIRM 1.8
        floorStep();                                                            // C2
        if (iT>=avgStart && (iT-avgStart)%AVG_EVERY==0) tmean.sample(sLattice, superGeometry, cvel);

        if (ADM_EVERY>0 && iT%ADM_EVERY==0) {                                   // C6 optional ADM filter
            // CONFIRM 1.8: SuperLatticeADM3D<T,DESCRIPTOR> admF(sLattice, adm_sigma, adm_order);
            //             admF.execute(superGeometry, MAT_FLUID);   // approximate deconvolution
        }
        if (iT%CHECK==0) {
            T maxU = 0;
            if (!healthy(sLattice, superGeometry, &maxU)) { return 2; }         // C4 abort on divergence
            peakU = std::max(peakU, maxU);
            timer.update(iT); timer.printStep();
            clout << "iT=" << iT << " t=" << converter.getPhysTime(iT) << "s maxU_lb=" << maxU
                  << " ramp=" << bridge::AblVelocityF3D<T,DESCRIPTOR,UnitConverter<T,DESCRIPTOR>>::smoothstep(
                        rampSteps>0?(T)iT/(T)rampSteps:T(1)) << std::endl;
        }
    }
    { T maxU = 0; if (!healthy(sLattice, superGeometry, &maxU)) return 2; peakU = std::max(peakU, maxU); }
    timer.stop(); timer.printSummary();
    clout << "peak lattice |u| over run = " << peakU << " (Mach " << peakU*std::sqrt(3.0) << ")" << std::endl;
    if (AVG_FT>0) { tmean.write(OUT+"/uavg.f32", dx);
        clout << "wrote time mean " << OUT << "/uavg.f32 (" << tmean.nSamples << " samples)" << std::endl; }

    // ── snapshot the developed LIVE field for Stage C ──
    exportLiveFlow(sLattice, converter, superGeometry, nx, ny, nz, dx, OUT);
    // Stage C reads OUT only (visualize_forward.py): the geometry it masks with, and a
    // meta.txt saying umean_full.f32 is already in m/s. forward_city wrote LATTICE velocity
    // there and the viz scales by U_inlet/U_LB (~69x) unless told otherwise.
    for (const char* fn : {"geom_type.u8", "source_mask.u8", "receptor_w.f32"}) {
        std::string c = "cp -f '" + GEOM + "/" + fn + "' '" + OUT + "/' 2>/dev/null"; if (system(c.c_str())) {}
    }
    if (FILE* mf = fopen((OUT+"/meta.txt").c_str(), "w")) {
        fprintf(mf, "# written by urban_flow (OpenLB)\nU_inlet_ms %.6g\nvelocity_units ms\n", (double)U_INLET);
        fprintf(mf, "grid %d %d %d\ndx_m %.4f\ndt_s %.6g\ntau %.6g\nnu_eff_m2s %.6g\nabl_seed %d\n",
                nx, ny, nz, dx, (double)converter.getPhysDeltaT(), (double)TAU, (double)NU_EFF, envi("ABL_SEED",1000));
        fclose(mf);
    }

    if (!envi("STEP4",0)) {
        clout << "urban_flow COMPLETE (airflow only; STEP4=1 runs the burst). "
              << "Wrote umean_full.f32 to " << OUT << "/" << std::endl;
        return 0;
    }

    // ════════════════════════ STEP 4: live-flow burst transport ════════════════════════
    // Continue the LIVE flow and run the accidental burst over Ω on it (no frozen mean).
    // STEP4_UNIFORM_U > 0 (m/s) is the Phase-6 gate mode: no NSE at all, the scalar rides a
    // frozen uniform +x wind, so transport and deposition are tested in isolation.
    OstreamManager cl4(std::cout,"step4");
    if (WIND_DEG != 0.0) cl4 << "WARNING: the measured budget assumes +x wind; WIND_DEG=" << WIND_DEG << std::endl;
    std::vector<uint8_t> src((size_t)nx*ny*nz, 0);
    if (const char* sc = getenv("SRC_CELLS")) {          // "x,y,z;x,y,z" (linearity guard)
        std::string str(sc); size_t p=0;
        while (p < str.size()) {
            size_t q = str.find(';', p); if (q==std::string::npos) q = str.size();
            int x,y,z; if (sscanf(str.substr(p,q-p).c_str(), "%d,%d,%d", &x,&y,&z)==3
                           && x>=0&&x<nx&&y>=0&&y<ny&&z>=0&&z<nz) src[mat.idx(x,y,z)] = 1;
            p = q+1;
        }
    } else {
        bridge::GridField<uint8_t> sm;
        if (!bridge::load_source_mask(GEOM+"/source_mask.u8", sm)) { cl4<<"no source_mask.u8"<<std::endl; return 2; }
        src = sm.data;
    }
    std::vector<float> vd((size_t)nx*ny*nz, 0.f);
    { bridge::GridField<float> dv; if (bridge::read_grid(GEOM+"/dep_vel.f32", dv)) vd = dv.data;
      else cl4 << "no dep_vel.f32: deposition off" << std::endl; }
    if (envd("VD_SCALE",1.0) != 1.0) for (auto& v : vd) v *= (float)envd("VD_SCALE",1.0);

    const T dt = converter.getPhysDeltaT();
    const T D_MOL = envd("D_MOL", 1e-5), SC_T = envd("SC_T", 0.7);
    // S3: constant D_eff = D_mol + nu_t/Sc_t with nu_t = the lattice's background viscosity
    // (the unresolved part; resolved eddies advect the scalar themselves). Per-cell WALE
    // nu_t coupling is a later refinement.
    const T D_EFF   = envd("D_EFF", D_MOL + NU_EFF/SC_T);
    const T omegaAD = converter.template getLatticeRelaxationFrequencyFromDiffusivity<AD_DESCRIPTOR>(D_EFF);
    cl4 << "D_eff=" << D_EFF << " m^2/s -> D_lb=" << D_EFF*dt/(dx*dx) << ", tau_AD=" << 1/omegaAD << std::endl;
    if (1/omegaAD < 0.505) cl4 << "WARNING: tau_AD < 0.505" << std::endl;

    step4::ADLat adLattice(superGeometry);
    step4::prepare(adLattice, superGeometry, omegaAD);
    step4::Ops ops;
    ops.init(superGeometry, mat, src, vd, dt/(T)dx);
    { bridge::GridField<float> wg; if (bridge::read_grid(GEOM+"/receptor_w.f32", wg) && wg.size()==(size_t)nx*ny*nz) ops.w = wg.data;
      else cl4 << "no receptor_w.f32: J(t) not logged" << std::endl; }
    cl4 << "Omega source cells " << ops.nSrc << ", deposition cells " << ops.nDep
        << ", control volume x in [" << ops.xLo << "," << ops.xHi << "]" << std::endl;
    if (ops.nSrc == 0) { cl4 << "no source cells" << std::endl; return 2; }

    const T U_UNIFORM = envd("STEP4_UNIFORM_U", 0.0);
    const bool uniform = U_UNIFORM > 0;
    SuperLatticeCoupling coupling(NavierStokesAdvectionDiffusionVelocityCoupling{},
                                  names::NavierStokes{}, sLattice, names::Concentration0{}, adLattice);
    coupling.restrictTo(superGeometry.getMaterialIndicator({MAT_FLUID,MAT_POROUS,MAT_SPONGE,MAT_INLET,MAT_OUTLET}));
    if (uniform) { ops.uniformVelocity(adLattice, converter.getLatticeVelocity(U_UNIFORM));
                   cl4 << "frozen uniform wind " << U_UNIFORM << " m/s (NSE not stepped)" << std::endl; }
    const T wsLB = converter.getLatticeVelocity(envd("W_SETTLE", 0.0));

    const T   PULSE_S  = envd("PULSE_S", 2.0);
    const T   Q_RATE   = envd("Q_RATE", 1.0);               // ΔC per Ω cell per step (lattice)
    const T   CLEAR    = envd("CLEAR_FRAC", 0.01);          // 0 = run all MAX_BURST_STEPS
    const int MAXB     = envi("MAX_BURST_STEPS", 20*stepsPerFT);
    const int TS_EVERY = envi("TS_EVERY", 200);
    // emit is the CV's emission (what the budget closes against); emitOut is Ω beyond the CV.
    double emit=0, emitOut=0, depCV=0, depOut=0, outDown=0, outUp=0, airCV=0, airAll=0, peakAir=0; int endStep=MAXB;
    { double a0=0; const double m0 = ops.accumulate(adLattice, superGeometry, 0, a0);
      cl4 << "pre-release CV mass " << m0 << " (must be 0)" << std::endl; }
    FILE* ts=fopen((OUT+"/exposure_timeseries.csv").c_str(),"w");
    if(ts) fprintf(ts,"step,t_s,airborne_cv,deposited_cv,emitted,out_downstream,out_upstream,budget_resid,airborne_all,J\n");

    for (int iB=0; iB<MAXB; ++iB) {
        if (!uniform) {
            setBoundaryValues(sLattice, converter, superGeometry, MAX_STEPS+iB, 0);   // inlet stays live
            refreshWALE();
            sLattice.collideAndStream();                                              // NSE (live)
            floorStep();
            coupling.execute();                                                       // u -> AD VELOCITY
            ops.settle(adLattice, wsLB);                                              // −w_s ẑ
        }
        adLattice.collideAndStream();
        double fd=0, fu=0; ops.faceFlux(adLattice, fd, fu); outDown+=fd; outUp+=fu;
        const bool pulseOn = (iB*dt) < PULSE_S;
        if (pulseOn) emit += ops.inject(adLattice, Q_RATE, emitOut);
        depCV += ops.deposit(adLattice, depOut);
        airCV = ops.accumulate(adLattice, superGeometry, dt, airAll);
        peakAir = std::max(peakAir, airCV);
        const double resid = emit>0 ? (emit-(depCV+airCV+outDown+outUp))/emit : 0.0;
        if (!std::isfinite(airCV)) { OstreamManager c(std::cout,"DIVERGED"); c<<"AD non-finite at burst step "<<iB<<std::endl; if(ts)fclose(ts); return 2; }
        if (iB%TS_EVERY==0) {
            // J(t) = <w,Θ(t)>/M_released: Stage C's J accumulated so far (lattice units)
            const double Jt = (emit+emitOut)>0 ? ops.wTheta/(emit+emitOut) : 0.0;
            if (ts) { fprintf(ts,"%d,%.4f,%.9e,%.9e,%.9e,%.9e,%.9e,%.3e,%.9e,%.9e\n",iB,iB*dt,airCV,depCV,emit,outDown,outUp,resid,airAll,Jt); fflush(ts); }
            cl4 << "iB=" << iB << " t=" << iB*dt << "s air=" << airCV << " dep=" << depCV
                << " out=" << outDown << "+" << outUp << " emit=" << emit << " resid=" << resid
                << " airborne_frac=" << ((emit+emitOut)>0 ? airAll/(emit+emitOut) : 0.0) << " J=" << Jt << std::endl;
        }
        if (CLEAR>0 && !pulseOn && peakAir>0 && airCV < CLEAR*peakAir) { endStep=iB+1; break; }
    }
    if (ts) fclose(ts);

    // export Θ = ∫C dt and the deposition map for Stage C (J = ⟨w,Θ⟩/|Ω|)
    std::vector<float> th(ops.theta.begin(), ops.theta.end()), de(ops.dep.begin(), ops.dep.end());
    writeField5(OUT+"/theta.f32",      th, nx,ny,nz,dx);
    writeField5(OUT+"/deposition.f32", de, nx,ny,nz,dx);

    const double drained = outDown + outUp;
    const double closure = emit>0 ? std::fabs(emit-(depCV+airCV+drained))/emit : 0.0;
    { FILE* mf=fopen((OUT+"/meta_flow.txt").c_str(),"w"); if(mf){
        fprintf(mf,"grid %d %d %d\ndx_m %.4f\nomega_cells %ld\n",nx,ny,nz,dx,ops.nSrc);
        fprintf(mf,"burst_steps %d\ndt_s %.6g\nD_eff_m2s %.6g\ntau_AD %.6g\n",endStep,(double)dt,(double)D_EFF,(double)(1/omegaAD));
        fprintf(mf,"# control volume x in [%d,%d]; budget: emitted = deposited + airborne + drained\n",ops.xLo,ops.xHi);
        fprintf(mf,"mass_emitted %.9e\nmass_deposited %.9e\nmass_airborne %.9e\nmass_drained %.9e\n",emit,depCV,airCV,drained);
        fprintf(mf,"mass_emitted_beyond_cv %.9e\nmass_emitted_total %.9e\n",emitOut,emit+emitOut);
        fprintf(mf,"mass_out_downstream %.9e\nmass_out_upstream %.9e\nmass_deposited_beyond_cv %.9e\n",outDown,outUp,depOut);
        fprintf(mf,"budget_closure %.6e\ndeposited_frac %.6f\n",closure, emit+emitOut>0?(depCV+depOut)/(emit+emitOut):0.0);
        fprintf(mf,"theta_layout 5xint32[nx,ny,nz,dx*1000,1] units lattice_C*s\n");
        fprintf(mf,"# Stage C: J = (1/omega_cells) * sum_x receptor_w(x) * theta(x)\n"); fclose(mf);} }

    if (emitOut > 0) cl4 << "Omega beyond the control volume emitted " << emitOut << " (not in the budget)" << std::endl;
    cl4 << "budget: emitted " << emit << " = deposited " << depCV << " + airborne " << airCV
        << " + drained " << drained << " (down " << outDown << ", up " << outUp << ")  closure "
        << closure << (closure<0.01 ? "  [<1%]" : "  [>=1%]") << std::endl;
    clout << "urban_flow COMPLETE — live airflow + burst transport. Wrote umean_full.f32, "
          << "theta.f32, deposition.f32, exposure_timeseries.csv, meta_flow.txt to " << OUT << "/" << std::endl;
    return 0;
}
