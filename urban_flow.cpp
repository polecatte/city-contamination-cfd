// urban_flow.cpp — STAGE B (airflow, LIVE flow): OpenLB 1.8.1 NSE + WALE LES engine for the
// urban_openlbm migration. Consumes the geometry-bridge outputs (material_map.dat +
// source_mask.u8), develops a LIVE turbulent wind field on the city, and hands it to the
// scalar burst (Step 4). No time-averaging — this is the FORWARD_LIVE regime: the burst
// rides the instantaneous, evolving flow (matching forward_city.cpp), so the airflow stage
// only needs to reach a statistically-developed turbulent state, then snapshot it.
//
// ┌─ SCOPE ──────────────────────────────────────────────────────────────────────┐
// │ Steps 3+4 of OPENLB_MIGRATION_PLAN.md §7: D3Q19 + WALE LES, ABL/RFG velocity   │
// │ inlet, pressure outlet, free-slip sides/top; then a coupled D3Q7 AD lattice     │
// │ with settling, Ω burst, deposition sink and Θ=∫C dt. Developed LIVE (no avg).   │
// └──────────────────────────────────────────────────────────────────────────────┘
//
// ⚠ VERIFICATION STATUS (read before trusting a number out of this file)
//   This file has NEVER BEEN COMPILED — there is no OpenLB in the authoring environment.
//   It was ported against the OpenLB 1.8.1 Doxygen (the User Guide's listings are stale
//   and do not compile against 1.8). Treat it as a well-informed first compile candidate,
//   not as working code. Follow the phased plan in OPENLB_PORT_STATUS_AND_VERIFICATION.md
//   §3 — in particular do NOT skip Gate 5 (geometry round-trip), which is what catches a
//   silently-scrambled material map.
//   Remaining genuinely-uncertain spots are marked "CONFIRM 1.8". They are now few: the
//   whole 1.4-era idiom layer (olbInit, instances::, CuboidGeometry3D, the set*Boundary
//   free functions) has been replaced with the 1.8 declarative API.
//
// CORRECTIVE / STABILISATION METHODS wired here (the numerical safety net the custom
// engine had — see CHANGES.md on ABL drift and low-Re stability):
//   (C1) Mach / CFL / τ / ω_AD preflight — refuse to run outside the stable regime.
//   (C2) GROUND treatment. DEFAULT IS BOUNCE-BACK, deliberately: OpenLB 1.8 has NO
//        aerodynamic-roughness (z0) wall function (see G1 in the audit), so the rough-wall
//        branch cannot be written against a stock 1.8 API. Measure the drift with
//        bounce-back first (Gate 6a) and only then decide whether to write the custom
//        z0 post-processor. Set GROUND_WALLFUNCTION=1 to compile the experimental branch.
//   (C3) Inlet startup RAMP (smoothstep over 1 flow-through) — no pressure shock.
//   (C4) DIVERGENCE / NaN guard — abort cleanly on non-finite or super-Mach state.
//   (C5) Outlet SPONGE layer — a GRADED Smagorinsky fringe (per-cell C_s via
//        ExternalSmagorinskyBGKdynamics) absorbing turbulence before the pressure outlet.
//   (C6) LES stabilisation — selectable collision (WALE / consistent-Smagorinsky /
//        regularized) + optional ADM deconvolution filter.
//
// Build (drop in examples/urban/urban_flow/ with the supplied Makefile, then `make`).
// Run: GEOM_DIR=./geom_out ./urban_flow

#define URBAN_FLOW_WITH_OPENLB 1

#include "olb3D.h"
#ifndef OLB_PRECOMPILED
#include "olb3D.hh"
#endif

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace olb;
using namespace olb::descriptors;
typedef double T;

// ─────────────────────────────────────────────────────────────────────────────
// (B5) Custom descriptor fields. These were previously referenced as
// descriptors::THETA / descriptors::DEPOSIT but never defined anywhere — the file could
// not have compiled. They are ours, so they are declared here and used UNQUALIFIED.
//   THETA   — running Θ = ∫C dt (the quantity Stage C contracts against w)
//   DEPOSIT — cumulative deposited mass at wall-adjacent fluid cells
// ─────────────────────────────────────────────────────────────────────────────
struct THETA   : public descriptors::FIELD_BASE<1> {};
struct DEPOSIT : public descriptors::FIELD_BASE<1> {};

// ── LES collision selection (C6) ──  -DCOLLISION_MODEL=0|1|2  (default 0 = WALE)
#ifndef COLLISION_MODEL
#define COLLISION_MODEL 0
#endif

// (B6) The NSE descriptor. WALED3Q19Descriptor is D3Q19<EFFECTIVE_OMEGA,VELO_GRAD> and
// carries NO POROSITY field, so PorousBGKdynamics could not instantiate on it. We declare
// the descriptor explicitly with the porosity field the park canopy needs.
using DESCRIPTOR = D3Q19<EFFECTIVE_OMEGA, VELO_GRAD, POROSITY>;

#if   COLLISION_MODEL==0
  using BULK_DYNAMICS = WALEBGKdynamics<T,DESCRIPTOR>;
#elif COLLISION_MODEL==1
  using BULK_DYNAMICS = ConStrainSmagorinskyBGKdynamics<T,DESCRIPTOR>;
#else
  using BULK_DYNAMICS = RLBdynamics<T,DESCRIPTOR>;
#endif

// ── Step 4: advection–diffusion scalar lattice (D3Q7) ──
using AD_DESCRIPTOR = D3Q7<VELOCITY, THETA, DEPOSIT>;
using AD_DYNAMICS   = AdvectionDiffusionBGKdynamics<T,AD_DESCRIPTOR>;

// (B3) Dynamics aliases. The whole `olb::instances` namespace and the `Dynamics*`
// overload of defineDynamics are gone in 1.8; dynamics are now named types passed as
// template arguments. If your 1.8 build prefers the template-template spelling
// (defineDynamics<BounceBack>(...)), change it HERE in one place, not at eight call sites.
using BounceBackNS  = BounceBack<T,DESCRIPTOR>;
using NoDynamicsNS  = NoDynamics<T,DESCRIPTOR>;
using BounceBackAD  = BounceBack<T,AD_DESCRIPTOR>;
using NoDynamicsAD  = NoDynamics<T,AD_DESCRIPTOR>;

#include "geometry_loader.h"   // Stage-A material map + source mask reader/stamper
#include "abl_inlet_olb.h"     // verified ABL/RFG inlet as AnalyticalF3D (with startup ramp)

static double envd(const char* k,double d){const char* e=getenv(k);return e?atof(e):d;}
static int    envi(const char* k,int d){const char* e=getenv(k);return e?atoi(e):d;}

// Material numbers — MUST match openlb_geometry.h (note MAT_GROUND split from MAT_WALL,
// and MAT_SPONGE is a SOLVER-LOCAL material carved out of MAT_FLUID near the outlet (C5)).
enum { MAT_VOID=0, MAT_FLUID=1, MAT_WALL=2, MAT_INLET=3, MAT_OUTLET=4,
       MAT_SLIP=5, MAT_POROUS=6, MAT_GROUND=7, MAT_SPONGE=8 };

static abl::ABLInlet gInlet;   // built in main, referenced in setBoundaryValues

// ─────────────────────────────────────────────────────────────────────────────
// (S1) Global lattice origin of a LOCAL block.
//
// THE BUG THIS REPLACES: every block loop used `block.getOrigin()[0]` as a global cell
// INDEX. BlockGeometry::getOrigin() returns the origin "in SI units (meter)", so at
// dx=4 m every index was off by ~4× and the geometry was stamped scrambled. It only
// looked right because a single-cuboid serial run has origin (0,0,0) — it would have
// silently corrupted every field the moment the run went MPI or multi-block.
// ─────────────────────────────────────────────────────────────────────────────
template <class SGEOM>
static Vector<int,3> blockOriginOf(SGEOM& sg, int iCloc) {
    auto& cd       = sg.getCuboidDecomposition();              // CONFIRM 1.8 accessor name
    const int iCg  = sg.getLoadBalancer().glob(iCloc);
    auto  latticeR = cd.getMotherCuboid().getLatticeR(cd.get(iCg).getOrigin());
    return Vector<int,3>{ (int)latticeR[0], (int)latticeR[1], (int)latticeR[2] };
}

// ─────────────────────────────────────────────────────────────────────────────
// (C5) Carve a sponge band out of the fluid cells adjacent to the outlet face, on the
// HOST material array before stamping. Wind-aligned: default +x → band at x∈[nx-n,nx).
// Solver-local (MAT_SPONGE); the geometry bridge stays pure geometry.
// ─────────────────────────────────────────────────────────────────────────────
static int spongeAxis(double wind_deg, int& sign) {
    double wd = std::fmod(wind_deg,360.0); if (wd<0) wd+=360.0;
    if (wd < 45 || wd >= 315) { sign=+1; return 0; }   // +x
    if (wd < 135)             { sign=+1; return 1; }   // +y
    if (wd < 225)             { sign=-1; return 0; }   // -x
    sign=-1; return 1;                                 // -y
}

static void carve_sponge(bridge::GridField<int32_t>& m, double wind_deg, int nSponge) {
    if (nSponge <= 0) return;
    const int nx=m.nx, ny=m.ny, nz=m.nz;
    int sign=0; const int axis = spongeAxis(wind_deg, sign);
    auto mark=[&](int x,int y,int z){ size_t i=m.idx(x,y,z); if(m.data[i]==MAT_FLUID) m.data[i]=MAT_SPONGE; };
    if (axis==0 && sign>0) for(int z=0;z<nz;++z)for(int y=0;y<ny;++y)for(int x=nx-nSponge;x<nx;++x) mark(x,y,z);
    else if (axis==0)      for(int z=0;z<nz;++z)for(int y=0;y<ny;++y)for(int x=0;x<nSponge;++x)     mark(x,y,z);
    else if (sign>0)       for(int z=0;z<nz;++z)for(int x=0;x<nx;++x)for(int y=ny-nSponge;y<ny;++y) mark(x,y,z);
    else                   for(int z=0;z<nz;++z)for(int x=0;x<nx;++x)for(int y=0;y<nSponge;++y)     mark(x,y,z);
}

// ─────────────────────────────────────────────────────────────────────────────
// (C5 / G3) Graded Smagorinsky constant across the sponge band.
//
// THE BUG THIS REPLACES: SmagorinskyBGKdynamics reads its constant from
// collision::LES::SMAGORINSKY, which is a LATTICE-GLOBAL parameter — you cannot raise it
// on MAT_SPONGE alone, so the "graded absorbing fringe" the design calls for was not what
// the code did. ExternalSmagorinskyBGKdynamics takes C_s from a per-cell field, which is
// what makes the grading expressible.
// ─────────────────────────────────────────────────────────────────────────────
template <typename TT>
class SpongeCsF3D : public AnalyticalF3D<TT,TT> {
public:
    SpongeCsF3D(TT csBulk, TT csMax, TT bandStart_m, TT bandEnd_m, int axis)
        : AnalyticalF3D<TT,TT>(1), _cs0(csBulk), _cs1(csMax),
          _a(bandStart_m), _b(bandEnd_m), _axis(axis) {}
    bool operator()(TT output[], const TT input[]) override {
        TT s = (_b != _a) ? (input[_axis] - _a) / (_b - _a) : TT(1);
        s = s < 0 ? TT(0) : (s > 1 ? TT(1) : s);
        s = s*s*(3-2*s);                       // smoothstep: no C_s discontinuity
        output[0] = _cs0 + (_cs1 - _cs0) * s;
        return true;
    }
private:
    TT _cs0,_cs1,_a,_b; int _axis;
};

// ─────────────────────────────────────────────────────────────────────────────
// (C1) Mach / CFL / relaxation preflight — pure converter arithmetic, no lattice yet.
//
// The τ gate is now 0.505, not 0.5. THE BUG THIS REPLACES: at the previous operating
// point τ = 0.5000001, which passed `τ > 0.5` by 1.4e-7 while being physically ZERO
// molecular viscosity — BGK at τ→0.5 with no SGS contribution is unconditionally
// unstable, so the preflight was a false-negative sieve that would have cleared a run
// destined to blow up. See `viscosityFloorFor()` in main() for how τ is now pinned.
// ─────────────────────────────────────────────────────────────────────────────
static bool preflight(UnitConverter<T,DESCRIPTOR> const& c, T omegaAD, T ReEff) {
    OstreamManager clout(std::cout,"preflight");
    const T uLB  = c.getCharLatticeVelocity();
    const T tau  = c.getLatticeRelaxationTime();
    const T Ma   = uLB * std::sqrt(3.0);
    bool ok=true; auto need=[&](bool cnd,const char* msg){ clout<<"["<<(cnd?"PASS":"FAIL")<<"] "<<msg<<std::endl; if(!cnd) ok=false; };
    clout << "uLB=" << uLB << "  tau=" << tau << "  Ma=" << Ma
          << "  omega_AD=" << omegaAD << "  Re_eff=" << ReEff << std::endl;
    need(std::isfinite(uLB)&&uLB>0, "lattice velocity finite and positive");
    need(tau > 0.505,               "relaxation tau > 0.505 (a real viscosity, not tau->0.5)");
    need(Ma  < 0.1,                 "inlet Mach < 0.1 (safe low-compressibility regime)");
    need(uLB < 0.1,                 "lattice velocity < 0.1 (CFL / stability margin)");
    need(omegaAD > 0.5 && omegaAD < 1.98, "AD relaxation omega in (0.5, 1.98)");
    if (omegaAD > 1.9)
        clout << "[WARN] omega_AD=" << omegaAD << " is strongly over-relaxed (D_eff is small "
              << "in lattice units). Stable in principle, but if the scalar rings, raise SC_T "
              << "or coarsen dt." << std::endl;
    if (ReEff < 1.1e4)
        clout << "[WARN] effective Re=" << ReEff << " is below the Re-independence threshold "
              << "(~1.1e4, Snyder 1972): the viscosity floor that buys stability also makes the "
              << "building-scale flow Re-dependent. This is the SAME regime the custom engine "
              << "ran in — it is exactly what Gate 6b (Xr/H) measures. Not fatal; know it."
              << std::endl;
    clout << "preflight " << (ok?"PASS — cleared to run":"FAIL — refusing to run (set FORCE=1 to override)") << std::endl;
    return ok;
}

// ─────────────────────────────────────────────────────────────────────────────
// prepareLattice — dynamics + boundary conditions + correctives by material number.
// (B3) defineDynamics<TYPE>(geometry, material) ; (B4) the declarative boundary:: API.
// ─────────────────────────────────────────────────────────────────────────────
void prepareLattice(SuperLattice<T,DESCRIPTOR>& sLattice,
                    UnitConverter<T,DESCRIPTOR> const& converter,
                    SuperGeometry<T,3>& superGeometry, int nSponge,
                    double wind_deg, int nx, int ny, double dx) {
    OstreamManager clout(std::cout, "prepareLattice");

    sLattice.template defineDynamics<NoDynamicsNS>(superGeometry, MAT_VOID);

    // bulk fluid + inlet + outlet carry the selected LES dynamics (C6)
    auto bulkInd = superGeometry.getMaterialIndicator({MAT_FLUID, MAT_INLET, MAT_OUTLET});
    sLattice.template defineDynamics<BULK_DYNAMICS>(bulkInd);

    // buildings: smooth no-slip bounce-back (COST 732 — buildings are aerodynamically smooth)
    sLattice.template defineDynamics<BounceBackNS>(superGeometry, MAT_WALL);

    // (C2) GROUND. See the header note: OpenLB 1.8 has no z0 rough-wall function, so
    // bounce-back is the DEFAULT and the wall-function branch is opt-in and unbuilt.
    // G1 decision path (audit §2.3), cheapest first:
    //   1. measure the drift with bounce-back (Gate 6a) — do this before writing anything;
    //   2. setTurbulentWallModel + WallModelParameters (smooth Musker) tuned to reproduce
    //      a z0=0.045 m log law at the first fluid node — a calibration, ~1 day;
    //   3. a custom rough-wall post-processor enforcing u(z1)=(u*/κ)ln(z1/z0) — the
    //      physically right answer, ~60 lines, but it must be GPU-side for the A4000.
#if defined(GROUND_WALLFUNCTION)
  #error "GROUND_WALLFUNCTION: OpenLB 1.8.1 has no z0 roughness wall function. \
wallFunctionParam has no z0 member (its fields are latticeWalldistance / vonKarman), and \
setTurbulentWallModel is a SMOOTH-wall Musker model. Implement G1 option 2 or 3 from the \
comment above before enabling this. See OPENLB_PORT_STATUS_AND_VERIFICATION.md §2.3."
#else
    sLattice.template defineDynamics<BounceBackNS>(superGeometry, MAT_GROUND);
#endif

    // porous park canopy (PorousBGK reads the POROSITY field the descriptor now carries)
    auto porousInd = superGeometry.getMaterialIndicator({MAT_POROUS});
    sLattice.template defineDynamics<PorousBGKdynamics<T,DESCRIPTOR>>(porousInd);
    {   // calibrated to the custom engine's park canopy (C_d≈0.2, LAD≈1 → perm 0.8)
        AnalyticalConst3D<T,T> por(envd("PARK_POROSITY", 0.8));
        sLattice.template defineField<POROSITY>(porousInd, por);            // CONFIRM 1.8
    }

    // (C5) SPONGE fringe: graded per-cell Smagorinsky just before the outlet (G3).
    if (nSponge > 0) {
        auto spongeInd = superGeometry.getMaterialIndicator({MAT_SPONGE});
        sLattice.template defineDynamics<ExternalSmagorinskyBGKdynamics<T,DESCRIPTOR>>(spongeInd); // CONFIRM 1.8
        int sign=0; const int axis = spongeAxis(wind_deg, sign);
        const int nAxis = (axis==0) ? nx : ny;
        const T outerEdge = (sign>0) ? T(nAxis)*dx : T(0);
        const T innerEdge = (sign>0) ? T(nAxis-nSponge)*dx : T(nSponge)*dx;
        SpongeCsF3D<T> csF(envd("CS_BULK",0.12), envd("CS_SPONGE",0.5), innerEdge, outerEdge, axis);
        sLattice.template defineField<collision::LES::SMAGORINSKY>(spongeInd, csF);   // CONFIRM 1.8
    }

    // ── domain boundaries (B4: the 1.8 declarative boundary:: API) ──
    boundary::set<boundary::InterpolatedVelocity<T,DESCRIPTOR>>(
        sLattice, superGeometry, MAT_INLET);                                 // CONFIRM 1.8
    boundary::set<boundary::InterpolatedPressure<T,DESCRIPTOR>>(
        sLattice, superGeometry, MAT_OUTLET);                                // CONFIRM 1.8
    boundary::set<boundary::FullSlip<T,DESCRIPTOR>>(
        sLattice, superGeometry, MAT_SLIP);                                  // CONFIRM 1.8

    // initial condition: rest; the inlet ramps in over the first flow-through
    auto allFluidish = superGeometry.getMaterialIndicator({MAT_FLUID,MAT_INLET,MAT_OUTLET,MAT_POROUS,MAT_SPONGE});
    AnalyticalConst3D<T,T> rho1(T(1)), u0(T(0),T(0),T(0));
    sLattice.defineRhoU(allFluidish, rho1, u0);
    sLattice.iniEquilibrium(allFluidish, rho1, u0);

    sLattice.template setParameter<descriptors::OMEGA>(converter.getLatticeRelaxationFrequency());
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
    using AblF = bridge::AblVelocityF3D<T,DESCRIPTOR,UnitConverter<T,DESCRIPTOR>>;
    const T physT = converter.getPhysTime(iT);
    const T ramp  = AblF::smoothstep(rampSteps>0 ? (T)iT/(T)rampSteps : T(1));
    AblF ablU(converter, gInlet, physT, ramp);
    sLattice.defineU(superGeometry.getMaterialIndicator({MAT_INLET}), ablU);
    // GPU: defineU writes host-side, so push the change to the device before collide.
    sLattice.setProcessingContext(ProcessingContext::Simulation);             // CONFIRM 1.8
}

// ─────────────────────────────────────────────────────────────────────────────
// (C4) divergence / NaN guard — true = healthy, false = diverged (abort the run).
// ─────────────────────────────────────────────────────────────────────────────
static bool healthy(SuperLattice<T,DESCRIPTOR>& sLattice) {
    const T maxU = sLattice.getStatistics().getMaxU();
    if (!std::isfinite(maxU) || maxU > 0.4) {                   // >0.4 LU ⇒ super-Mach / blow-up
        OstreamManager clout(std::cout,"DIVERGED");
        clout << "*** FATAL: non-finite or super-Mach state (maxU=" << maxU
              << ") — coarsen dx / lower U / raise TAU_TARGET / enable sponge, then retry ***" << std::endl;
        return false;
    }
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// exportLiveFlow — write the INSTANTANEOUS live field (NO averaging) in the project's
// 5-int format so Stage C (visualize_forward.py) reads it unchanged:
//   umean_full.f32 : ncomp=4 (ux,uy,uz,nut). "mean" here is a filename kept for viz
//   compatibility; the payload is the live snapshot the burst rides.
// ν_t is now really exported (it used to be written as a literal 0, which left Stage C's
// eddy-viscosity panels blank).
// ─────────────────────────────────────────────────────────────────────────────
void exportLiveFlow(SuperLattice<T,DESCRIPTOR>& sLattice,
                    UnitConverter<T,DESCRIPTOR> const& converter,
                    SuperGeometry<T,3>& superGeometry,
                    int nx,int ny,int nz,double dx,const std::string& outdir) {
    OstreamManager clout(std::cout,"exportLiveFlow");
    sLattice.setProcessingContext(ProcessingContext::Evaluation);            // GPU→host

    const size_t Ncell=(size_t)nx*ny*nz;
    std::vector<float> uf(4*Ncell, 0.f);
    auto IDX=[&](int x,int y,int z){ return (size_t)z*ny*nx + (size_t)y*nx + x; };

    const T cv      = converter.getConversionFactorVelocity();
    const T cnu     = converter.getConversionFactorViscosity();
    const T nuLB    = converter.getLatticeViscosity();
    auto& load = sLattice.getLoadBalancer();
    for (int iC=0; iC<load.size(); ++iC) {
        auto& block = sLattice.getBlock(iC);
        auto& bgeo  = superGeometry.getBlockGeometry(iC);
        const Vector<int,3> o = blockOriginOf(superGeometry, iC);            // (S1) fixed
        const int bnx=bgeo.getNx(), bny=bgeo.getNy(), bnz=bgeo.getNz();
        for (int x=0;x<bnx;++x) for(int y=0;y<bny;++y) for(int z=0;z<bnz;++z) {
            int X=o[0]+x, Y=o[1]+y, Z=o[2]+z;
            if (X<0||X>=nx||Y<0||Y>=ny||Z<0||Z>=nz) continue;
            auto cell = block.get(x,y,z);
            T u[3]={0,0,0};
            cell.computeU(u);
            size_t id=IDX(X,Y,Z);
            uf[id]         = (float)(u[0]*cv);
            uf[Ncell+id]   = (float)(u[1]*cv);
            uf[2*Ncell+id] = (float)(u[2]*cv);
            // ν_t from the WALE effective relaxation frequency: ν_eff = (1/ω_eff − 0.5)/3.
            T omEff = cell.template getField<EFFECTIVE_OMEGA>();             // CONFIRM 1.8
            float nut = 0.f;
            if (std::isfinite(omEff) && omEff > 1e-6) {
                T nuEffLB = (T(1)/omEff - T(0.5)) / T(3);
                nut = (float)std::max<T>(T(0), (nuEffLB - nuLB) * cnu);      // physical m²/s
            }
            uf[3*Ncell+id] = nut;
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
// acceptance gate is tests/linearity_guard.cpp — it drives THIS binary three times via
// SRC_CELLS (below) and checks Θ_{a+b} = Θ_a + Θ_b. Run it before trusting any J.
//
// §6.3 / G2 (GPU + cost): `couple` and `accumulateTheta` are still full-domain HOST sweeps;
// `injectBurst` and `deposit` now walk PRECOMPUTED cell lists (Ω is 22 812 cells and the
// deposition faces ~95 k, versus 2.63 M for a full sweep), which removes the two most
// expensive sweeps and makes the small-box CPU gates practical. This does NOT make the city
// run tractable and does NOT make any of it GPU-resident — for the A4000, `couple` must
// become NavierStokesAdvectionDiffusionVelocityCoupling via SuperLatticeCoupling, and the
// other three must become on-device post-processors. Gate Step 4 on a 40³ box, not the city.

// per-cell Stage-A inputs mirrored to the host, grid-indexed
struct ScalarInputs {
    int nx,ny,nz; double dx;
    std::vector<uint8_t> src;   // Ω source mask (source_mask.u8)
    std::vector<float>   vd;    // surface dry-deposition velocity m/s (dep_vel.f32)
    std::vector<int32_t> mat;   // material map (for wall-adjacency / outflow tests)
    inline size_t idx(int x,int y,int z)const{return (size_t)z*ny*nx+(size_t)y*nx+x;}
};

// Precomputed per-block work lists (built once; see the G2 note above).
struct BurstPlan {
    struct Cell { int bx,by,bz; T alpha; };          // block-local coords (+ deposition α)
    std::vector<std::vector<Cell>> inject;           // per local cuboid
    std::vector<std::vector<Cell>> deposit;
    long nInject=0, nDeposit=0;
};

// ─────────────────────────────────────────────────────────────────────────────
// buildBurstPlan — resolve Ω cells and deposition-face cells to block-local coordinates
// once, and precompute each deposition cell's α.
//
// (S2) THE BUG THIS REPLACES: α was computed as min(1, 8·v_d) with v_d taken straight from
// dep_vel.f32, which voxelize.h documents as **m/s**. CONTAMINANT_BC.md's closure is
// α = 8·v_d_**lb**. At dx=4 m, dt=0.05 s the conversion factor is dt/dx = 0.0125, so the
// code over-deposited by 80× (α = 1.6e-2 where it should be 2.0e-4) — everything would
// have deposited within metres of the source, nothing would have advected, and the
// exposure ranking would have been quietly destroyed while still producing a finite J.
// ─────────────────────────────────────────────────────────────────────────────
template <class SGEOM>
static BurstPlan buildBurstPlan(SGEOM& sg, const ScalarInputs& in,
                                UnitConverter<T,DESCRIPTOR> const& converter,
                                int nLocal) {
    static const int dxn[6]={1,-1,0,0,0,0},dyn[6]={0,0,1,-1,0,0},dzn[6]={0,0,0,0,1,-1};
    BurstPlan p; p.inject.resize(nLocal); p.deposit.resize(nLocal);
    for (int iC=0; iC<nLocal; ++iC) {
        auto& bg = sg.getBlockGeometry(iC);
        const Vector<int,3> o = blockOriginOf(sg, iC);
        const int bnx=bg.getNx(), bny=bg.getNy(), bnz=bg.getNz();
        for(int x=0;x<bnx;++x)for(int y=0;y<bny;++y)for(int z=0;z<bnz;++z){
            const int X=o[0]+x, Y=o[1]+y, Z=o[2]+z;
            if(X<0||X>=in.nx||Y<0||Y>=in.ny||Z<0||Z>=in.nz) continue;
            const size_t id = in.idx(X,Y,Z);
            if (in.src[id]) { p.inject[iC].push_back({x,y,z,T(0)}); ++p.nInject; }

            const int32_t mt = in.mat[id];
            if (mt!=MAT_FLUID && mt!=MAT_POROUS && mt!=MAT_SPONGE) continue;
            float vd=0.f;
            for(int d=0;d<6;++d){
                const int xx=X+dxn[d], yy=Y+dyn[d], zz=Z+dzn[d];
                if(xx<0||xx>=in.nx||yy<0||yy>=in.ny||zz<0||zz>=in.nz) continue;
                const int32_t nm=in.mat[in.idx(xx,yy,zz)];
                if(nm==MAT_WALL||nm==MAT_GROUND||nm==MAT_POROUS) vd=std::max(vd,in.vd[in.idx(xx,yy,zz)]);
            }
            if (vd<=0.f) continue;
            const T vd_lb = converter.getLatticeVelocity((T)vd);      // (S2) m/s → lattice
            const T alpha = std::min<T>(T(1), T(8)*vd_lb);
            p.deposit[iC].push_back({x,y,z,alpha}); ++p.nDeposit;
        }
    }
    return p;
}

// per-step transport operators (host loops; see the G2 note above)
namespace step4 {
using ADLat = SuperLattice<T,AD_DESCRIPTOR>;

// copy live NSE velocity (+ settling offset) into the AD VELOCITY field
template <class SGEOM>
inline void couple(ADLat& ad, SuperLattice<T,DESCRIPTOR>& ns, SGEOM& sg, T w_s_lb) {
    auto& load = ad.getLoadBalancer();
    for (int iC=0; iC<load.size(); ++iC) {
        auto& ab=ad.getBlock(iC); auto& nb=ns.getBlock(iC);
        auto& bg=sg.getBlockGeometry(iC);
        const int bnx=bg.getNx(),bny=bg.getNy(),bnz=bg.getNz();
        for(int x=0;x<bnx;++x)for(int y=0;y<bny;++y)for(int z=0;z<bnz;++z){
            T u[3]; nb.get(x,y,z).computeU(u); u[2]-=w_s_lb;               // −w_s ẑ settling
            ab.get(x,y,z).template setField<VELOCITY>(u);
        }
    }
}

// inject the burst at Ω cells during the pulse; returns mass emitted this step
inline double injectBurst(ADLat& ad, const BurstPlan& plan, T rate_lb, bool on) {
    if(!on) return 0.0; double emitted=0;
    auto& load=ad.getLoadBalancer();
    for(int iC=0;iC<load.size();++iC){ auto& ab=ad.getBlock(iC);
        for(const auto& c : plan.inject[iC]){
            auto cell=ab.get(c.bx,c.by,c.bz);
            cell.defineRho(cell.computeRho()+rate_lb);
            emitted+=rate_lb; }
    }
    return emitted;
}

// deposition sink at fluid cells adjacent to a wall/ground/park face; returns mass deposited
inline double deposit(ADLat& ad, const BurstPlan& plan) {
    double dep=0; auto& load=ad.getLoadBalancer();
    for(int iC=0;iC<load.size();++iC){ auto& ab=ad.getBlock(iC);
        for(const auto& c : plan.deposit[iC]){
            auto cell=ab.get(c.bx,c.by,c.bz);
            const T rho=cell.computeRho();
            const T removed=(c.alpha/T(8))*rho;      // α/8·C half-way bounce-back closure
            cell.defineRho(rho-removed);
            cell.template setField<DEPOSIT>(cell.template getField<DEPOSIT>()+removed);
            dep+=removed; }
    }
    return dep;
}

// accumulate Θ += C·dt (on the THETA field) and return current airborne mass ΣC
template <class SGEOM>
inline double accumulateTheta(ADLat& ad, SGEOM& sg, const ScalarInputs& in, T dt_lb) {
    double airborne=0; auto& load=ad.getLoadBalancer();
    for(int iC=0;iC<load.size();++iC){ auto& ab=ad.getBlock(iC);
        auto& bg=sg.getBlockGeometry(iC);
        const Vector<int,3> o = blockOriginOf(sg, iC);                     // (S1) fixed
        const int bnx=bg.getNx(),bny=bg.getNy(),bnz=bg.getNz();
        for(int x=0;x<bnx;++x)for(int y=0;y<bny;++y)for(int z=0;z<bnz;++z){
            int X=o[0]+x,Y=o[1]+y,Z=o[2]+z; if(X<0||X>=in.nx||Y<0||Y>=in.ny||Z<0||Z>=in.nz)continue;
            int32_t mt=in.mat[in.idx(X,Y,Z)]; if(mt==MAT_VOID||mt==MAT_WALL||mt==MAT_GROUND)continue;
            auto c=ab.get(x,y,z); T C=c.computeRho();
            c.template setField<THETA>(c.template getField<THETA>()+C*dt_lb);
            airborne+=C; }
    }
    return airborne;
}

// gather a THETA or DEPOSIT field to a host float array (grid-indexed) for export
template <class SGEOM>
inline void gatherField(ADLat& ad, SGEOM& sg, const ScalarInputs& in,
                        int which /*0=THETA 1=DEPOSIT*/, std::vector<float>& out) {
    out.assign((size_t)in.nx*in.ny*in.nz, 0.f); auto& load=ad.getLoadBalancer();
    for(int iC=0;iC<load.size();++iC){ auto& ab=ad.getBlock(iC);
        auto& bg=sg.getBlockGeometry(iC);
        const Vector<int,3> o = blockOriginOf(sg, iC);                     // (S1) fixed
        const int bnx=bg.getNx(),bny=bg.getNy(),bnz=bg.getNz();
        for(int x=0;x<bnx;++x)for(int y=0;y<bny;++y)for(int z=0;z<bnz;++z){
            int X=o[0]+x,Y=o[1]+y,Z=o[2]+z; if(X<0||X>=in.nx||Y<0||Y>=in.ny||Z<0||Z>=in.nz)continue;
            auto c=ab.get(x,y,z);
            out[in.idx(X,Y,Z)] = which==0 ? (float)c.template getField<THETA>()
                                          : (float)c.template getField<DEPOSIT>(); }
    }
}
} // namespace step4

// ─────────────────────────────────────────────────────────────────────────────
// prepareScalarLattice — AD dynamics + BCs + the AD relaxation rate.
//
// (S3) THE BUG THIS REPLACES: omegaAD was `1.0; // placeholder` and then `(void)omegaAD` —
// literally discarded. At ω=1 the D3Q7 diffusivity is D=(1/ω−0.5)/4 = 0.125 lu ≈ 40 m²/s
// physical, ~100× the turbulent diffusivity intended: the plume would have been pure
// diffusion with the wind barely mattering. ω is now derived from D_eff and APPLIED.
//
// D_eff = D_mol + ν_t/Sc_t. ν_t is spatially varying under WALE, so the physically complete
// version is a per-cell ω. Per the audit's own sequencing this starts CONSTANT (using the
// converter's viscosity as the ν_t scale) — get the constant case through Gate 7a/7b first,
// then switch to per-cell. The per-cell upgrade is sketched at the end of this function.
// ─────────────────────────────────────────────────────────────────────────────
T adOmegaFor(UnitConverter<T,DESCRIPTOR> const& converter, T D_mol, T Sc_t) {
    const T nuPhys  = converter.getPhysViscosity();
    const T D_eff   = D_mol + nuPhys / Sc_t;                       // physical m²/s
    const T dt      = converter.getPhysDeltaT();
    const T dxP     = converter.getPhysDeltaX();
    const T D_lb    = D_eff * dt / (dxP*dxP);
    return T(1) / (T(4)*D_lb + T(0.5));                            // D3Q7: D=(1/ω−0.5)/4
}

void prepareScalarLattice(SuperLattice<T,AD_DESCRIPTOR>& adLattice,
                          UnitConverter<T,DESCRIPTOR> const& converter,
                          SuperGeometry<T,3>& superGeometry, T D_mol, T Sc_t) {
    OstreamManager clout(std::cout,"prepareScalarLattice");
    const T omegaAD = adOmegaFor(converter, D_mol, Sc_t);

    auto fluidish = superGeometry.getMaterialIndicator({MAT_FLUID,MAT_INLET,MAT_OUTLET,MAT_POROUS,MAT_SPONGE});
    adLattice.template defineDynamics<AD_DYNAMICS>(fluidish);
    adLattice.template defineDynamics<NoDynamicsAD>(superGeometry, MAT_VOID);
    // zero-flux walls: bounce-back on buildings + ground (deposition handled by the sink op)
    adLattice.template defineDynamics<BounceBackAD>(superGeometry, MAT_WALL);
    adLattice.template defineDynamics<BounceBackAD>(superGeometry, MAT_GROUND);
    // top/lateral: zero-flux
    adLattice.template defineDynamics<BounceBackAD>(superGeometry, MAT_SLIP);
    // outlet: advective outflow / zero inflow
    boundary::set<boundary::ZeroDistribution<T,AD_DESCRIPTOR>>(
        adLattice, superGeometry, MAT_OUTLET);                              // CONFIRM 1.8

    AnalyticalConst3D<T,T> c0(T(0)), u0(T(0),T(0),T(0));
    adLattice.defineRhoU(fluidish, c0, u0);
    adLattice.iniEquilibrium(fluidish, c0, u0);
    adLattice.template setParameter<descriptors::OMEGA>(omegaAD);           // (S3) APPLIED
    adLattice.initialize();

    clout << "AD lattice: D_mol=" << D_mol << " Sc_t=" << Sc_t
          << " -> D_eff=" << (D_mol + converter.getPhysViscosity()/Sc_t) << " m^2/s"
          << ", omega_AD=" << omegaAD << std::endl;
    // PER-CELL UPGRADE (do this after Gate 7b passes with the constant case):
    //   read EFFECTIVE_OMEGA from the NSE cell -> ν_t -> D_eff(x) -> ω(x), and write it
    //   with adLattice.defineField<descriptors::OMEGA>(indicator, functor) instead of
    //   setParameter. Re-run Gate 7a/7b afterwards: a per-cell ω changes the mass budget.
}

// write Θ and deposition fields in the project 5-int format for Stage C (J = ⟨w,Θ⟩/|Ω|)
static void writeField5(const std::string& fn,const std::vector<float>& d,int nx,int ny,int nz,double dx){
    FILE* f=fopen(fn.c_str(),"wb"); if(!f)return; int h[5]={nx,ny,nz,(int)std::lround(dx*1000.0),1};
    fwrite(h,sizeof(int),5,f); fwrite(d.data(),sizeof(float),d.size(),f); fclose(f);
}

// ─────────────────────────────────────────────────────────────────────────────
// SRC_CELLS — override Ω with an explicit cell list, "x,y,z;x,y,z;...".
// This is what makes the §6.2 linearity guard runnable: tests/linearity_guard.cpp drives
// this binary three times (release {a}, {b}, {a,b}) and checks Θ_{a+b} = Θ_a + Θ_b.
// ─────────────────────────────────────────────────────────────────────────────
static long applySrcOverride(ScalarInputs& in, const char* spec) {
    std::fill(in.src.begin(), in.src.end(), (uint8_t)0);
    long n=0; int x,y,z; const char* p=spec;
    while (*p) {
        if (sscanf(p, "%d,%d,%d", &x,&y,&z) == 3) {
            if (x>=0&&x<in.nx&&y>=0&&y<in.ny&&z>=0&&z<in.nz) { in.src[in.idx(x,y,z)]=1; ++n; }
            else fprintf(stderr,"[SRC_CELLS] out of range: %d,%d,%d\n",x,y,z);
        }
        const char* semi = strchr(p, ';');
        if (!semi) break;
        p = semi+1;
    }
    return n;
}

// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    olb::initialize(&argc, &argv);                                          // (B1)
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

    // ── unit converter ───────────────────────────────────────────────────────
    // dt is set by the Mach constraint (uLB ≈ 0.05), which at urban scale is the binding
    // one. A converter built from resolution+τ instead (the obvious "derive dt" idiom)
    // cannot be used here: with molecular ν=1.5e-5 and dx=4 m, τ≈0.51 implies dt≈3555 s
    // and uLB≈3555 — wildly super-Mach. So dt comes from Mach, and τ is pinned away from
    // 0.5 by an explicit SGS VISCOSITY FLOOR — exactly the NU_FLOOR mechanism the custom
    // engine used (CHANGES.md Fix 1: it ran at ν_lb ≈ 1e-3…5e-3, i.e. τ ≈ 0.503…0.515).
    // WALE adds its own ν_sgs on top; this floor is only the stability backstop.
    const T U_INLET = envd("U_INLET", 4.0), Z_REF = envd("ABL_ZREF", 4.0);
    const T nu_mol  = envd("NU_MOL", 1.5e-5);
    const int RES   = envi("RESOLUTION", 1);
    const T dxLat   = (T)dx / RES;
    const T uLBtgt  = envd("ULB_TARGET", 0.05);
    const T tauTgt  = envd("TAU_TARGET", 0.505);          // ≥0.505: the preflight gate
    const T dtPhys  = uLBtgt * dxLat / U_INLET;
    const T nuFloor = ((tauTgt - T(0.5))/T(3)) * dxLat*dxLat / dtPhys;
    const T nuUsed  = std::max(nu_mol, nuFloor);
    const T charL   = (T)(nz*dx);
    const T ReEff   = U_INLET * charL / nuUsed;
    if (nuUsed > nu_mol)
        clout << "viscosity floor active: nu=" << nuUsed << " m^2/s (molecular " << nu_mol
              << ") to hold tau=" << tauTgt << std::endl;

    UnitConverter<T,DESCRIPTOR> converter(
        /*physDeltaX*/ dxLat, /*physDeltaT*/ dtPhys,
        /*charPhysLength*/ charL, /*charPhysVelocity*/ U_INLET,
        /*physViscosity*/ nuUsed, /*physDensity*/ (T)1.2);
    converter.print();

    const T D_mol=envd("D_MOL",1e-5), Sc_t=envd("SC_T",0.7);
    const T omegaAD = adOmegaFor(converter, D_mol, Sc_t);

    // (C1) preflight — refuse to run outside the stable regime unless FORCE=1
    if (!preflight(converter, omegaAD, ReEff) && !envi("FORCE",0)) {
        clout<<"aborting on preflight"<<std::endl; return 4; }

    // ── OpenLB geometry over an nx×ny×nz cuboid (1:1 with the imported map) ──
    // (B2 + §2.4) CuboidGeometry3D is CuboidDecomposition3D in 1.8, and the EXPLICIT-EXTENT
    // constructor is used deliberately: building from an IndicatorCuboid3D may yield nx or
    // nx+1 nodes per axis, and every index downstream assumes exact 1:1 with the imported
    // map. Stating the extent in cells removes the ambiguity rather than hoping.
    Vector<T,3> origin(0,0,0);
    Vector<int,3> extentCells(nx,ny,nz);
#ifdef PARALLEL_MODE_MPI
    const int nCuboids = singleton::mpi().getSize();
#else
    const int nCuboids = 1;
#endif
    CuboidDecomposition3D<T> cuboidDecomposition(origin, (T)dx, extentCells, nCuboids); // CONFIRM 1.8
    HeuristicLoadBalancer<T> loadBalancer(cuboidDecomposition);
    // overlap 3: the 1.8 default, and what interpolated boundaries + WALE velocity
    // gradients want (it was 2, which is too thin for both).
    SuperGeometry<T,3> superGeometry(cuboidDecomposition, loadBalancer, 3);
    bridge::stampSuperGeometry(superGeometry, mat);   // geometry bridge + sponge, applied
    superGeometry.communicate();                      // §2.4: halo materials, else MPI-wrong
    superGeometry.updateStatistics();
    superGeometry.getStatistics().print();

    // ── GATE 5 (audit §3 Phase 4): OpenLB's own per-material counts must equal Stage A's.
    // This single check catches the getOrigin() unit bug, any nx-vs-nx+1 off-by-one, and
    // any overlap/padding indexing error — the three things most likely to silently poison
    // everything downstream. Expect fluid to be short by exactly the carved sponge cells.
    {
        std::vector<long> host(9,0);
        for (auto v : mat.data) if (v>=0 && v<9) host[v]++;
        clout << "── geometry round-trip (Gate 5) ──" << std::endl;
        bool ok=true;
        for (int m=0;m<=8;++m) {
            const long olbN = superGeometry.getStatistics().getNvoxel(m);
            if (host[m]==0 && olbN==0) continue;
            const bool match = (olbN==host[m]);
            if(!match) ok=false;
            clout << "  MAT " << m << "  stageA=" << host[m] << "  olb=" << olbN
                  << (match ? "  [OK]" : "  [MISMATCH]") << std::endl;
        }
        clout << "Gate 5: " << (ok?"PASS":"FAIL — do NOT proceed; the map arrived scrambled") << std::endl;
        if (!ok && !envi("FORCE",0)) return 5;
    }

    // ── verified ABL/RFG inlet ──
    gInlet.z0 = envd("ABL_Z0", 0.045); gInlet.d = 0.0; gInlet.wind_angle = WIND_DEG*M_PI/180.0;
    gInlet.L_turb = envd("ABL_LTURB", 20.0); gInlet.n_modes = envi("ABL_NMODES", 100);
    gInlet.sigma_u_ratio=2.5; gInlet.sigma_v_ratio=1.9; gInlet.sigma_w_ratio=1.25;
    gInlet.init(U_INLET, Z_REF, (unsigned)envi("ABL_SEED", 1000));
    clout << "ABL inlet u*=" << gInlet.u_star
          << " (verified: mean 0.11%, sigma 0.35%, div 5.6%)" << std::endl;

    // ── lattice ──
    SuperLattice<T,DESCRIPTOR> sLattice(superGeometry);
    prepareLattice(sLattice, converter, superGeometry, nSponge, WIND_DEG, nx, ny, dx);

    // ── develop the LIVE turbulent flow (no averaging) ──
    const int SPIN_FT   = envi("SPINUP_FT", 3);
    const int stepsPerFT= (int)std::llround((nx*dx)/U_INLET / converter.getPhysDeltaT());
    const int MAX_STEPS = envi("MAX_STEPS", SPIN_FT*stepsPerFT);
    const int rampSteps = stepsPerFT;                      // (C3) ramp over 1 flow-through
    const int CHECK     = envi("CHECK_EVERY", 200);        // (C4) divergence-guard cadence
    const int ADM_EVERY = envi("ADM_EVERY", 0);            // (C6) 0 = ADM off
    clout << "live spin-up: " << SPIN_FT << " flow-throughs = " << MAX_STEPS
          << " steps (" << stepsPerFT << "/FT), ramp " << rampSteps << ", check " << CHECK << std::endl;

    using AblF = bridge::AblVelocityF3D<T,DESCRIPTOR,UnitConverter<T,DESCRIPTOR>>;
    for (int iT=0; iT<MAX_STEPS; ++iT) {
        setBoundaryValues(sLattice, converter, superGeometry, iT, rampSteps);   // C3
        sLattice.collideAndStream();

        if (ADM_EVERY>0 && iT%ADM_EVERY==0) {                                   // C6 optional ADM
            // CONFIRM 1.8: SuperLatticeADM3D<T,DESCRIPTOR> admF(sLattice, adm_sigma, adm_order);
            //             admF.execute(superGeometry, MAT_FLUID);
        }
        if (iT%CHECK==0) {
            if (!healthy(sLattice)) { return 2; }                               // C4
            clout << "iT=" << iT << " t=" << converter.getPhysTime(iT)
                  << "s ramp=" << AblF::smoothstep(rampSteps>0?(T)iT/(T)rampSteps:T(1)) << std::endl;
        }
    }
    if (!healthy(sLattice)) return 2;

    // ── snapshot the developed LIVE field for Stage C ──
    exportLiveFlow(sLattice, converter, superGeometry, nx, ny, nz, dx, OUT);
    if (envi("AIRFLOW_ONLY",0)) {
        clout << "AIRFLOW_ONLY=1 — stopping after the airflow stage (Gate 6a/6b)." << std::endl;
        return 0;
    }

    // ════════════════════════ STEP 4: live-flow burst transport ════════════════════════
    ScalarInputs sin; sin.nx=nx; sin.ny=ny; sin.nz=nz; sin.dx=dx; sin.mat=mat.data;
    { bridge::GridField<uint8_t> sm; if(!bridge::load_source_mask(GEOM+"/source_mask.u8",sm)){clout<<"no source_mask"<<std::endl;return 2;} sin.src=sm.data; }
    { bridge::GridField<float> dv; if(bridge::read_grid(GEOM+"/dep_vel.f32",dv)) sin.vd=dv.data; else sin.vd.assign((size_t)nx*ny*nz,0.f); }
    if (const char* spec = getenv("SRC_CELLS")) {
        const long n = applySrcOverride(sin, spec);
        clout << "SRC_CELLS override: releasing from " << n << " explicit cell(s)" << std::endl;
    }
    long nOmega=0; for(auto v:sin.src) if(v) ++nOmega;
    if (nOmega==0) { clout << "*** no source cells — nothing to release ***" << std::endl; return 6; }

    SuperLattice<T,AD_DESCRIPTOR> adLattice(superGeometry);
    prepareScalarLattice(adLattice, converter, superGeometry, D_mol, Sc_t);

    const BurstPlan plan = buildBurstPlan(superGeometry, sin, converter,
                                          superGeometry.getLoadBalancer().size());
    clout << "burst plan: " << plan.nInject << " source cells, "
          << plan.nDeposit << " deposition-face cells" << std::endl;

    const T   dt        = converter.getPhysDeltaT();
    const T   PULSE_S   = envd("PULSE_S", 2.0);                         // burst duration
    const T   rate_lb   = envd("Q_RATE", 1.0);                          // per-Ω-cell emission/step
    const T   w_s_lb    = converter.getLatticeVelocity(envd("W_SETTLE", 0.0));  // settling (0 = gas)
    const T   CLEAR     = envd("CLEAR_FRAC", 0.01);                     // stop at 99% clearance
    const int MAXB      = envi("MAX_BURST_STEPS", 20*stepsPerFT);
    const int TS_EVERY  = envi("TS_EVERY", 200);
    double emit=0, dep=0, peakAir=0, air=0; int endStep=MAXB;
    FILE* ts=fopen((OUT+"/exposure_timeseries.csv").c_str(),"w");
    if(ts) fprintf(ts,"step,t_s,airborne,deposited,emitted,outflow\n");

    for (int iB=0; iB<MAXB; ++iB) {
        setBoundaryValues(sLattice, converter, superGeometry, MAX_STEPS+iB, 0);  // live, no ramp
        sLattice.collideAndStream();                                             // NSE (live)
        sLattice.setProcessingContext(ProcessingContext::Evaluation);            // host reads u
        step4::couple(adLattice, sLattice, superGeometry, w_s_lb);               // advect on live u
        adLattice.collideAndStream();                                            // AD
        bool pulseOn = (iB*dt) < PULSE_S;
        emit += step4::injectBurst(adLattice, plan, rate_lb, pulseOn);
        dep  += step4::deposit(adLattice, plan);
        air   = step4::accumulateTheta(adLattice, superGeometry, sin, dt);
        peakAir = std::max(peakAir, air);
        if(!std::isfinite(air)){ OstreamManager c(std::cout,"DIVERGED"); c<<"AD non-finite at burst step "<<iB<<std::endl; if(ts)fclose(ts); return 2; }
        if(iB%TS_EVERY==0 && ts){ double outfl=emit-dep-air; fprintf(ts,"%d,%.4f,%.6e,%.6e,%.6e,%.6e\n",iB,iB*dt,air,dep,emit,outfl); fflush(ts);
            clout<<"burst iB="<<iB<<" t="<<iB*dt<<"s air="<<air<<" dep="<<dep<<" emit="<<emit<<std::endl; }
        if(!pulseOn && peakAir>0 && air < CLEAR*peakAir){ endStep=iB; break; }   // clearance
    }
    if(ts) fclose(ts);

    // export Θ = ∫C dt and the deposition map for Stage C (J = ⟨w,Θ⟩/|Ω|)
    std::vector<float> theta, deposition;
    step4::gatherField(adLattice, superGeometry, sin, 0, theta);
    step4::gatherField(adLattice, superGeometry, sin, 1, deposition);
    writeField5(OUT+"/theta.f32",      theta,      nx,ny,nz,dx);
    writeField5(OUT+"/deposition.f32", deposition, nx,ny,nz,dx);

    // §2.4: mass_drained was emit − dep, which double-counts the still-airborne mass
    // (the CSV always had it right). The budget is emit = drained + dep + air.
    const double drained = emit - dep - air;
    const double closure = (emit>0) ? std::fabs(emit - (drained+dep+air))/emit : 0.0;
    { FILE* mf=fopen((OUT+"/meta_flow.txt").c_str(),"w"); if(mf){
        fprintf(mf,"grid %d %d %d\ndx_m %.4f\nomega_cells %ld\n",nx,ny,nz,dx,nOmega);
        fprintf(mf,"burst_steps %d\nmass_emitted %.6e\nmass_deposited %.6e\nmass_airborne %.6e\nmass_drained %.6e\n",
                endStep,emit,dep,air,drained);
        fprintf(mf,"budget_closure %.3e\ndeposited_frac %.4f\ntheta_layout 5xint32[nx,ny,nz,dx*1000,1]\n",
                closure, emit>0?dep/emit:0.0);
        fprintf(mf,"# Stage C: J = (1/omega_cells) * sum_x receptor_w(x) * theta(x)\n"); fclose(mf);} }

    clout << "urban_flow COMPLETE — live airflow + burst transport. Wrote umean_full.f32, "
          << "theta.f32, deposition.f32, exposure_timeseries.csv to " << OUT << "/  (Omega="
          << nOmega << ", emitted=" << emit << ", deposited=" << dep
          << ", airborne=" << air << ", drained=" << drained << ")" << std::endl;
    return 0;
}
