// lbm_solver.cpp — Host logic: solver loop, convergence, unit conversion
// Compile: g++ -O3 -std=c++17 -c lbm_solver.cpp -o solver.o

#include <cstdlib>
#include "lbm_gpu.h"
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <vector>

namespace lbm {

// ─── Physical ↔ lattice unit conversion ──────────────────────────────
// LBM runs in lattice units (Δx=Δt=1). We fix the lattice inlet speed via a
// low Mach number so compressibility error stays O(Ma²): u_lb = Ma·c_s with
// c_s=1/√3. The physical timestep then follows from matching the inlet speed,
//   dt = u_lb·Δx_phys / U_inlet      (so U_inlet [m/s] ↔ u_lb [LU]),
// and any diffusivity converts as ν_lb = ν_phys·dt/Δx²  (length²/time scaling).
static void unit_convert(Solver::Impl& I) {
    const float Ma = 0.1f;
    const float cs = 1.f / sqrtf(3.f);
    I.u_lb    = Ma * cs;                                           // ~0.0577
    I.dt_phys = I.u_lb * I.cfg.cell_size / I.cfg.U_inlet;         // s per LB step
    I.nu_lb   = I.cfg.nu_phys * I.dt_phys / (I.cfg.cell_size * I.cfg.cell_size);
    I.D_lb    = I.cfg.D_mol  * I.dt_phys / (I.cfg.cell_size * I.cfg.cell_size);

    printf("[LBM] u_lb=%.5f  nu_lb=%.2e  D_lb=%.2e  dt=%.4fs\n",
           I.u_lb, I.nu_lb, I.D_lb, I.dt_phys);
    // At practical resolutions ν_lb, D_lb ≪ 1e-3, so the stability floor (τ≥0.503)
    // dominates: the effective Reynolds/Péclet number is set by the floor, not by
    // the physical value. Consequence: wind *speed* barely changes the flow
    // pattern (only the floor-set effective Re does) — wind *direction* is the
    // meaningful optimization variable.
    // Stability floor. At tau->0.5 (nu_lb->0) the LBM has almost no dissipative
    // margin and a turbulent inlet will eventually drive a cell unstable over a
    // long run (observed: stable ~30k steps at nu=1e-3/tau=0.503, diverged by
    // ~100k). nu_floor=1e-2 -> tau=0.53 gives a robust margin. The regime is
    // already floor-dominated (see ARCHITECTURE.md §8), so this lowers the
    // effective Re further but does not change the qualitative regime; the cost
    // is modestly more numerical diffusion. Lower toward ~5e-3 only if a long
    // run stays stable. D_floor left at 1e-3 (the scalar rode the flow; it did
    // not diverge independently) — raise it too if the scalar ever blows up.
    // Floors are runtime-overridable (NU_FLOOR / D_FLOOR env vars) so one binary
    // can sweep the effective Reynolds number for the Re-independence test
    // without recompiling. Defaults: nu=1e-2 (tau=0.53, the stable production
    // value); see the stability note above.
    const char* envnu = std::getenv("NU_FLOOR");
    const char* envd  = std::getenv("D_FLOOR");
    float nu_floor = envnu ? (float)std::atof(envnu) : 1e-2f;
    float D_floor  = envd  ? (float)std::atof(envd)  : 1e-3f;
    // APPLY the floor to the values the kernels actually use (previously this was
    // only printed as a diagnostic while I.nu_lb kept the unfloored physical
    // value ~4e-8, so the kernel's own hardcoded 1e-3 floor governed and NU_FLOOR
    // had no effect — the bug that made the tau-sweep produce bit-identical runs).
    I.nu_lb = std::max(I.nu_lb, nu_floor);
    I.D_lb  = std::max(I.D_lb,  D_floor);
    float tau_f_eff = 3 * I.nu_lb + .5f;
    float tau_c_eff = 3 * I.D_lb  + .5f;
    printf("[LBM] Effective floors: tau_f=%.4f  tau_c=%.4f  (nu_lb=%.2e  D_lb=%.2e)\n",
           tau_f_eff, tau_c_eff, I.nu_lb, I.D_lb);

    // ── Collision mode (env COLLISION=mrt|reg) ──────────────────────────────
    // reg = projected-regularized MRT: ghost moments relaxed to equilibrium each
    // step, filtering the aliased non-hydrodynamic modes that drive the τ→0.5
    // (low-ν) blow-up. This is what lets NU_FLOOR drop below ~1e-2 without NaN.
    { const char* envc = std::getenv("COLLISION");
      // env OVERRIDES if set; otherwise KEEP the driver's cfg.collision_mode (set
      // explicitly by the driver, or the struct default reg). The old `else→reg`
      // here SILENTLY DISCARDED a driver-set HRR whenever COLLISION was unset —
      // the Phase-0 "ran reg instead of HRR" bug. Do not reintroduce it.
      if      (envc && (envc[0]=='m'||envc[0]=='M')) I.cfg.collision_mode = 0;  // mrt
      else if (envc && (envc[0]=='h'||envc[0]=='H')) I.cfg.collision_mode = 2;  // hrr
      else if (envc && (envc[0]=='r'||envc[0]=='R')) I.cfg.collision_mode = 1;  // reg
      else if (I.cfg.collision_mode < 0 || I.cfg.collision_mode > 2)
                                                     I.cfg.collision_mode = 1;  // guard uninitialised
      const char* envs = std::getenv("HRR_SIGMA");
      float s = envs ? (float)std::atof(envs) : 0.98f;
      I.cfg.hrr_sigma = (s<0.f)?0.f:(s>1.f?1.f:s); }
    // Live-flow forward scalar scheme. The production forward-dose driver sets
    // QUICK (3) as its default (the linear 3rd-order upstream scheme that matches
    // the adjoint operator); other drivers keep their explicit choice. An explicit
    // SCALAR env always overrides, so a run can be forced onto any scheme without
    // silently changing drivers (e.g. the numdiff cross-check) that chose one on
    // purpose.
    { const char* envq = std::getenv("SCALAR");
      if      (envq && (envq[0]=='v'||envq[0]=='V')) I.cfg.scalar_advection = 0; // van Leer TVD
      else if (envq && (envq[0]=='u'||envq[0]=='U')) I.cfg.scalar_advection = 1; // upwind
      else if (envq && (envq[0]=='c'||envq[0]=='C')) I.cfg.scalar_advection = 2; // central
      else if (envq && (envq[0]=='q'||envq[0]=='Q')) I.cfg.scalar_advection = 3; // QUICK
    }
    { const char* nm[4]={"van Leer TVD","first-order upwind","central","QUICK (linear 3rd-order upstream)"};
      int a=I.cfg.scalar_advection; if(a>=0&&a<4)
        printf("[LBM] Scalar advection: %s%s\n", nm[a], a==3?" [live-flow forward production]":""); }
    if(I.cfg.collision_mode==2)
        printf("[LBM] Collision: HRR (hybrid recursive-regularized, sigma=%.3f%s)\n",
               I.cfg.hrr_sigma, I.cfg.hrr_sigma>=1.f ? " => pure RR" : "");
    else
        printf("[LBM] Collision: %s\n",
               I.cfg.collision_mode ? "regularized MRT (ghost->equilibrium)"
                                    : "MRT (d'Humieres tuned ghost rates)");

    // ── WALE subgrid constant (env CW; default = Config value, 0.325) ───────
    // Cw scales ν_t ∝ Cw², i.e. the absolute SGS dissipation level (it does NOT
    // change the wake/channel ν_t ratio). Exposed for the WS2 sweep; unset ⇒
    // unchanged behaviour. 0.325 sits at the low end of the published 0.3–0.6
    // range (Nicoud & Ducros 1999); higher Cw adds SGS dissipation and stability.
    { const char* envcw = std::getenv("CW");
      if(envcw){ float v=(float)std::atof(envcw); if(v>=0.f) I.cfg.Cw=v; } }
    printf("[LBM] WALE Cw = %.3f\n", I.cfg.Cw);

    // ── Rough-wall (log-law) floor model (env WALL_MODEL=1) ─────────────────
    // Replaces no-slip floor bounce-back with a moving-wall bounce-back at the
    // ABL log-law slip velocity, so the ground sustains the inlet profile
    // (fixes ABL horizontal-homogeneity drift; Blocken et al. 2007, Atmos.Env.).
    // u* is derived to MATCH the ABL inlet: U(zref)=U_inlet ⇒ u*=κU_inlet/ln((zref+z0)/z0).
    { const char* envw = std::getenv("WALL_MODEL");
      if(envw) I.cfg.wall_model = (std::atoi(envw) != 0) ? 1 : 0;   // env overrides; else keep driver/default
    }
    I.u_wallX = I.u_wallY = 0.f;
    if (I.cfg.wall_model) {
        const double kappa = 0.41;
        double z0   = I.cfg.abl_z0   > 0 ? I.cfg.abl_z0   : 0.7;
        double zref = I.cfg.abl_zref > 0 ? I.cfg.abl_zref : 40.0;
        double ustar = kappa * I.cfg.U_inlet / std::log((zref + z0) / z0);
        double zwall = 0.5 * I.cfg.cell_size;                        // bounce-back plane height
        double Uwall = (ustar / kappa) * std::log((zwall + z0) / z0);// log-law at the floor plane
        double uwall_lb = Uwall * ((double)I.u_lb / I.cfg.U_inlet);  // physical m/s → lattice
        double a = I.cfg.wind_angle;
        I.u_wallX = (float)(uwall_lb * std::cos(a));
        I.u_wallY = (float)(uwall_lb * std::sin(a));
        printf("[LBM] Wall model: rough-wall floor slip u_wall=%.4e LU  (z0=%.2f m, u*=%.3f m/s)\n",
               uwall_lb, z0, ustar);
    }

    // ── Gravitational settling velocity (Stokes + Cunningham slip) ──
    // w_s = (ρ_p - ρ_air)·g·d_p²·Cc / (18·μ),  Cc = 1 + Kn(1.257 + 0.4 e^{-1.1/Kn})
    // Kn = 2λ/d_p.  μ = ν_air·ρ_air keeps consistency with cfg.nu_phys.
    float w_s = 0.f;
    float dp  = I.cfg.particle_diam;
    if (dp > 0.f) {
        const float rho_air = 1.204f;     // kg/m³ at 20 °C
        const float g       = 9.81f;      // m/s²
        const float lambda  = 6.6e-8f;    // mean free path of air (m)
        float mu  = I.cfg.nu_phys * rho_air;            // dynamic viscosity (Pa·s)
        float Kn  = 2.f * lambda / dp;
        float Cc  = 1.f + Kn * (1.257f + 0.4f * expf(-1.1f / Kn));
        w_s = (I.cfg.particle_density - rho_air) * g * dp * dp * Cc / (18.f * mu);
        if (w_s < 0.f) w_s = 0.f;         // buoyant particles don't settle here
    }
    I.w_settle_lb = w_s * I.dt_phys / I.cfg.cell_size;   // (m/s) → LU
    if (dp > 0.f)
        printf("[LBM] Settling: d_p=%.3g m, rho_p=%.0f -> w_s=%.4g m/s (%.3e LU)\n",
               dp, I.cfg.particle_density, w_s, I.w_settle_lb);
    else
        printf("[LBM] Settling: disabled (passive tracer)\n");
}

// ─── Wind direction → inlet/outlet faces ─────────────────────────────
// Decompose the inlet velocity u_in = u_lb·(cosθ, sinθ, 0). The inlet is placed
// on the domain face the wind blows FROM (the dominant axis of u_in), the outlet
// on the opposite face; the off-axis component is carried as a skewed inflow.
static void wind_setup(Solver::Impl& I) {
    float a = I.cfg.wind_angle;
    I.uInX = I.u_lb * cosf(a);
    I.uInY = I.u_lb * sinf(a);
    I.uInZ = 0.f;
    // Legacy single-inlet designation (kept for the log line, source_cell, and the
    // WIND_BC=legacy code path): the dominant axis is the inlet, its opposite the outlet.
    if (fabsf(cosf(a)) >= fabsf(sinf(a))) {        // wind mostly along x
        I.inAx = 0;  I.inSide = (cosf(a) > 0) ? 0 : 1;
    } else {                                        // wind mostly along y
        I.inAx = 1;  I.inSide = (sinf(a) > 0) ? 0 : 1;
    }
    I.outAx = I.inAx;  I.outSide = 1 - I.inSide;

    // ── Per-face classification: dot(u_mean, inward-normal) sign per vertical face ──
    // −x normal=+x̂ (flux=+uInX); +x normal=−x̂ (−uInX); −y (+uInY); +y (−uInY).
    // >0 ⇒ INLET, <0 ⇒ OUTFLOW, ≈0 ⇒ SYMMETRY (the aligned case keeps ±y as symmetry
    // planes exactly). This is the fix for the oblique-wind corner blow-up: it makes
    // −y a genuine inlet (with a turbulent fetch) instead of a reflecting wall.
    const float eps = 1e-6f * I.u_lb;
    auto classify = [&](float flux) -> int {
        return (flux > eps) ? BC_INLET : (flux < -eps) ? BC_OUTFLOW : BC_SYM;
    };
    I.face_bc[0] = classify( I.uInX);   // −x
    I.face_bc[1] = classify(-I.uInX);   // +x
    I.face_bc[2] = classify( I.uInY);   // −y
    I.face_bc[3] = classify(-I.uInY);   // +y

    // WIND_BC=legacy forces the original single-inlet + ±y-symmetry kernel path
    // (for reproducing the oblique divergence / the exp-#6 comparison). Default = per-face.
    const char* wb = std::getenv("WIND_BC");
    I.per_face = (wb && (wb[0]=='l' || wb[0]=='L')) ? 0 : 1;   // "legacy"
}

// Map the physical source location (m) to a cell index. z is clamped to ≥1 so
// the source never lands in the z=0 ground plane (it would not be injected there).
static int source_cell(const Solver::Impl& I) {
    int sx = std::clamp((int)(I.cfg.source_x / I.cfg.cell_size), 0, I.cfg.nx - 1);
    int sy = std::clamp((int)(I.cfg.source_y / I.cfg.cell_size), 0, I.cfg.ny - 1);
    int sz = std::clamp(std::max(1, (int)(I.cfg.source_z / I.cfg.cell_size)), 1, I.cfg.nz - 1);
    return sz * I.cfg.ny * I.cfg.nx + sy * I.cfg.nx + sx;
}

// Refill the inlet-plane velocity buffer for the current physical time. The
// generator returns physical m/s; we convert to lattice units (u_lb <-> U_inlet)
// here so the kernel can write the equilibrium directly. Backend-agnostic: the
// CPU kernel reads inlet_plane directly; the CUDA path uploads it (see .cu).
static void fill_inlet_plane(Solver::Impl& I, double t_phys) {
    if (I.cfg.inlet_profile != 1) return;
    const int nx=I.cfg.nx, ny=I.cfg.ny, nz=I.cfg.nz, S=I.inlet_stride;
    const double cell=I.cfg.cell_size;
    const double sc = I.u_lb / I.cfg.U_inlet;          // m/s -> lattice u

    // (1) x-inlet plane: full ABL (mean+RFG) on whichever x-face is an inlet, vary (y,z).
    //     face_bc[0]=−x, [1]=+x. In legacy mode fall back to the designated inlet axis.
    bool xin = (I.per_face ? (I.face_bc[0]==BC_INLET || I.face_bc[1]==BC_INLET)
                           : (I.inAx==0));
    if (xin) {
        int xside = I.per_face ? (I.face_bc[0]==BC_INLET ? 0 : 1) : I.inSide;
        double xp = (xside==0) ? 0.0 : (nx-1)*cell;
        #pragma omp parallel for collapse(2) schedule(static)
        for (int z=0; z<nz; ++z) for (int y=0; y<ny; ++y) {
            double ux,uy,uz; I.abl.velocity(xp, y*cell, z*cell, t_phys, ux,uy,uz);
            int k=z*S+y; I.inlet_plane[3*k]=ux*sc; I.inlet_plane[3*k+1]=uy*sc; I.inlet_plane[3*k+2]=uz*sc;
        }
    }
    // (2) y-inlet plane: only when a y-face is an inlet (oblique wind), vary (x,z).
    bool yin = (I.per_face ? (I.face_bc[2]==BC_INLET || I.face_bc[3]==BC_INLET)
                           : (I.inAx==1));
    if (yin && !I.inlet_plane_y.empty()) {
        int yside = I.per_face ? (I.face_bc[2]==BC_INLET ? 0 : 1) : I.inSide;
        double yp = (yside==0) ? 0.0 : (ny-1)*cell;
        #pragma omp parallel for collapse(2) schedule(static)
        for (int z=0; z<nz; ++z) for (int x=0; x<nx; ++x) {
            double ux,uy,uz; I.abl.velocity(x*cell, yp, z*cell, t_phys, ux,uy,uz);
            int k=z*S+x; I.inlet_plane_y[3*k]=ux*sc; I.inlet_plane_y[3*k+1]=uy*sc; I.inlet_plane_y[3*k+2]=uz*sc;
        }
    }
    // (3) RFG-free MEAN profile vs z — written at the shared vertical edge of two
    //     inlet faces so the seam is continuous (no RFG mismatch, no data jump).
    if (!I.mean_plane.empty()) {
        const double ca=cos(I.cfg.wind_angle), sa=sin(I.cfg.wind_angle);
        for (int z=0; z<nz; ++z) {
            double U = I.abl.mean_speed(z*cell);         // m/s, RFG-free
            I.mean_plane[3*z]   = (float)(U*ca*sc);
            I.mean_plane[3*z+1] = (float)(U*sa*sc);
            I.mean_plane[3*z+2] = 0.f;
        }
    }
}

// ─── Constructor / Destructor ────────────────────────────────────────

Solver::Solver(const Config& cfg) : p_(new Impl) {
    p_->cfg = cfg;
    p_->N   = cfg.nx * cfg.ny * cfg.nz;
    printf("[LBM] Grid %d×%d×%d = %.1fM cells\n",
           cfg.nx, cfg.ny, cfg.nz, p_->N / 1e6f);

    gpu::alloc(*p_);
    unit_convert(*p_);
    wind_setup(*p_);

    // ── Sheared turbulent ABL inlet (opt-in; cfg.inlet_profile==1) ──────────
    // Mean: log law (Richards & Hoxey 1993). Turbulence: RFG (Kraichnan 1970;
    // Smirnov, Shi & Celik 2001). sigma/u* ratios: Panofsky & Dutton (1984).
    p_->inlet_stride = std::max(cfg.nx, cfg.ny);
    if (cfg.inlet_profile == 1) {
        auto& A = p_->abl;
        A.z0            = cfg.abl_z0        > 0 ? cfg.abl_z0        : 0.7;
        A.L_turb        = cfg.abl_Lturb     > 0 ? cfg.abl_Lturb     : 40.0;
        A.n_modes       = cfg.abl_nmodes    > 0 ? cfg.abl_nmodes    : 100;
        A.sigma_u_ratio = cfg.abl_sigu_ratio> 0 ? cfg.abl_sigu_ratio: 2.5;
        A.sigma_v_ratio = cfg.abl_sigv_ratio> 0 ? cfg.abl_sigv_ratio: 1.9;
        A.sigma_w_ratio = cfg.abl_sigw_ratio> 0 ? cfg.abl_sigw_ratio: 1.25;
        A.wind_angle    = cfg.wind_angle;
        // Diagnosis experiment #3: mean-vs-turbulent discriminator. ABL_RFG=0
        // keeps the log-law MEAN inlet but disables the RFG fluctuation, so an
        // oblique divergence that persists is driven by the mean cross-stream (a
        // boundary problem) rather than the synthetic turbulence. Default: on.
        { const char* rfg = std::getenv("ABL_RFG");
          if (rfg && std::atoi(rfg) == 0) { A.enable_turb = false;
              printf("[LBM] ABL turbulence: DISABLED — mean log-law only (ABL_RFG=0)\n"); } }
        double zref     = cfg.abl_zref      > 0 ? cfg.abl_zref      : 40.0;
        A.init(cfg.U_inlet, zref, cfg.abl_seed);
        p_->inlet_plane.assign((size_t)3 * cfg.nz * p_->inlet_stride, 0.0f);
        // Second inlet plane + mean profile are only needed when the per-face rule
        // makes a y-face an inlet (oblique wind). Allocate whenever per_face is on.
        if (p_->per_face) {
            p_->inlet_plane_y.assign((size_t)3 * cfg.nz * p_->inlet_stride, 0.0f);
            p_->mean_plane.assign((size_t)3 * cfg.nz, 0.0f);
        }
        printf("[LBM] ABL inlet: log-law (z0=%.2f m, u*=%.3f m/s) + RFG turbulence "
               "(%d modes, L=%.0f m, sigma_u/u*=%.1f)\n",
               A.z0, A.u_star, A.n_modes, A.L_turb, A.sigma_u_ratio);
    }

    // Report the per-face boundary map (the oblique-wind fix; degenerates to the
    // original single-inlet layout at wind 0°).
    if (p_->per_face) {
        static const char* NM[3]={"sym","INLET","outflow"};
        printf("[LBM] Wind BC (per-face):  -x=%s  +x=%s  -y=%s  +y=%s\n",
               NM[p_->face_bc[0]],NM[p_->face_bc[1]],NM[p_->face_bc[2]],NM[p_->face_bc[3]]);
    } else {
        printf("[LBM] Wind BC (legacy single-inlet): inlet axis=%d side=%d, +/-y=symmetry"
               " (oblique cross-stream reflects — reproduces the divergence)\n",
               p_->inAx, p_->inSide);
    }

    p_->srcIdx = source_cell(*p_);
    printf("[LBM] Source cell %d  inlet: axis=%d side=%d\n",
           p_->srcIdx, p_->inAx, p_->inSide);

    // ── Experiment toggles (env-gated; default = ORIGINAL behavior) ──
    { const char* lb = std::getenv("LATERAL_BC");
      p_->lateral_bc = (lb && (lb[0]=='o' || lb[0]=='O')) ? 1 : 0;   // "open"
      const char* du = std::getenv("DUMP_UMAX_LOC");
      p_->dump_umax  = (du && std::atoi(du) != 0);
      if (p_->lateral_bc)
          printf("[LBM] Lateral ±y BC: OPEN zero-gradient (experiment #6; default is symmetry)\n");
      if (p_->dump_umax)
          printf("[LBM] Divergence-location diagnostic: ON (experiment #5)\n");
    }
}

Solver::~Solver() {
    gpu::free_all(*p_);
    delete p_;
}

void Solver::load_geometry(const uint8_t* tp, const float* pm, const float* inh,
                           const float* dvel) {
    gpu::upload_geometry(*p_, tp, pm, inh, dvel);
}

// Optional per-cell source distribution mask (FORWARD_LIVE.md item 1). The caller
// owns `mask` (length N) and keeps it alive across run(). We store the pointer and
// count the set cells |Ω| for the burst mass-budget scaling. mask=nullptr clears it
// (reverts to the single srcIdx source). CPU backend only; a GPU path must upload
// the mask to device memory here instead of aliasing the host pointer.
void Solver::load_source_mask(const uint8_t* mask) {
    p_->srcmask = mask;
    long cnt = 0;
    if (mask) for (int i = 0; i < p_->N; ++i) if (mask[i]) ++cnt;
    p_->n_srcmask = cnt;
    if (mask)
        printf("[LBM] source mask loaded: |Omega|=%ld release cells (distribution release)\n", cnt);
}
long Solver::source_mask_count() const { return p_->n_srcmask; }

void Solver::reset() {
    gpu::init_equilibrium(*p_);
    wind_setup(*p_);
}

void Solver::export_velocity(const char* filename, int z_slice) {
    int nx = p_->cfg.nx, ny = p_->cfg.ny;
    std::vector<float> h_ux(nx*ny), h_uy(nx*ny);
    gpu::export_velocity_slice(*p_, z_slice, h_ux.data(), h_uy.data(), nx, ny);
    FILE* f = fopen(filename, "wb");
    if (!f) { printf("[LBM] ERROR: cannot open %s\n", filename); return; }
    int hdr[2] = {nx, ny};
    fwrite(hdr, sizeof(int), 2, f);
    fwrite(h_ux.data(), sizeof(float), nx*ny, f);
    fwrite(h_uy.data(), sizeof(float), nx*ny, f);
    fclose(f);
    printf("[Export] %s (z=%d, %dx%d)\n", filename, z_slice, nx, ny);
}

void Solver::export_deposition(const char* filename, int z_slice) {
    int nx = p_->cfg.nx, ny = p_->cfg.ny;
    std::vector<float> h_dep(nx*ny);
    gpu::export_deposition_slice(*p_, z_slice, h_dep.data(), nx, ny);
    FILE* f = fopen(filename, "wb");
    if (!f) { printf("[LBM] ERROR: cannot open %s\n", filename); return; }
    int hdr[2] = {nx, ny};
    fwrite(hdr, sizeof(int), 2, f);
    fwrite(h_dep.data(), sizeof(float), nx*ny, f);
    fclose(f);
    printf("[Export] %s (deposition, z=%d, %dx%d)\n", filename, z_slice, nx, ny);
}

void Solver::export_concentration(const char* filename, int z_slice) {
    int nx = p_->cfg.nx, ny = p_->cfg.ny;
    std::vector<float> h_C(nx*ny);
    if (p_->n_avg > 0) gpu::export_avg_scalar_slice(*p_, z_slice, h_C.data(), nx, ny);
    else               gpu::export_scalar_slice(*p_, z_slice, h_C.data(), nx, ny, p_->g_final);
    FILE* f = fopen(filename, "wb");
    if (!f) { printf("[LBM] ERROR: cannot open %s\n", filename); return; }
    int hdr[2] = {nx, ny};
    fwrite(hdr, sizeof(int), 2, f);
    fwrite(h_C.data(), sizeof(float), nx*ny, f);
    fclose(f);
    printf("[Export] %s (concentration, z=%d, %dx%d)\n", filename, z_slice, nx, ny);
}

void Solver::export_tiac(const char* filename, int z_slice) {
    int nx = p_->cfg.nx, ny = p_->cfg.ny;
    std::vector<float> h_cint(nx*ny);
    gpu::export_tiac_slice(*p_, z_slice, h_cint.data(), nx, ny);
    FILE* f = fopen(filename, "wb");
    if (!f) { printf("[LBM] ERROR: cannot open %s\n", filename); return; }
    int hdr[2] = {nx, ny};
    fwrite(hdr, sizeof(int), 2, f);
    fwrite(h_cint.data(), sizeof(float), nx*ny, f);
    fclose(f);
    printf("[Export] %s (TIAC ∫C dt, z=%d, %dx%d)\n", filename, z_slice, nx, ny);
}

void Solver::copy_mean_flow_to_host(std::vector<float>& ux, std::vector<float>& uy,
                                    std::vector<float>& uz, std::vector<float>& nut) {
    int N = p_->N;
    ux.resize(N); uy.resize(N); uz.resize(N); nut.resize(N);
    gpu::download_mean_flow(*p_, ux.data(), uy.data(), uz.data(), nut.data());
}

void Solver::copy_tiac_to_host(std::vector<float>& tiac) {   // full TIAC (Θ=∫C dt)
    int N = p_->N; tiac.resize(N);
    gpu::download_tiac(*p_, tiac.data());
}

void Solver::copy_density_to_host(std::vector<float>& rho) {
    int N = p_->N;
    rho.resize(N);
    gpu::download_density(*p_, rho.data());
}

// ── Time-resolved visualization support ───────────────────────────────────
void Solver::set_snapshot_callback(void (*cb)(void*,int,int,int,bool,bool), void* ctx) {
    p_->snapshot_cb = cb; p_->snapshot_ctx = ctx;
}

// Full instantaneous scalar field from ping-pong buffer gbuf, assembled slice by
// slice (reuses the verified per-slice reducer; no new kernel).
void Solver::copy_scalar_to_host(std::vector<float>& C, int gbuf) {
    const int nx = p_->cfg.nx, ny = p_->cfg.ny, nz = p_->cfg.nz;
    C.assign((size_t)nx*ny*nz, 0.f);
    std::vector<float> sl((size_t)nx*ny);
    for (int z = 0; z < nz; ++z) {
        gpu::export_scalar_slice(*p_, z, sl.data(), nx, ny, gbuf);
        std::copy(sl.begin(), sl.end(), C.begin() + (size_t)z*nx*ny);
    }
}

void Solver::copy_instant_velocity_to_host(std::vector<float>& ux, std::vector<float>& uy,
                                           std::vector<float>& uz) {
    int N = p_->N; ux.resize(N); uy.resize(N); uz.resize(N);
    gpu::download_instant_velocity(*p_, ux.data(), uy.data(), uz.data());
}

void Solver::copy_deposition_to_host(std::vector<float>& dep) {
    const int nx = p_->cfg.nx, ny = p_->cfg.ny, nz = p_->cfg.nz;
    dep.assign((size_t)nx*ny*nz, 0.f);
    std::vector<float> sl((size_t)nx*ny);
    for (int z = 0; z < nz; ++z) {
        gpu::export_deposition_slice(*p_, z, sl.data(), nx, ny);
        std::copy(sl.begin(), sl.end(), dep.begin() + (size_t)z*nx*ny);
    }
}

int Solver::averaging_samples() const { return p_->n_avg; }
int Solver::effective_collision() const { return p_->cfg.collision_mode; }
int Solver::effective_scalar()    const { return p_->cfg.scalar_advection; }

float Solver::settling_velocity_lattice() const { return p_->w_settle_lb; }

float Solver::velocity_phys_to_lattice() const {
    return p_->dt_phys / p_->cfg.cell_size;
}

// Warm-restart checkpoint: [magic,nx,ny,nz] [cell,U_inlet,wind_angle] [f: 19*N floats]
static const int FLOW_CKPT_MAGIC = 0x4C424D31;  // "LBM1"

bool Solver::save_flow_checkpoint(const char* filename) {
    auto& I = *p_;
    int hdr[4] = {FLOW_CKPT_MAGIC, I.cfg.nx, I.cfg.ny, I.cfg.nz};
    float meta[3] = {I.cfg.cell_size, I.cfg.U_inlet, I.cfg.wind_angle};
    std::vector<float> h_f((size_t)19*I.N);
    gpu::download_flow(I, I.g_final, h_f.data());   // last-written flow buffer
    FILE* f = fopen(filename, "wb");
    if (!f) { printf("[LBM] ERROR: cannot open %s for flow checkpoint\n", filename); return false; }
    fwrite(hdr, sizeof(int), 4, f);
    fwrite(meta, sizeof(float), 3, f);
    fwrite(h_f.data(), sizeof(float), h_f.size(), f);
    fclose(f);
    printf("[LBM] Saved flow checkpoint %s (%d×%d×%d)\n", filename, I.cfg.nx, I.cfg.ny, I.cfg.nz);
    return true;
}

bool Solver::load_flow_checkpoint(const char* filename) {
    auto& I = *p_;
    FILE* f = fopen(filename, "rb");
    if (!f) return false;
    int hdr[4]; float meta[3];
    if (fread(hdr, sizeof(int), 4, f) != 4 || fread(meta, sizeof(float), 3, f) != 3) {
        fclose(f); return false;
    }
    bool grid_match = (hdr[0]==FLOW_CKPT_MAGIC &&
                       hdr[1]==I.cfg.nx && hdr[2]==I.cfg.ny && hdr[3]==I.cfg.nz);
    if (!grid_match) {
        fclose(f);
        printf("[LBM] Warm restart skipped: grid mismatch (cold start)\n");
        return false;   // encapsulation: silently fall back to equilibrium
    }
    I.warm_f.resize((size_t)19*I.N);
    size_t got = fread(I.warm_f.data(), sizeof(float), I.warm_f.size(), f);
    fclose(f);
    if (got != I.warm_f.size()) { I.warm_f.clear(); return false; }
    I.warm_pending = true;
    printf("[LBM] Warm restart: loaded flow field (%d×%d×%d)\n", I.cfg.nx, I.cfg.ny, I.cfg.nz);
    return true;
}

// ─── Divergence-location diagnostic (experiment #5) ──────────────────
// Downloads the instantaneous velocity and reports where |u| is largest and
// where the field first goes non-finite, with a boundary/corner label. This
// turns a bare "diverged" into "diverged HERE" — the direct test of whether a
// blow-up sits on the ±y symmetry planes / inlet corner (the oblique-wind
// hypothesis) versus somewhere in the interior.
static void dump_umax_location(Solver::Impl& I, const char* tag, int step) {
    const int N = I.N, nx = I.cfg.nx, ny = I.cfg.ny, nz = I.cfg.nz;
    static std::vector<float> ux, uy, uz;
    ux.resize(N); uy.resize(N); uz.resize(N);
    gpu::download_instant_velocity(I, ux.data(), uy.data(), uz.data());
    double umax = -1.0; int im = -1; long n_nan = 0; int first_nan = -1;
    for (int id = 0; id < N; ++id) {
        float a = ux[id], b = uy[id], c = uz[id];
        if (!(std::isfinite(a) && std::isfinite(b) && std::isfinite(c))) {
            ++n_nan; if (first_nan < 0) first_nan = id; continue;
        }
        double m = std::sqrt((double)a*a + (double)b*b + (double)c*c);
        if (m > umax) { umax = m; im = id; }
    }
    auto classify = [&](int id, char* buf, size_t n) {
        if (id < 0) { std::snprintf(buf, n, "(none)"); return; }
        int z = id / (nx*ny), r = id % (nx*ny), y = r / nx, x = r % nx;
        char faces[16] = ""; int fi = 0;
        if (x==0)    faces[fi++]='<';                       // −x (inlet face here)
        if (x==nx-1) faces[fi++]='>';                       // +x (outlet)
        if (y==0)    { faces[fi++]='['; }                   // −y lateral
        if (y==ny-1) { faces[fi++]=']'; }                   // +y lateral
        if (z==0)    faces[fi++]='_';                       // floor
        if (z==nz-1) faces[fi++]='^';                       // top
        faces[fi] = '\0';
        std::snprintf(buf, n, "(x=%d,y=%d,z=%d) boundary='%s'%s",
                      x, y, z, faces[0]?faces:"interior",
                      (fi>=2)?" [CORNER/EDGE]":"");
    };
    char b1[128], b2[128];
    classify(im, b1, sizeof b1);
    classify(first_nan, b2, sizeof b2);
    printf("[DIAG:%s step=%d] max|u|=%.4g at %s | non-finite=%ld", tag, step, umax, b1, n_nan);
    if (n_nan > 0) printf(" first at %s", b2);
    printf("\n"); std::fflush(stdout);
}

// ─── Main solver loop ────────────────────────────────────────────────

Result Solver::run() {
    reset();                      // equilibrium f, zero scalar g, zero deposition
    auto& I = *p_;
    gpu::sync_source_mask(I);     // make the source-distribution mask resident (GPU upload / CPU no-op)
    // Warm restart: overwrite the equilibrium flow with a matching prior field.
    // Scalar/deposition/TIAC stay freshly zeroed (the release always starts clean).
    if (I.warm_pending) {
        gpu::upload_flow(I, I.warm_f.data());
        I.warm_pending = false;
        printf("[LBM] Flow initialised from warm restart\n");
    }
    int ci = std::max(1, I.cfg.check_interval);

    Result res{};
    res.avg_start = -1;

    const float dt = I.dt_phys;
    int n_release = (I.cfg.release_time > 0.f)
                    ? std::max(1, (int)lroundf(I.cfg.release_time / dt))
                    : I.cfg.max_steps;            // fallback if release_time unset

    printf("[LBM] dt = %.5f s/step\n", dt);

    int gstep = 0;

    // Auto defaults from one domain flow-through (steps for the free stream to
    // cross the longest horizontal dimension): n_ft = L / u_lb.
    int Lmax = std::max(I.cfg.nx, I.cfg.ny);
    int n_ft = std::max(1, (int)std::lround(Lmax / std::max(1e-6, (double)I.u_lb)));
    // Spin-up length in FLOW-THROUGHS (default 2). Exposed as SPINUP_FT so the
    // publishable convergence sweep can flush a slow bluff-body wake transient
    // BEFORE averaging starts — and, being a multiplier on n_ft, it stays the same
    // PHYSICAL spin-up across the H=24/36/54 triplet. An explicit cfg.spinup_steps
    // (absolute) still wins if set.
    int spin_ft = I.cfg.spinup_ft > 0 ? I.cfg.spinup_ft : 2;   // driver default (else 2)
    { const char* s=std::getenv("SPINUP_FT"); if(s){ int v=std::atoi(s); if(v>=1) spin_ft=v; } } // env wins
    int spin   = I.cfg.spinup_steps > 0 ? I.cfg.spinup_steps : spin_ft * n_ft;  // flush transient
    int window = I.cfg.avg_steps    > 0 ? I.cfg.avg_steps    : std::max(500, n_ft / 2);
    float stat_tol = I.cfg.avg_threshold > 0 ? I.cfg.avg_threshold : 5e-3f;
    int cap = I.cfg.max_warmup > 0 ? I.cfg.max_warmup : 12 * n_ft;            // averaging-loop cap

    printf("[LBM] Phase A: spin-up %d (%d flow-through) + time-average to stationarity "
           "(|Δmean|/mean<%.0e per %d-step window, cap %d)  [n_ft≈%d]\n",
           spin, spin_ft, stat_tol, window, cap, n_ft);
    printf("[LBM] Phase B: release source, advect-diffuse %.1fs = %d steps\n",
           n_release * dt, n_release);

    // ── Phase A1: spin-up (flow only, no averaging) ──
    for (int s = 0; s < spin; ++s, ++gstep) {
        int cur = gstep & 1, nxt = 1 - cur;
        fill_inlet_plane(I, gstep * I.dt_phys);
        gpu::step(I, cur, nxt, /*with_scalar=*/false);
        if (s > 0 && s % ci == 0) {
            float rel = gpu::rms_velocity_delta(I) / I.u_lb;
            if (!std::isfinite(rel)) {
                if (I.dump_umax) dump_umax_location(I, "DIVERGED", s);
                printf("[LBM] FATAL: flow diverged (RMS=NaN/inf) at spin-up step %d "
                       "— raise nu_floor. Aborting.\n", s); std::fflush(stdout); std::exit(2);
            }
            if (I.dump_umax) dump_umax_location(I, "spinup", s);
            if (s % (ci * 5) == 0) printf("[LBM] spin-up step %d  RMS(Δu)/U=%.2e\n", s, rel);
        }
    }

    // ── Phase A2: accumulate running MEAN, stop at statistical stationarity ──
    gpu::reset_averages(I);
    res.avg_start = gstep;
    int min_avg = 3 * n_ft;      // require ≥3 flow-throughs of averaging regardless of the probe
    bool stationary = false; float probe_prev = -1.f; int n_conv = 0;
    for (int s = 0; gstep < spin + cap; ++s, ++gstep) {
        int cur = gstep & 1, nxt = 1 - cur;
        fill_inlet_plane(I, gstep * I.dt_phys);
        gpu::step(I, cur, nxt, /*with_scalar=*/false);
        gpu::accumulate(I, cur);                        // running mean (ux_sum/…/n_avg)
        if (s > 0 && s % window == 0) {
            float mv = 0.f, mc = 0.f;
            gpu::compute_averaged_results(I, mv, mc);   // max |mean u| — the stationarity probe
            if (!std::isfinite(mv)) {
                printf("[LBM] FATAL: mean flow diverged. Aborting.\n"); std::fflush(stdout); std::exit(2);
            }
            float rel = (probe_prev > 0.f) ? std::fabs(mv - probe_prev) / probe_prev : 1.f;
            printf("[LBM] avg step %d  n_avg=%d  max|u_mean|=%.4f  Δ=%.2e%s\n",
                   s, I.n_avg, mv, rel, (s < min_avg ? "  (min-window)" : ""));
            if (rel < stat_tol && s >= min_avg) { if (++n_conv >= 2) { stationary = true;
                    printf("[LBM] ──▶ mean flow stationary at avg step %d (Δ=%.2e, %d samples)\n",
                           s, rel, I.n_avg);
                    ++gstep; break; } }
            else if (rel >= stat_tol) n_conv = 0;
            probe_prev = mv;
        }
    }
    res.steps_flow = gstep;
    if (!stationary)
        printf("[LBM] WARNING: mean flow not stationary within cap (%d steps); "
               "mean is averaged over n_avg=%d samples anyway\n", cap, I.n_avg);

    // ── Phase B: source released; advection-diffusion on the live flow ──
    const bool  burst     = I.cfg.burst_release;
    const double M_release= burst ? (double)I.cfg.Q_source : 0.0; // reuse Q_source as burst PER-CELL mass
    // With a source-distribution mask, EACH of the |Ω|=n_srcmask cells emits
    // M_release, so the total released mass is M_release·|Ω| (mask absent ⇒ ×1).
    const double mask_cells = (I.n_srcmask > 0) ? (double)I.n_srcmask : 1.0;
    const double M_total   = M_release * mask_cells;
    const float  Q_saved  = I.cfg.Q_source;
    int cap_release = n_release;
    int n_pulse = 1;
    if (burst) {
        // Injection spread over a finite pulse: M over n_pulse steps (impulse when
        // n_pulse=1). Q·dt per step × n_pulse = M ⇒ Q = M/(n_pulse·dt).
        n_pulse = std::max(1, (int)lroundf(I.cfg.pulse_seconds / dt));
        I.cfg.Q_source = (float)(M_release / (n_pulse * dt));
        cap_release = (I.cfg.max_steps > 0) ? I.cfg.max_steps : 300000;
        printf("[LBM] Phase B: BURST M=%.4e/cell x %.0f cells = %.4e total over %d-step pulse (%.2fs), "
               "run to %.0f%% clearance (cap %d steps)\n",
               M_release, mask_cells, M_total, n_pulse, n_pulse * dt,
               100.0*(1.0 - I.cfg.clearance_frac), cap_release);
    }
    bool cleared = false;
    for (int i = 0; i < cap_release; ++i, ++gstep) {
        int cur = gstep & 1, nxt = 1 - cur;
        fill_inlet_plane(I, gstep * I.dt_phys);
        gpu::step(I, cur, nxt, /*with_scalar=*/true);
        if (burst && i == n_pulse - 1) I.cfg.Q_source = 0.f;    // source off after the pulse
        // ── time-resolved snapshot (viz): at release, then every snapshot_every ──
        if (I.snapshot_cb && I.cfg.snapshot_every > 0
            && (i == 0 || i % I.cfg.snapshot_every == 0)) {
            I.g_final = nxt; I.snapshot_cb(I.snapshot_ctx, i, gstep, nxt, /*release=*/i == 0, /*end=*/false);
        }
        if (burst && i >= n_pulse) {                            // clearance only after injection ends
            if ((i + 1) % (ci * 2) == 0) {
                double air = gpu::total_airborne(I, 1 - (gstep & 1));
                if (M_total > 0.0 && air < I.cfg.clearance_frac * M_total) {
                    printf("[LBM] burst cleared: airborne=%.3e (<%.0f%%) at step %d (t=%.1fs)\n",
                           air, 100.0*I.cfg.clearance_frac, i + 1, (i + 1) * dt);
                    if (I.snapshot_cb && I.cfg.snapshot_every > 0) {
                        I.g_final = nxt; I.snapshot_cb(I.snapshot_ctx, i, gstep, nxt, /*release=*/false, /*end=*/true);
                    }
                    ++gstep; cleared = true; break;
                }
            }
        } else if (!burst && (i + 1) % (ci * 5) == 0) {
            printf("[LBM] release step %d/%d  (t=%.1fs)\n", i + 1, n_release, (i + 1) * dt);
        }
    }
    I.cfg.Q_source = Q_saved;
    if (burst && !cleared)
        printf("[LBM] WARNING: burst did not reach %.0f%% clearance within cap — Θ truncated\n",
               100.0*(1.0 - I.cfg.clearance_frac));
    res.steps_total = gstep;

    // ── Results: instantaneous final state + mass budget ──
    int gfin = 1 - ((gstep - 1) & 1);
    I.g_final = gfin;
    // End snapshot when the loop hit the cap without a clearance break (the
    // clearance path already fired its end snapshot in-loop).
    if (!cleared && I.snapshot_cb && I.cfg.snapshot_every > 0)
        I.snapshot_cb(I.snapshot_ctx, cap_release - 1, gstep, gfin, /*release=*/false, /*end=*/true);
    gpu::compute_results(I, gfin, res.max_velocity, res.max_concentration);

    res.total_emitted   = burst ? M_total
                                : (double)(I.cfg.Q_source * dt) * (double)n_release;
    res.total_deposited = gpu::total_deposited(I);
    res.total_airborne  = gpu::total_airborne(I, gfin);
    res.w_settle_phys   = I.w_settle_lb * I.cfg.cell_size / dt;
    double outflow = res.total_emitted - res.total_deposited - res.total_airborne;

    printf("[LBM] Instantaneous final: max|u|=%.4f  maxC=%.4e\n",
           res.max_velocity, res.max_concentration);
    if (res.total_emitted > 0.0) {
        printf("[LBM] Mass budget (LU): emitted=%.4e  deposited=%.4e (%.1f%%)  "
               "airborne=%.4e (%.1f%%)  outflow=%.4e (%.1f%%)\n",
               res.total_emitted,
               res.total_deposited, 100.0*res.total_deposited/res.total_emitted,
               res.total_airborne,  100.0*res.total_airborne /res.total_emitted,
               outflow,             100.0*outflow            /res.total_emitted);
    }
    return res;
}

void Solver::export_avg_velocity(const char* filename, int z_slice) {
    int nx = p_->cfg.nx, ny = p_->cfg.ny;
    if (p_->n_avg < 1) {
        printf("[LBM] WARNING: no averaged data — call run() with avg_steps>0 first\n");
        return;
    }
    std::vector<float> h_ux(nx*ny), h_uy(nx*ny);
    gpu::export_avg_velocity_slice(*p_, z_slice, h_ux.data(), h_uy.data(), nx, ny);
    FILE* f = fopen(filename, "wb");
    if (!f) { printf("[LBM] ERROR: cannot open %s\n", filename); return; }
    int hdr[2] = {nx, ny};
    fwrite(hdr, sizeof(int), 2, f);
    fwrite(h_ux.data(), sizeof(float), nx*ny, f);
    fwrite(h_uy.data(), sizeof(float), nx*ny, f);
    fclose(f);
    printf("[Export] %s (z=%d, %dx%d, avg over %d steps)\n",
           filename, z_slice, nx, ny, p_->n_avg);
}

} // namespace lbm
