#pragma once
// lbm_solver.h — Public interface.  Pure C++, no CUDA dependency.
//
// Build:
//   nvcc -O3 -arch=sm_XX --extended-lambda -c lbm_kernels.cu -o kernels.o
//   g++ -O3 -std=c++17 -c lbm_solver.cpp -o solver.o
//   g++ kernels.o solver.o -L/usr/local/cuda/lib64 -lcudart -o solver

#include <cstdint>
#include <vector>

namespace lbm {

struct Config {
    int   nx, ny, nz;
    float cell_size;          // m per cell
    float U_inlet;            // physical wind speed m/s
    float wind_angle;         // 0→+x, π/2→+y, π→-x, 3π/2→-y
    float nu_phys;            // air viscosity (1.5e-5)
    float Cw;                 // WALE constant (0.325)
    float Sc_t;               // turbulent Schmidt (0.7)
    float D_mol;              // molecular diffusivity (1e-5)
    float source_x, source_y, source_z;
    float Q_source;           // emission rate

    // ── Particulate settling + surface deposition ──
    // Gravitational settling velocity is computed from particle size via
    // Stokes' law with Cunningham slip correction. Set particle_diam = 0 for
    // a passive (non-settling) tracer gas. Per-surface deposition velocity is
    // supplied per cell through load_geometry() (see voxelize.h).
    float particle_diam;      // particle diameter d_p (m), e.g. 2.5e-6 = PM2.5; 0 = no settling
    float particle_density;   // particle density ρ_p (kg/m³), e.g. 1000–2600

    // Scalar advection scheme (fixes high-cell-Péclet oscillation):
    //   0 = van Leer TVD  (default; positivity-preserving, accurate)
    //   1 = first-order upwind (most diffusive, most robust)
    //   2 = legacy central (TRT equilibrium advection; oscillatory at high Pe)
    int   scalar_advection = 0;  // default van Leer; SCALAR env or driver overrides (forward=QUICK)

    int   max_steps;
    int   check_interval;
    float conv_threshold;
    int   scalar_extra_steps;   // (legacy; ignored by the warm-up→release loop)

    // ── Run structure: warm up flow (spin-up + time-averaging), then release ──
    // Phase A1 (spin-up): evolve flow only, no averaging, to flush the initial
    //   transient (~1-2 domain flow-throughs).
    // Phase A2 (averaging): accumulate the running MEAN flow and stop when it is
    //   STATISTICALLY STATIONARY — the mean-field probe changes by < avg_threshold
    //   between windows of avg_window steps. This replaces the old instantaneous
    //   RMS(Δu)/U test, which never converges for turbulent (ABL/RFG) inflow.
    // Phase B: release the source and run advection-diffusion on the live flow.
    int   max_warmup;       // cap on total Phase-A steps (spin-up + averaging)
    float avg_threshold;    // stationarity tol: |Δ(mean probe)|/probe between windows
    int   avg_steps;        // = avg_window: steps between stationarity checks (0 → auto)
    int   spinup_steps;     // Phase-A1 spin-up steps before averaging starts (0 → auto)
    int   spinup_ft = 0;    // spin-up length in FLOW-THROUGHS (0 → default 2). Set by
                            // the driver so the physical spin-up is resolution-independent;
                            // env SPINUP_FT overrides both. Live-forward production uses 3.
    float release_time;     // Phase-B duration in SECONDS (n_steps = release_time/dt_phys)
    // ── Accidental-release (burst) forward mode ────────────────────────────
    // burst_release: inject the TOTAL source mass once at the first Phase-B step
    // (an impulse), then no ongoing source, and run until the airborne fraction
    // falls below clearance_frac (default 1% => 99% deposited/advected out). This
    // makes the cumulative dose Θ=∫C dt self-terminating and window-independent.
    // abl_seed varies the RFG turbulence per run so an ensemble of bursts samples
    // independent turbulent realisations (see FORWARD_LIVE.md).
    bool     burst_release  = false;
    float    clearance_frac = 0.01f;
    unsigned abl_seed       = 12345u;
    // Release pulse width (seconds). The burst mass M is injected uniformly over
    // n = round(pulse_seconds/dt) steps. Default ~2 s is impulse-equivalent in an
    // urban canopy (pulse ≪ eddy-turnover time ~ H/U ≈ 5–15 s) while trimming the
    // sharpest single-instant turbulent noise. Set 0 (→ n=1) for a true one-step
    // delta impulse; the dosage Θ=∫C dt is unchanged either way (mass-conserving).
    float    pulse_seconds  = 2.0f;

    // ── Time-resolved snapshots for visualization ──────────────────────────
    // If > 0, the solver invokes the snapshot callback (set via
    // set_snapshot_callback) every `snapshot_every` Phase-B steps, plus once at
    // release (first step) and once at clearance/end, passing the current scalar
    // buffer index. A driver uses it to dump concentration/deposition slices and
    // the exposure time series. 0 = off (no snapshots; zero overhead).
    int      snapshot_every = 0;

    // ── Sheared turbulent ABL inlet (abl_inlet.h) ──────────────────────────
    // inlet_profile: 0 = legacy uniform plug flow (default; zero-initialised);
    //                1 = neutral-ABL log-law mean (Richards & Hoxey 1993) +
    //                    Random Flow Generation turbulence (Kraichnan 1970;
    //                    Smirnov, Shi & Celik 2001).
    int   inlet_profile;
    float abl_z0;          // aerodynamic roughness length z0 (m); ~0.5-1 urban
    float abl_zref;        // reference height at which mean speed = U_inlet (m)
    float abl_Lturb;       // turbulence integral length scale (m)
    int   abl_nmodes;      // number of Fourier modes
    float abl_sigu_ratio;  // sigma_u / u*  (Panofsky & Dutton 1984: ~2.5)
    float abl_sigv_ratio;  //               (~1.9)
    float abl_sigw_ratio;  //               (~1.25)

    // collision_mode: 0 = plain MRT (d'Humières tuned ghost rates; default,
    //   original behaviour). 1 = projected-regularized MRT — the non-hydrodynamic
    //   ("ghost") moments are relaxed to their zero equilibrium every step (rate 1),
    //   which filters the aliased ghost modes that drive the τ→0.5 (low-ν) blow-up.
    //   This is the moment-space form of regularized LBM (Latt & Chopard 2006) and
    //   the stability mechanism behind the HRR family (Jacob, Malaspinas & Sagaut
    //   2018). Runtime-selectable via env COLLISION=mrt|reg.
    int   collision_mode = 1;   // default reg; env COLLISION or a driver may override
    // hrr_sigma: HRR hybrid blend in [0,1] for COLLISION=hrr (collision_mode=2).
    //   a3 = sigma*a3_recursive + (1-sigma)*a3_finite_difference. sigma=1 => pure
    //   recursive-regularized (RR, local, no added dissipation); sigma<1 adds the
    //   FD hyperviscosity that stabilizes high-Re/low-nu (Jacob, Malaspinas &
    //   Sagaut 2018). Env HRR_SIGMA (default 0.98). Ignored unless collision_mode==2.
    float hrr_sigma;
    // wall_model: 0 = no-slip half-way bounce-back floor (default). 1 = rough-wall
    //   log-law moving-wall bounce-back on the DOMAIN FLOOR only (not buildings),
    //   imposing the ABL slip u_wall so the ground sustains the log law. Env WALL_MODEL.
    int   wall_model = 0;   // default off; env WALL_MODEL or a driver may set
};

struct Result {
    int    steps_flow;
    int    steps_total;
    int    avg_start;
    float  max_velocity;
    float  max_concentration;

    // Particulate mass budget (lattice units, accumulated over the averaging
    // window in averaging mode, or the whole run in legacy mode).
    double total_emitted;     // scalar mass injected at the source
    double total_deposited;   // scalar mass captured on surfaces
    double total_airborne;    // scalar mass still suspended at the end
    float  w_settle_phys;     // settling velocity used (m/s), for reference
};

class Solver {
public:
    Solver(const Config& cfg);
    ~Solver();
    void   load_geometry(const uint8_t* type, const float* perm, const float* inh,
                         const float* dep_vel);
    // Optional per-cell source MASK for a simultaneous source-DISTRIBUTION release
    // (FORWARD_LIVE.md item 1). mask is an N-length array (nx*ny*nz); every nonzero
    // cell emits the burst at once. The caller OWNS the memory and must keep it
    // alive for the duration of run(). Pass mask=nullptr (or never call this) to
    // keep the original single-point source_x/y/z behavior. CPU backend only.
    void   load_source_mask(const uint8_t* mask);
    long   source_mask_count() const;   // |Ω| set by load_source_mask (0 if unset)
    Result run();
    void   reset();
    // Export ground-level velocity to binary: [nx,ny as int32], [ux as float32], [uy as float32]
    void   export_velocity(const char* filename, int z_slice = 1);
    void   export_avg_velocity(const char* filename, int z_slice = 1);
    // Export deposited-mass map at a z-slice: [nx,ny as int32], [dep as float32].
    // z=1 (default) is the near-ground layer, i.e. the ground deposition map.
    void   export_deposition(const char* filename, int z_slice = 1);
    // Export concentration map at a z-slice (time-averaged if available):
    // [nx,ny as int32], [C as float32].
    void   export_concentration(const char* filename, int z_slice = 1);
    // Export time-integrated air concentration (TIAC = ∫C dt over Phase B) at a
    // z-slice: [nx,ny as int32], [cint as float32]. Feeds airborne inhalation.
    void   export_tiac(const char* filename, int z_slice = 1);

    // Copy the frozen MEAN flow (time-averaged velocity) and eddy viscosity to
    // host vectors of length N (=nx*ny*nz). Used to drive the reverse (adjoint)
    // transport solver, which freezes this flow. Call after run().
    void   copy_mean_flow_to_host(std::vector<float>& ux, std::vector<float>& uy,
                                  std::vector<float>& uz, std::vector<float>& nut);

    // Copy the frozen MEAN density to a host vector of length N. Mean pressure is
    // p = c_s²·ρ (c_s² = 1/3 in lattice units); the pressure coefficient is
    // Cp = (p − p_ref)/(½ ρ_ref U_ref²). Call after run().
    void   copy_tiac_to_host(std::vector<float>& tiac);  // full ∫C dt (Θ) field
    void   copy_density_to_host(std::vector<float>& rho);

    // ── Time-resolved visualization support ────────────────────────────────
    // Register a callback fired during Phase B (see Config::snapshot_every) with
    // signature (phaseB_step, global_step, scalar_buffer, is_release, is_end).
    // Inside it, use the copy_* methods below with the given scalar_buffer to pull
    // the CURRENT (instantaneous) fields. POD ABI (plain function pointer + ctx) so
    // this header carries no standard-library callable wrapper (which nvcc's
    // cudafe++ mis-parses); the .cpp driver wraps its own lambda behind it.
    // Pass cb=nullptr to clear. See forward_city.cpp for the wrapper pattern.
    void   set_snapshot_callback(void (*cb)(void*,int,int,int,bool,bool), void* ctx);
    // Full INSTANTANEOUS scalar field C from a specific ping-pong buffer (0/1).
    void   copy_scalar_to_host(std::vector<float>& C, int gbuf);
    // Full INSTANTANEOUS macroscopic velocity (live field, not the time-average).
    void   copy_instant_velocity_to_host(std::vector<float>& ux, std::vector<float>& uy,
                                         std::vector<float>& uz);
    // Full accumulated deposited-mass field (Σ over Phase B so far).
    void   copy_deposition_to_host(std::vector<float>& dep);

    // Number of samples in the time-average (0 ⇒ averaging never ran → the mean is
    // just the instantaneous field). The smoke test checks this is > 0.
    int    averaging_samples() const;
    int    effective_collision() const;  // the collision_mode actually in force (after env resolution)
    int    effective_scalar() const;     // the scalar_advection actually in force

    // Gravitational settling velocity in LATTICE units (>=0, downward), as used by
    // the LBM scalar — so the reverse solver settles consistently with the forward.
    float  settling_velocity_lattice() const;

    // Factor converting a physical velocity (m/s) to lattice units (= dt_phys /
    // cell_size). Used to convert per-surface deposition velocities for the
    // reverse solver consistently with the LBM's own nondimensionalization.
    float  velocity_phys_to_lattice() const;

    // Warm restart: save/load the flow distribution f. load_flow_checkpoint()
    // initializes the flow ONLY if the stored grid (nx,ny,nz) matches the current
    // one; otherwise it leaves the equilibrium init untouched and returns false.
    // The Phase-A steady check then self-corrects any geometry change.
    bool   save_flow_checkpoint(const char* filename);
    bool   load_flow_checkpoint(const char* filename);
    struct Impl;              // public for gpu:: wrapper access
private:
    Impl* p_;
};

} // namespace lbm
