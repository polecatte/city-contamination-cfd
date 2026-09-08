#pragma once
// lbm_gpu.h — Internal interface between host (.cpp) and device (.cu).
// No CUDA headers needed — device pointers are plain float*/uint8_t*.

#include "lbm_solver.h"
#include "abl_inlet.h"
#include <cstdint>
#include <cstddef>
#include <vector>

namespace lbm {

// Per-face boundary classification (shared by solver + both kernels).
enum bc_mode { BC_SYM = 0, BC_INLET = 1, BC_OUTFLOW = 2 };

// ── Per-cell DEVICE memory footprint ────────────────────────────────────────
// Sum of all N-sized arrays allocated in gpu::alloc(). The memory pre-flight
// guard (in alloc(), on every run) compares bytes_per_cell()*N against the free
// device memory and aborts before allocating if it won't fit.
//
//   ⚠ MAINTENANCE: update the counts below whenever a per-cell array is added to
//   or removed from alloc() — otherwise the guard silently under/over-estimates.
//   Last synced with lbm_kernels.cu alloc():
//     distributions  f[2]×19 + g[2]×7                     = 38 + 14 = 52 floats
//     macroscopic    rho,ux,uy,uz,nut                     =           5 floats
//     prev-step u    ux0,uy0,uz0  (RMS Δu)                =           3 floats
//     fields         pm(perm),inh                         =           2 floats
//     scalar temps   dvel,dep,cint,cmac[0],cmac[1]        =           5 floats
//     time-average   ux_sum,uy_sum,uz_sum,rho_sum,C_sum   =           5 floats
//     type           tp (uint8)                           =           1 byte
inline size_t device_floats_per_cell(){ return 52 + 5 + 3 + 2 + 5 + 5; }   // = 72
inline size_t bytes_per_cell(){ return device_floats_per_cell()*sizeof(float) + 1; } // +1B type

// Solver internal state (visible to both .cpp and .cu)
struct Solver::Impl {
    Config  cfg;
    int     N;

    // Device arrays (allocated by gpu_alloc, opaque to .cpp)
    float   *f[2], *g[2];             // ping-pong flow/scalar distributions
    float   *rho, *ux, *uy, *uz;      // macroscopic (stored for WALE + scalar)
    float   *ux0, *uy0, *uz0;         // previous-checkpoint velocity
    uint8_t *tp;                       // cell type
    float   *pm, *inh;                // permeability, inhabitance
    float   *nut;                      // WALE turbulent viscosity (for scalar kernel)
    float   *d_inlet_plane;            // device mirror of inlet_plane (ABL inlet; uploaded per step)
    float   *dvel;                     // per-cell deposition sticking prob α∈[0,1] (surfaces only)
    float   *dep;                      // per-cell accumulated deposited scalar mass
    float   *cint;                     // per-cell ∫C dt accumulator (time-integrated air conc.)
    float   *cmac[2];                  // ping-pong macroscopic scalar C (for TVD advection stencil)

    // Time-averaging accumulators
    float   *ux_sum, *uy_sum, *uz_sum, *C_sum;
    float   *rho_sum;                  // mean density → mean pressure (Cp): p = c_s²·ρ
    int      n_avg;
    int      g_final;                  // scalar buffer index holding the last state

    // Warm restart (flow only)
    std::vector<float> warm_f;         // staged flow field to load after reset()
    bool     warm_pending;

    // Sheared turbulent ABL inlet (host-computed plane, backend-agnostic).
    // inlet_plane holds the per-inlet-cell lattice velocity (ux,uy,uz) indexed
    // by [z*inlet_stride + (y or x)]; refilled each step for the current time.
    abl::ABLInlet      abl;
    std::vector<float> inlet_plane;      // ABL plane on the x-inlet face   (idx z*stride + y)
    std::vector<float> inlet_plane_y;    // ABL plane on the y-inlet face   (idx z*stride + x) — oblique wind
    std::vector<float> mean_plane;       // RFG-free mean velocity vs z (3*nz): shared-edge tie-break
    float             *d_inlet_plane_y = nullptr;  // device mirrors (CUDA path; nullptr on CPU)
    float             *d_mean_plane    = nullptr;
    int                inlet_stride;

    // GPU streams and events (stored as void* to avoid CUDA types in header)
    void    *stream_flow, *stream_scalar, *evt_macro_done;

    // Derived unit-conversion constants
    float   u_lb, nu_lb, D_lb, dt_phys;
    float   w_settle_lb;               // gravitational settling speed in LU (downward)
    int     srcIdx;
    // ── Optional per-cell source MASK (distribution release; FORWARD_LIVE.md item 1) ──
    // When srcmask != nullptr, the scalar burst is injected at EVERY cell with
    // srcmask[id]!=0 (each at the same per-cell rate Qdt) instead of the single
    // srcIdx cell — releasing a whole candidate source set Ω simultaneously in ONE
    // run. By QUICK linearity the aggregate dose is the exact sum of per-cell
    // bursts. n_srcmask = |Ω| (set-cell count), used to scale the burst mass budget
    // and clearance test. Default nullptr/0 ⇒ the original single-cell path,
    // byte-identical behavior (CPU host pointer; GPU mask upload is a TODO).
    const uint8_t* srcmask = nullptr;   // host mask (CPU backend reads this directly)
    long           n_srcmask = 0;
    uint8_t*       d_srcmask = nullptr;  // device mirror (CUDA backend; uploaded by gpu::sync_source_mask)
    int     inAx, inSide, outAx, outSide;
    float   uInX, uInY, uInZ;

    // ── Per-face inflow/outflow classification (oblique-wind fix) ──────────────
    // Each vertical domain face is INLET / OUTFLOW / SYMMETRY by the sign of the
    // mean normal flux dot(u_mean, inward-normal). For oblique wind BOTH upwind
    // faces (−x and −y) are inlets, so the injected cross-stream has a real fetch
    // instead of reflecting off a symmetry wall. At wind 0° this degenerates to
    // the original single −x inlet + ±y symmetry. Order: [−x, +x, −y, +y].
    // Values: 0=SYM, 1=INLET, 2=OUTFLOW  (see bc_mode below).
    int     face_bc[4] = {1, 2, 0, 0};
    int     per_face   = 1;            // 1 = per-face BC (default); 0 = legacy single-inlet (WIND_BC=legacy)

    // Rough-wall (log-law) ground slip, lattice units. When cfg.wall_model==1 the
    // domain-floor bounce-back becomes a moving-wall bounce-back at this velocity,
    // imposing the ABL wall shear stress τw=ρu*² instead of no-slip (fixes the
    // ABL horizontal-homogeneity drift; Blocken, Stathopoulos & Carmeliet 2007).
    float   u_wallX, u_wallY;

    // ── Diagnostic / experiment toggles (env-gated; default = ORIGINAL behavior) ──
    // These are read from the environment in the Solver ctor and default to the
    // unpatched behavior when unset, so the production battery is unaffected.
    //   lateral_bc: 0 = specular symmetry on ±y (default/original, T5-tuned);
    //               1 = OPEN zero-gradient (copy the interior y-neighbor, like the
    //                   outlet) so a mean cross-stream (oblique-wind) component can
    //                   advect OUT of ±y instead of being reflected back in.
    //                   Env LATERAL_BC=symmetry|open.  (Diagnosis experiment #6.)
    int   lateral_bc = 0;
    //   dump_umax: when true, the spin-up loop reports the location and value of
    //   max|u| (argmax over the instantaneous field) and the first non-finite cell
    //   at every check_interval and at divergence, with a boundary/corner label —
    //   so a blow-up is self-locating instead of a bare "diverged". Env
    //   DUMP_UMAX_LOC=1.  (Diagnosis experiment #5.)
    bool  dump_umax = false;

    // Time-resolved snapshot callback (viz). Fired during Phase B when
    // cfg.snapshot_every>0. Plain POD function-pointer + context, deliberately NOT
    // the standard-library callable wrapper (its variadic template trips nvcc's
    // cudafe++ front-end on some CUDA/gcc combinations), so no header nvcc parses
    // carries that type. The host driver wraps its capturing lambda behind this POD
    // ABI (see forward_city.cpp / Solver::set_snapshot_callback). nullptr = off.
    // Signature: (ctx, phaseB_step, global_step, scalar_buffer, is_release, is_end).
    void (*snapshot_cb)(void*,int,int,int,bool,bool) = nullptr;
    void*  snapshot_ctx = nullptr;
};

// ── GPU wrapper functions (implemented in lbm_kernels.cu) ──────────

namespace gpu {

void alloc(Solver::Impl& I);
void free_all(Solver::Impl& I);

void upload_geometry(Solver::Impl& I,
                     const uint8_t* h_tp, const float* h_pm, const float* h_inh,
                     const float* h_dvel);

// Ensure the per-cell source-distribution mask (Impl::srcmask, host) is resident
// where the active backend's scalar kernel reads it. CUDA: uploads to Impl::d_srcmask.
// CPU: no-op (the kernel reads the host pointer directly). Safe to call every run.
void sync_source_mask(Solver::Impl& I);

void init_equilibrium(Solver::Impl& I);

// One solver timestep: macroscopic → flow → (optionally) scalar
void step(Solver::Impl& I, int cur, int nxt, bool with_scalar);

// Save current velocity for convergence check
void save_checkpoint(Solver::Impl& I);

// RMS velocity change across all fluid cells (for convergence)
float rms_velocity_delta(Solver::Impl& I);

// Max |u| and max C from final state
void compute_results(Solver::Impl& I, int gfin,
                     float& max_vel, float& max_conc);

// Export velocity slice to host arrays
void export_velocity_slice(Solver::Impl& I, int z_slice,
                           float* h_ux, float* h_uy, int nx, int ny);

// Download the frozen MEAN flow (time-averaged velocity ux_sum/n_avg, falling
// back to the instantaneous field if no averaging was accumulated) and the eddy
// viscosity nut to host arrays of length N. Feeds the reverse (adjoint) solver.
    void download_tiac(Solver::Impl& I, float* h_cint);
void download_mean_flow(Solver::Impl& I, float* h_ux, float* h_uy,
                        float* h_uz, float* h_nut);
// Download the frozen MEAN density (rho_sum/n_avg, falling back to instantaneous
// rho). Mean pressure p = c_s²·ρ (c_s²=1/3 LU) → pressure coefficient Cp.
void download_density(Solver::Impl& I, float* h_rho);

// Download the INSTANTANEOUS macroscopic velocity (I.ux/uy/uz — the live field,
// NOT the time-average) to host arrays of length N. Mirror of download_mean_flow
// but for the current step; feeds the divergence-location diagnostic (dump_umax).
void download_instant_velocity(Solver::Impl& I, float* h_ux, float* h_uy, float* h_uz);

// Particulate deposition + mass budget
double total_deposited(Solver::Impl& I);            // Σ deposited mass (LU)
double total_airborne(Solver::Impl& I, int gbuf);   // Σ suspended scalar C (LU)
void   export_deposition_slice(Solver::Impl& I, int z_slice,
                               float* h_dep, int nx, int ny);
void   export_tiac_slice(Solver::Impl& I, int z_slice,
                         float* h_cint, int nx, int ny);

// Warm restart: move the flow distribution f between host and device.
void   download_flow(Solver::Impl& I, int buf, float* h_f);   // 19*N floats from f[buf]
void   upload_flow(Solver::Impl& I, const float* h_f);        // → both f[0] and f[1]
void   export_scalar_slice(Solver::Impl& I, int z_slice,
                           float* h_C, int nx, int ny, int gbuf);
void   export_avg_scalar_slice(Solver::Impl& I, int z_slice,
                               float* h_C, int nx, int ny);

// Time-averaging
void reset_averages(Solver::Impl& I);
void accumulate(Solver::Impl& I, int g_buf);
void compute_averaged_results(Solver::Impl& I,
                              float& max_vel, float& max_conc);
void export_avg_velocity_slice(Solver::Impl& I, int z_slice,
                               float* h_ux, float* h_uy, int nx, int ny);

} // namespace gpu
} // namespace lbm
