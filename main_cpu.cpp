// main_cpu.cpp — Small-domain test: city → voxelize → LBM solver
//
// Build (macOS with Homebrew OpenMP):
//   clang++ -O3 -std=c++17 -Xpreprocessor -fopenmp -I$(brew --prefix libomp)/include \
//           -c lbm_kernels_cpu.cpp -o kernels.o
//   clang++ -O3 -std=c++17 -c lbm_solver.cpp -o solver.o
//   clang++ -O3 -std=c++17 -c main_cpu.cpp -o main.o
//   clang++ kernels.o solver.o main.o -L$(brew --prefix libomp)/lib -lomp -o solver
//
// Build (Linux):
//   g++ -O3 -std=c++17 -fopenmp -c lbm_kernels_cpu.cpp -o kernels.o
//   g++ -O3 -std=c++17 -c lbm_solver.cpp -o solver.o
//   g++ -O3 -std=c++17 -c main_cpu.cpp -o main.o
//   g++ kernels.o solver.o main.o -fopenmp -o solver
//
// Run:
//   ./solver

#include "city_builder7.h"
#include "voxelize.h"
#include "lbm_solver.h"
#include <cstdio>
#include <cmath>

int main() {
    printf("═══════════════════════════════════════════════\n");
    printf("  Urban LBM Dispersion Solver\n");
    printf("═══════════════════════════════════════════════\n\n");

    // ── City parameters ──
    city::Params p{};
    p.city_w = 600; p.city_h = 600;
    p.block_w = 48; p.block_d = 24;
    p.base_height = 12;
    p.cbd_peak = 80; p.cbd_decay = 8e-6;
    p.cbd_aspect = 1.0; p.cbd_angle = 0;   // ignored (circular model)
    p.biz_inner_frac = 0.05; p.biz_aspect = 1.0;   // biz_aspect ignored (circular)

    p.park_centrality = 0.5; p.park_fraction = 0.10;
   // mild deterministic zoning non-uniformity
    p.roughness = 0.15; p.road_w_x=20; p.road_w_y=20;
    p.population_total = 20000;
    p.wind_direction = 0;

    // ── Domain sizing ──
    // Standard: 200m uniform buffer (5 × 40m ref height), power-of-2 grid
    city::default_buffers(p);

    // Override individual sides if needed, e.g.:
    // p.buf_xp = 400;  // extra downstream buffer for wake recovery

    // Source upwind of city (in buffer zone)
    p.source_x = 50; p.source_y = p.buf_yn + p.city_h * 0.5;
    city::ensure_source_buffer(p);

    // Compute domain size from buffers
    city::compute_domain(p);

    // CBD and business center at city center (domain coords)
    double city_cx = p.buf_xn + p.city_w * 0.5;
    double city_cy = p.buf_yn + p.city_h * 0.5;
    p.cbd_x = city_cx; p.cbd_y = city_cy;
    p.biz_center_x = city_cx; p.biz_center_y = city_cy;

    printf("[City] Generating...\n");
    auto r = city::generate(p);
    city::print_summary(p, r);

    printf("\n[Voxel] Voxelizing...\n");
    auto g = city::voxelize(p, r);
    city::print_voxel_summary(g);

    // Export city for Python renderer
    city::export_city("city_export.txt", p, r);
    printf("[Export] city_export.txt\n");

    // Verify grid dimensions
    printf("\n[Grid] %d × %d × %d\n", g.nx, g.ny, g.nz);

    // ── LBM solver ──
    lbm::Config cfg{};
    cfg.nx = g.nx; cfg.ny = g.ny; cfg.nz = g.nz;
    cfg.cell_size = g.cell_size;
    cfg.U_inlet = 5.0f;           // 5 m/s wind
    cfg.wind_angle = 0.0f;        // +x direction
    cfg.nu_phys = 1.5e-5f;
    cfg.Cw = 0.325f;
    cfg.Sc_t = 0.7f;
    cfg.D_mol = 1e-5f;
    cfg.source_x = p.source_x;
    cfg.source_y = p.source_y;
    cfg.source_z = 4.0f;          // 4m above ground (one cell)
    cfg.Q_source = 1.0f;

    // Particulate: PM10-ish coarse dust (settles + deposits per-surface).
    // Set particle_diam = 0 for a passive (non-settling) tracer gas.
    cfg.particle_diam    = 10.0e-6f;   // 10 µm
    cfg.particle_density = 1800.0f;    // kg/m³ (mineral dust)

    cfg.max_steps = 20000;
    cfg.check_interval = 500;
    cfg.conv_threshold = 1e-4f;
    cfg.scalar_extra_steps = 2000;

    // Run structure: warm up flow to steadiness, then release the source.
    cfg.max_warmup = 20000;       // cap on flow warm-up
    cfg.avg_threshold = 2e-3f;    // flow is "steady" when RMS(Δu)/U < 0.2%
    cfg.avg_steps = 0;            // (legacy averaging off)
    cfg.release_time = 300.0f;    // release/disperse for 300 s of physical time

    // ── Sheared turbulent ABL inlet (abl_inlet.h) ──────────────────────────
    // Replaces uniform plug flow with a neutral-ABL log-law mean (Richards &
    // Hoxey 1993) + Random Flow Generation turbulence (Kraichnan 1970; Smirnov,
    // Shi & Celik 2001); sigma/u* ratios from Panofsky & Dutton (1984).
    cfg.inlet_profile  = 1;
    cfg.abl_z0         = 0.7f;    // urban aerodynamic roughness length (m)
    cfg.abl_zref       = 40.0f;   // height at which mean speed = U_inlet
    cfg.abl_Lturb      = 40.0f;   // turbulence integral length scale (m)
    cfg.abl_nmodes     = 100;
    cfg.abl_sigu_ratio = 2.5f; cfg.abl_sigv_ratio = 1.9f; cfg.abl_sigw_ratio = 1.25f;

    printf("\n");
    lbm::Solver solver(cfg);
    solver.load_geometry(g.type.data(), g.perm.data(), g.inh.data(), g.dep_vel.data());

    printf("\n[Solve] Running wind angle = 0° (+x)...\n");
    auto result = solver.run();

    printf("\n═══════════════════════════════════════════════\n");
    printf("  RESULT\n");
    if (result.avg_start >= 0)
        printf("  Time-averaged from step %d (%d samples)\n",
               result.avg_start, result.steps_total - result.avg_start);
    else
        printf("  Flow stage ended at step %d\n", result.steps_flow);
    printf("  Total steps: %d\n", result.steps_total);
    printf("  Max velocity: %.4f LU\n", result.max_velocity);
    printf("  Max concentration: %.4e\n", result.max_concentration);
    printf("═══════════════════════════════════════════════\n");

    // Export velocity fields for Python renderer
    solver.export_velocity("velocity_z1.bin", 1);
    solver.export_velocity("velocity_z2.bin", 2);

    // Export ground-level deposition map (z=1 = first fluid layer above ground)
    solver.export_deposition("deposition_z1.bin", 1);
    solver.export_concentration("concentration_z1.bin", 1);
    if (result.total_emitted > 0.0)
        printf("  Deposited fraction: %.1f%% of emitted mass captured on surfaces\n",
               100.0 * result.total_deposited / result.total_emitted);

    printf("\nTo visualize:\n");
    printf("  python3 render_city.py\n");
    printf("  (or see inline script in main_cpu.cpp comments)\n");

    // Inline Python visualization script (copy-paste into terminal):
    // python3 -c "
    // from render_city import *; import matplotlib.pyplot as plt, struct, numpy as np
    // bl,m = load_city('city_export.txt'); cell=m.get('cell',4.0)
    // with open('velocity_z1.bin','rb') as f:
    //     nx,ny=struct.unpack('2i',f.read(8))
    //     ux=np.frombuffer(f.read(nx*ny*4),dtype=np.float32).reshape(ny,nx)
    //     uy=np.frombuffer(f.read(nx*ny*4),dtype=np.float32).reshape(ny,nx)
    // fig,ax=plt.subplots(figsize=(10,10))
    // draw_2d(ax,bl,m,cell=cell)
    // draw_velocity(ax,ux,uy,m,cell=cell,stride=4)
    // plt.savefig('result.png',dpi=150); print('Saved result.png')
    // "

    return 0;
}
