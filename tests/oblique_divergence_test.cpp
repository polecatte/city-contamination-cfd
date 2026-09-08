// oblique_divergence_test.cpp — isolate the T_mass oblique-wind divergence.
//
// WHY THIS EXISTS
//   `airflow_validation all` crashed with "FATAL: flow diverged (RMS=NaN/inf)"
//   on the SECOND 336x176x96 solve. That grid is T_mass's cube domain, and
//   T_mass sweeps wind {0,15,30,45}deg. 0deg is stable; the first oblique angle
//   is what diverged. The suspected cause: the single x-normal inlet imposes a
//   cross-stream velocity uInY=u_lb*sin(theta) for theta!=0, but the +/-y
//   boundaries are SPECULAR SYMMETRY planes (zero cross-stream flux) — the two
//   are mutually inconsistent for any oblique angle, so the injected y-momentum
//   piles up against the +y plane / inlet corner and blows up.
//
//   This driver runs ONE controlled case per process (the solver hard-exits(2)
//   on divergence, so a multi-case loop would die at the first one — exactly
//   what bit the overnight run). run_oblique_diagnosis.sh loops over the seven
//   experiments, invoking this binary once per case and reading its exit code
//   (0 = finite, 2 = diverged) plus the [DIAG:...] location line it prints.
//
// BUILD (backend-agnostic — same source for both):
//   CPU : g++  -O3 -std=c++17 -fopenmp oblique_divergence_test.cpp kernels.o solver.o -o oblique_test_cpu
//   GPU : g++  -O3 -std=c++17 -fopenmp -DWITH_LBM oblique_divergence_test.cpp kernels.o solver.o -L$CUDA_LIB -lcudart -o oblique_test_gpu
//   (kernels.o from lbm_kernels_cpu.cpp for CPU, from lbm_kernels.cu for GPU.)
//
// USAGE:
//   oblique_test --deg D [--H n] [--empty] [--rfg0] [--open] [--spin n]
//   NU_FLOOR / COLLISION / WALL_MODEL are read from the environment (as in the
//   production battery). The driver sets ABL_RFG/LATERAL_BC/DUMP_UMAX_LOC from
//   its flags before constructing the Solver.

#include "lbm_solver.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
using std::vector;

static const uint8_t FLUID = 0, GROUND = 1;   // matches lbm_kernels_cpu.cpp enum
static inline int IDX(int x,int y,int z,int nx,int ny){ return z*ny*nx + y*nx + x; }

// ── geometry helpers (verbatim from airflow_validation.cpp) ──────────────────
static void add_ground(vector<uint8_t>& t,int nx,int ny){
    for(int y=0;y<ny;++y) for(int x=0;x<nx;++x) t[IDX(x,y,0,nx,ny)] = GROUND;
}
static void add_cube(vector<uint8_t>& t,int nx,int ny,int nz,int x0,int y0,int H){
    for(int z=1;z<=H&&z<nz;++z) for(int y=y0;y<y0+H&&y<ny;++y) for(int x=x0;x<x0+H&&x<nx;++x)
        t[IDX(x,y,z,nx,ny)] = GROUND;
}

// base_cfg mirror (matches airflow_validation.cpp::base_cfg for the ABL cube).
static lbm::Config base_cfg(int nx,int ny,int nz,float wind_angle){
    lbm::Config c{};
    c.nx=nx; c.ny=ny; c.nz=nz; c.cell_size=4.0f; c.U_inlet=5.0f; c.wind_angle=wind_angle;
    c.nu_phys=1.5e-5f; c.Cw=0.325f; c.Sc_t=0.7f; c.D_mol=1e-5f;
    c.source_x=-100; c.source_y=-100; c.source_z=-100; c.Q_source=0.0f;   // flow only
    c.particle_diam=0; c.particle_density=0; c.scalar_advection=0;
    c.max_steps=2000000; c.check_interval=500; c.conv_threshold=1e-4f;
    c.release_time=2.0f; c.scalar_extra_steps=0;
    // ABL log-law + RFG inlet (turbulence toggled off via ABL_RFG env if requested)
    c.inlet_profile=1; c.abl_z0=0.7f; c.abl_zref=40.0f; c.abl_Lturb=40.0f;
    c.abl_nmodes=100; c.abl_sigu_ratio=2.5f; c.abl_sigv_ratio=1.9f; c.abl_sigw_ratio=1.25f;
    return c;
}

int main(int argc,char** argv){
    double deg = 0.0; int H = 8; bool empty=false, rfg0=false, open=false;
    int spin = 3000;                       // Phase-A1 steps (divergence is caught here)
    for(int i=1;i<argc;++i){
        if(!strcmp(argv[i],"--deg")   && i+1<argc) deg=atof(argv[++i]);
        else if(!strcmp(argv[i],"--H")&& i+1<argc) H=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--spin")&&i+1<argc) spin=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--empty")) empty=true;
        else if(!strcmp(argv[i],"--rfg0"))  rfg0=true;
        else if(!strcmp(argv[i],"--open"))  open=true;
        else { fprintf(stderr,"unknown arg %s\n",argv[i]); return 3; }
    }

    // Cube domain sizing (COST-732 up5/down15/lat5/top5), same as make_cube().
    int nx=(5+1+15)*H, ny=(5*2+1)*H, nz=(5+1)*H;
    vector<uint8_t> t(nx*ny*nz, FLUID);
    add_ground(t,nx,ny);
    if(!empty) add_cube(t,nx,ny,nz, /*x0*/5*H, /*y0*/5*H, H);

    // Translate flags into the exact env vars the Solver reads (single source of
    // truth). NU_FLOOR / COLLISION / WALL_MODEL are inherited from the caller.
    setenv("DUMP_UMAX_LOC","1",1);                 // always locate the blow-up
    if(rfg0) setenv("ABL_RFG","0",1);
    setenv("LATERAL_BC", open ? "open" : "symmetry", 1);

    lbm::Config c = base_cfg(nx,ny,nz,(float)(deg*M_PI/180.0));
    c.spinup_steps = spin;                         // bounded, deterministic Phase A1
    c.max_warmup   = spin + 1500;                  // short averaging tail if it survives
    c.avg_steps    = 500;
    c.avg_threshold= 5e-3f;

    const char* nuf = getenv("NU_FLOOR"); const char* col = getenv("COLLISION");
    const char* wm  = getenv("WALL_MODEL");
    printf("================================================================\n");
    printf("[oblique] deg=%.0f  H=%d  grid=%dx%dx%d  cube=%s  rfg=%s  latBC=%s\n",
           deg,H,nx,ny,nz, empty?"OFF":"ON", rfg0?"OFF(mean-only)":"ON",
           open?"open":"symmetry");
    printf("[oblique] NU_FLOOR=%s  COLLISION=%s  WALL_MODEL=%s  spin=%d\n",
           nuf?nuf:"(default 1e-2)", col?col:"(default mrt)", wm?wm:"(default 0)", spin);
    printf("================================================================\n");
    fflush(stdout);

    // If Phase A diverges, the solver prints [DIAG:DIVERGED ...] then exit(2);
    // this driver never returns in that case. If it survives, we grade the field.
    vector<uint8_t> tmp = t;
    lbm::Solver s(c);
    vector<float> perm(nx*ny*nz,0), inh(nx*ny*nz,0), dvel(nx*ny*nz,0);
    s.load_geometry(t.data(), perm.data(), inh.data(), dvel.data());
    lbm::Result r = s.run();

    bool finite = std::isfinite(r.max_velocity) && std::isfinite(r.max_concentration);
    printf("[oblique] SURVIVED Phase A. max|u|=%.4f  finite=%d  (steps=%d)\n",
           r.max_velocity, (int)finite, r.steps_total);
    printf("RESULT deg=%.0f H=%d empty=%d rfg0=%d open=%d -> %s max_u=%.4f\n",
           deg,H,(int)empty,(int)rfg0,(int)open, finite?"FINITE":"NONFINITE", r.max_velocity);
    fflush(stdout);
    return finite ? 0 : 2;
}
