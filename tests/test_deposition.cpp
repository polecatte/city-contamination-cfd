// test_deposition.cpp — Validation for gravitational settling + surface deposition
//
// Geometry: a flat channel (deposition floor, reflecting ceiling) with a steady
// elevated point source. Wind blows +x; particles settle while advecting and
// deposit on the floor.
//
// NOTE ON SCOPE: the underlying D3Q7 scalar uses an equilibrium-drift advection
// with the project's diffusivity floor, giving a high cell-Péclet number
// (~u_lb/D_floor ≈ 60). For a *sharp elevated point source* this base scheme
// produces bounded central-difference oscillations (small negative C) that exist
// WITH OR WITHOUT this feature. We therefore validate settling + deposition with
// metrics that are robust to that pre-existing behaviour:
//   (1) No mass creation:        deposited ≤ emitted, always.
//   (2) Deposition is monotone:  larger surface v_d ⇒ more captured mass.
//   (3) Settling drives capture: a settling particle deposits far more than a
//                                passive tracer, and its deposition footprint
//                                lands near the ballistic distance h·U/w_s.
//
// Build (CPU):
//   g++ -O3 -std=c++17 -fopenmp -c lbm_kernels_cpu.cpp -o kernels.o
//   g++ -O3 -std=c++17 -c lbm_solver.cpp -o solver.o
//   g++ -O3 -std=c++17 -c test_deposition.cpp -o test_deposition.o
//   g++ kernels.o solver.o test_deposition.o -fopenmp -o test_deposition
//   ./test_deposition

#include "lbm_solver.h"
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>

using std::vector;

static const int   NX=160, NY=4, NZ=24;
static const int   X_SRC=20, Z_SRC=18;
static const float U=2.0f;
static const double U_LB = 0.1/std::sqrt(3.0);

struct Out { double emitted, deposited, x_mean, dep_mass; float w_s; };

static Out run_case(float dp,float rho_p,float ground_vd,const char* tag){
    int N=NX*NY*NZ;
    vector<uint8_t> type(N,0); vector<float> perm(N,0),inh(N,0),dvel(N,0);
    auto idx=[&](int x,int y,int z){return z*NY*NX+y*NX+x;};
    for(int y=0;y<NY;++y) for(int x=0;x<NX;++x){
        type[idx(x,y,0)]=1;   dvel[idx(x,y,0)]=ground_vd;   // deposition floor
        type[idx(x,y,NZ-1)]=1;                              // reflecting ceiling
    }
    lbm::Config cfg{};
    cfg.nx=NX; cfg.ny=NY; cfg.nz=NZ; cfg.cell_size=1.0f;
    cfg.U_inlet=U; cfg.wind_angle=0.0f;
    cfg.nu_phys=1.5e-5f; cfg.Cw=0.325f; cfg.Sc_t=0.7f; cfg.D_mol=1e-5f;
    cfg.source_x=X_SRC; cfg.source_y=NY/2; cfg.source_z=Z_SRC; cfg.Q_source=1.0f;
    cfg.particle_diam=dp; cfg.particle_density=rho_p;
    cfg.max_steps=6000; cfg.check_interval=500; cfg.conv_threshold=1e-9f;
    cfg.scalar_extra_steps=0; cfg.max_warmup=0; cfg.avg_threshold=0; cfg.avg_steps=0;

    printf("\n──── %s ────\n", tag);
    lbm::Solver s(cfg);
    s.load_geometry(type.data(),perm.data(),inh.data(),dvel.data());
    auto r=s.run();
    s.export_deposition("dep_tmp.bin",1);

    vector<float> dep(NX*NY);
    FILE* f=fopen("dep_tmp.bin","rb"); int h[2];
    size_t rr=fread(h,sizeof(int),2,f);(void)rr;
    rr=fread(dep.data(),sizeof(float),(size_t)NX*NY,f);(void)rr; fclose(f);

    double sx=0,sw=0;
    for(int y=0;y<NY;++y) for(int x=0;x<NX;++x){ float d=dep[y*NX+x]; sx+=(double)(x-X_SRC)*d; sw+=d; }
    Out o; o.emitted=r.total_emitted; o.deposited=r.total_deposited;
    o.dep_mass=sw; o.x_mean=(sw>0)?sx/sw:0; o.w_s=r.w_settle_phys;
    return o;
}

int main(){
    printf("═══════════════════════════════════════════════\n");
    printf("  Settling + Deposition Validation\n");
    printf("═══════════════════════════════════════════════\n");

    // Coarse heavy particle so settling acts visibly within a small domain.
    const float rho_p=2600.f;
    Out passive = run_case(0.f,    rho_p, 0.01f, "passive tracer");
    Out p40     = run_case(40e-6f, rho_p, 0.01f, "40 µm  (partial settling)");
    Out p80     = run_case(80e-6f, rho_p, 0.01f, "80 µm  (full settling)");

    printf("\n═══════════════════════════════════════════════\n");
    printf("  (1) No mass creation (deposited ≤ emitted)\n");
    bool c1 = passive.deposited<=passive.emitted*1.0001
           && p40.deposited <=p40.emitted *1.0001
           && p80.deposited <=p80.emitted *1.0001;
    printf("    passive: dep=%.3e / emit=%.3e\n", passive.deposited, passive.emitted);
    printf("    40 µm  : dep=%.3e / emit=%.3e\n", p40.deposited,    p40.emitted);
    printf("    80 µm  : dep=%.3e / emit=%.3e\n", p80.deposited,    p80.emitted);
    printf("    %s\n", c1?"✓ no run deposits more than it emitted":"✗ mass created");

    printf("\n  (2) Deposition monotone in settling velocity\n");
    printf("    w_s: passive=%.3g  40µm=%.3g  80µm=%.3g m/s\n", passive.w_s, p40.w_s, p80.w_s);
    bool c2 = (p80.deposited > p40.deposited*1.05) && (p40.deposited > passive.deposited*5.0);
    printf("    dep: passive=%.3e < 40µm=%.3e < 80µm=%.3e ?  %s\n",
           passive.deposited, p40.deposited, p80.deposited, c2?"✓ yes":"✗ no");

    printf("\n  (3) Settling footprint near ballistic distance (80 µm)\n");
    double ballistic = (double)(Z_SRC-1) * U / p80.w_s;   // h·U/w_s, cells
    bool c3 = (p80.x_mean > 0.3*ballistic && p80.x_mean < 3.0*ballistic);
    printf("    w_s=%.3g m/s (w/U=%.2f)\n", p80.w_s, p80.w_s/U);
    printf("    ⟨x_dep⟩=%.1f cells   ballistic h·U/w_s=%.1f cells\n", p80.x_mean, ballistic);
    printf("    %s\n", c3?"✓ deposition footprint lands near the ballistic estimate"
                        :"✗ settling footprint off");

    printf("\n═══════════════════════════════════════════════\n");
    bool pass=c1&&c2&&c3;
    printf("  %s\n", pass?"✓ PASS":"✗ FAIL");
    printf("═══════════════════════════════════════════════\n");
    return pass?0:1;
}
