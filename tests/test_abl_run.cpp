// Short stability test: empty tall channel, ABL turbulent inlet ON, ~400 steps.
#include "lbm_solver.h"
#include <cstdio>
#include <cmath>
#include <vector>
#include <cstdint>
int main(){
    const int nx=48,ny=24,nz=32; int N=nx*ny*nz;
    std::vector<uint8_t> type(N,0); std::vector<float> perm(N,0),inh(N,0),dv(N,0);
    auto idx=[&](int x,int y,int z){return z*ny*nx+y*nx+x;};
    for(int y=0;y<ny;++y)for(int x=0;x<nx;++x) type[idx(x,y,0)]=1; // ground
    lbm::Config c{};
    c.nx=nx;c.ny=ny;c.nz=nz;c.cell_size=4.0f;c.U_inlet=5.0f;c.wind_angle=0.0f;
    c.nu_phys=1.5e-5f;c.Cw=0.325f;c.Sc_t=0.7f;c.D_mol=1e-5f;
    c.source_x=-100;c.source_y=-100;c.source_z=-100;c.Q_source=0;
    c.particle_diam=0;c.particle_density=0;
    c.max_steps=2000;c.check_interval=100;c.conv_threshold=1e-6f;
    c.max_warmup=400;c.avg_threshold=1e-9f;c.release_time=0;
    // ── enable the sheared turbulent ABL inlet ──
    c.inlet_profile=1; c.abl_z0=0.7f; c.abl_zref=40.0f; c.abl_Lturb=40.0f; c.abl_nmodes=100;
    c.abl_sigu_ratio=2.5f; c.abl_sigv_ratio=1.9f; c.abl_sigw_ratio=1.25f;
    lbm::Solver s(c);
    s.load_geometry(type.data(),perm.data(),inh.data(),dv.data());
    auto r=s.run();
    printf("\n[TEST] steps=%d  max|u|=%.5f LU  finite=%s\n",
           r.steps_total, r.max_velocity, std::isfinite(r.max_velocity)?"YES":"NO");
    s.export_velocity("abl_vel_z_mid.bin", nz/2);
    return std::isfinite(r.max_velocity)&&r.max_velocity<0.3f ? 0:1;
}
