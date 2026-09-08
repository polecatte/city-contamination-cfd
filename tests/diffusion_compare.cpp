// diffusion_compare.cpp — quantify the numerical-diffusion penalty of the
// linear-upwind objective transport (the reverse solver's forward operator)
// against a van-Leer TVD reference and an analytical standard.
//
// MODES
//   bench (default): a controlled physical-standard test. A Gaussian blob is
//     advected in a UNIFORM DIAGONAL flow with ZERO physical diffusion. The exact
//     solution is pure translation — the blob keeps its width — so ANY spreading
//     is purely numerical diffusion. Diagonal flow is used so first-order upwind's
//     diffusion is not artificially hidden by grid alignment (which would vanish
//     at CFL=1). We advect the SAME blob with (a) the linear-upwind operator used
//     by the objective/reverse solver (adj::fwd_step with D=0), and (b) a van-Leer
//     TVD advection, then measure the cross-stream spread growth and back out an
//     effective numerical diffusivity D_num = (σ⊥² − σ0²)/(2·t) for each. The
//     analytical reference has D_num = 0. This is the diffusion penalty, in the
//     same lattice units as the eddy diffusivity nut/Sc_t, so the two are directly
//     comparable.
//
//   city <city_m> <release_s> <max_warmup>: runs the PRODUCTION van Leer D3Q7
//     scalar (inside the LBM) and the linear-upwind forward on the SAME frozen
//     city flow from the SAME point source, and writes both z=1 concentration
//     fields + a centreline/σ profile so the schemes can be compared on real
//     geometry. (Heavy; intended for the GPU build.)
//
// Output: diffusion_bench.csv / diffusion_city.csv (+ fields), plotted by
// plot_diffusion.py.

#include "adjoint_transport.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
using namespace adj;

// ── van-Leer flux limiter and a dimensionally-split TVD advection ────────────
static inline float vanleer(float r){ float a=std::fabs(r); return (r+a)/(1.f+a); }

// 1D van-Leer advection sweep along +axis with uniform positive velocity `vel`.
// Explicit, second-order TVD (MUSCL). Zero-gradient ends (blob kept interior).
// stride/len describe the 1D line within the flat 3D array.
static void vl_sweep_line(float* C, int len, int stride, float vel, float dt){
    if(len<2) return;
    float nu = vel*dt;                         // CFL (dx=1)
    std::vector<float> f(len, 0.f);            // flux at face i (=i+1/2)
    auto AT=[&](int i)->float& { return C[(size_t)i*stride]; };
    for(int i=0;i<len-1;++i){
        float Ci=AT(i), Cip=AT(i+1);
        float Cim=(i>0)?AT(i-1):Ci;
        float dCp=Cip-Ci;
        float r=(std::fabs(dCp)>1e-30f)?(Ci-Cim)/dCp:0.f;
        f[i]=vel*( Ci + 0.5f*(1.f-nu)*vanleer(r)*dCp );  // 2nd-order TVD upwind flux
    }
    // f[len-1] (outflow end): first-order (zero-gradient)
    f[len-1]=vel*AT(len-1);
    std::vector<float> Cn(len);
    for(int i=0;i<len;++i){
        float fl=(i>0)?f[i-1]:vel*AT(0)*0.f;   // inflow end: zero inflow
        Cn[i]=AT(i)-nu*(f[i]-fl);
    }
    for(int i=0;i<len;++i) AT(i)=Cn[i];
}

// one van-Leer step over the 3D field for uniform flow (vx,vy,0), vx,vy>=0
static void vanleer_step(std::vector<float>& C,int nx,int ny,int nz,
                         float vx,float vy,float dt){
    auto IDX=[&](int x,int y,int z){ return (size_t)(z*ny+y)*nx+x; };
    // x sweeps (stride 1)
    for(int z=0;z<nz;++z) for(int y=0;y<ny;++y)
        vl_sweep_line(&C[IDX(0,y,z)], nx, 1, vx, dt);
    // y sweeps (stride nx)
    for(int z=0;z<nz;++z) for(int x=0;x<nx;++x)
        vl_sweep_line(&C[IDX(x,0,z)], ny, nx, vy, dt);
}

// cross-stream (perpendicular to flow (1,1)) second moment of a field in z-mid plane
static void spread_perp(const std::vector<float>& C,int nx,int ny,int nz,
                        double& mass,double& sigma_perp,double& along,double& cross_c){
    auto IDX=[&](int x,int y,int z){ return (size_t)(z*ny+y)*nx+x; };
    int z=nz/2; const double inv2=1.0/std::sqrt(2.0);
    double m=0,sη=0,sη2=0,sξ=0;
    for(int y=0;y<ny;++y) for(int x=0;x<nx;++x){
        double c=C[IDX(x,y,z)]; if(c<=0) continue;
        double eta=( y - x )*inv2;     // perpendicular to (1,1)
        double xi =( x + y )*inv2;     // along flow
        m+=c; sη+=c*eta; sη2+=c*eta*eta; sξ+=c*xi;
    }
    mass=m;
    double etac=(m>0)?sη/m:0.0;
    sigma_perp=(m>0)?std::sqrt(std::max(0.0, sη2/m - etac*etac)):0.0;
    along=(m>0)?sξ/m:0.0; cross_c=etac;
}

static int run_bench(){
    const int nx=200, ny=200, nz=8, N=nx*ny*nz;
    std::vector<uint8_t> tp(N,(uint8_t)FLUID);   // all fluid (interior advection test)
    std::vector<float> ux(N,0.f),uy(N,0.f),uz(N,0.f),nut(N,0.f),vdep(N,0.f);
    const float Ud=0.10f;                        // uniform diagonal velocity per axis (LU)
    for(int i=0;i<N;++i){ ux[i]=Ud; uy[i]=Ud; }  // 45° to the grid

    // pure-advection field (no physical or turbulent diffusion, no settling)
    Field F{nx,ny,nz,N, tp.data(),ux.data(),uy.data(),uz.data(),nut.data(),vdep.data(),
            /*D0=*/0.f, /*Sc_t=*/0.7f, /*w_s=*/0.f, /*dt=*/0.f};
    F.dt = 2.0f;                                 // per-axis CFL = Ud*dt = 0.20

    // initial Gaussian blob (uniform in z) at (40,40)
    auto IDX=[&](int x,int y,int z){ return (size_t)(z*ny+y)*nx+x; };
    const double x0=40,y0=40,sig0=4.0;
    std::vector<float> C0(N,0.f);
    for(int z=0;z<nz;++z) for(int y=0;y<ny;++y) for(int x=0;x<nx;++x){
        double r2=(x-x0)*(x-x0)+(y-y0)*(y-y0);
        C0[IDX(x,y,z)]=(float)std::exp(-r2/(2*sig0*sig0));
    }
    double m0,s0,al0,cc0; spread_perp(C0,nx,ny,nz,m0,s0,al0,cc0);

    // advect until the blob has crossed ~120 cells diagonally
    int T=(int)std::round(120.0/(Ud*F.dt));      // steps to travel ~120 cells in x
    double t_phys=T*F.dt;

    // (a) linear-upwind (the objective/reverse-solver operator), D=0 → pure upwind
    std::vector<float> Cu=C0, Cn(N);
    for(int t=0;t<T;++t){ fwd_step(F,Cu.data(),Cn.data()); std::swap(Cu,Cn); }
    double mu,su,alu,ccu; spread_perp(Cu,nx,ny,nz,mu,su,alu,ccu);

    // (b) van-Leer TVD reference
    std::vector<float> Cv=C0;
    for(int t=0;t<T;++t) vanleer_step(Cv,nx,ny,nz,Ud,Ud,F.dt);
    double mv,sv,alv,ccv; spread_perp(Cv,nx,ny,nz,mv,sv,alv,ccv);

    // effective numerical diffusivity D_num = (σ⊥² − σ0²)/(2 t)
    double Dnum_up = (su*su - s0*s0)/(2*t_phys);
    double Dnum_vl = (sv*sv - s0*s0)/(2*t_phys);
    double Dnum_theory = Ud*1.0/2.0*(1.0 - Ud*F.dt);   // 1st-order upwind: a·dx/2·(1−CFL)

    printf("\n=== Numerical-diffusion benchmark (pure advection, exact answer = no spread) ===\n");
    printf("  grid %dx%dx%d  Ud=%.3f/axis  dt=%.2f  CFL/axis=%.2f  steps=%d  t=%.1f\n",
           nx,ny,nz,Ud,F.dt,Ud*F.dt,T,t_phys);
    printf("  initial perpendicular sigma          sigma0   = %.4f\n", s0);
    printf("  linear-upwind (objective)  sigma_up  = %.4f   mass ratio=%.4f\n", su, mu/m0);
    printf("  van-Leer TVD reference     sigma_vl  = %.4f   mass ratio=%.4f\n", sv, mv/m0);
    printf("  --- effective numerical diffusivity (lattice units) ---\n");
    printf("  D_num  linear-upwind = %.4e\n", Dnum_up);
    printf("  D_num  van-Leer      = %.4e\n", Dnum_vl);
    printf("  D_num  upwind theory = %.4e   (a*dx/2*(1-CFL))\n", Dnum_theory);
    printf("  penalty ratio  upwind/vanLeer = %.1fx\n", (Dnum_vl>0)?Dnum_up/Dnum_vl:0.0);
    printf("  For scale: a typical urban eddy diffusivity nut/Sc_t ~ 1e-2..1e-1 (LU).\n");

    FILE* f=fopen("diffusion_bench.csv","w");
    fprintf(f,"quantity,sigma0,sigma_upwind,sigma_vanleer,Dnum_upwind,Dnum_vanleer,Dnum_upwind_theory,t\n");
    fprintf(f,"value,%.6f,%.6f,%.6f,%.6e,%.6e,%.6e,%.3f\n",
            s0,su,sv,Dnum_up,Dnum_vl,Dnum_theory,t_phys);
    fclose(f);

    // cross-stream profiles through each blob centre (for plot_diffusion.py)
    auto profile=[&](const std::vector<float>& C,double along_c,const char* name){
        FILE* pf=fopen(name,"w"); fprintf(pf,"eta,C\n");
        int z=nz/2; const double inv2=1.0/std::sqrt(2.0);
        // sample along the perpendicular line through the centroid
        for(int y=0;y<ny;++y) for(int x=0;x<nx;++x){
            double xi=(x+y)*inv2; if(std::fabs(xi-along_c)>0.8) continue;
            double eta=(y-x)*inv2; fprintf(pf,"%.4f,%.6e\n",eta,(double)C[IDX(x,y,z)]);
        }
        fclose(pf);
    };
    profile(Cu,alu,"prof_upwind.csv");
    profile(Cv,alv,"prof_vanleer.csv");
    profile(C0,al0,"prof_initial.csv");
    printf("\n[bench] wrote diffusion_bench.csv, prof_{upwind,vanleer,initial}.csv\n");
    printf("[bench] plot: python3 plot_diffusion.py\n");
    return 0;
}

#ifdef WITH_LBM
#include "city_builder7.h"
#include "voxelize.h"
#include "lbm_solver.h"
#include "psd.h"
static int run_city(int argc,char**argv){
    using namespace city;
    double city_m=(argc>2)?atof(argv[2]):512.0;
    double rel_s =(argc>3)?atof(argv[3]):60.0;
    int    warm  =(argc>4)?atoi(argv[4]):4000;
    // city + buffers exactly as lab_test sets them up
    Params p{};
    p.city_w=city_m; p.city_h=city_m;
    p.block_w=48; p.block_d=24; p.base_height=12;
    p.cbd_peak=40; p.cbd_decay=8e-6; p.cbd_aspect=1; p.cbd_angle=0;
    p.biz_inner_frac=0.05; p.biz_aspect=1; p.park_centrality=0.5; p.park_fraction=0.10;
 p.roughness=0.15; p.road_w_x=20; p.road_w_y=20; p.population_total=20000;
    p.wind_direction=0;
    default_buffers(p); compute_domain(p);
    p.buf_xn=(p.Sx-p.city_w)*0.5; p.buf_yn=(p.Sy-p.city_h)*0.5;
    p.source_x=p.buf_xn-100.0; p.source_y=p.buf_yn+p.city_h*0.5;
    ensure_source_buffer(p);
    double cx=p.buf_xn+p.city_w*0.5, cy=p.buf_yn+p.city_h*0.5;
    p.cbd_x=cx; p.cbd_y=cy; p.biz_center_x=cx; p.biz_center_y=cy;
    Result r=generate(p);
    int NX=r.nx_cells,NY=r.ny_cells;
    VoxelGrid g=voxelize(p,r,true,0.0,1800.0,0.5);   // dep_dp=0 → passive tracer
    int NZ=g.nz, N3=NX*NY*NZ;
    lbm::Config c{}; c.nx=g.nx;c.ny=g.ny;c.nz=g.nz; c.cell_size=(float)CELL;
    c.U_inlet=5.f;c.wind_angle=0.f;c.nu_phys=1.5e-5f;c.Cw=0.325f;c.Sc_t=0.7f;c.D_mol=1e-5f;
    c.source_x=p.source_x;c.source_y=p.source_y;c.source_z=4.f;c.Q_source=1.f;
    c.particle_diam=0.f;c.particle_density=1800.f;            // passive tracer (no settling)
    c.max_steps=2000000;c.check_interval=500;c.conv_threshold=1e-4f;
    c.max_warmup=warm;c.avg_threshold=2e-3f;c.release_time=(float)rel_s;
    c.inlet_profile=1;c.abl_z0=0.7f;c.abl_zref=40.f;c.abl_Lturb=40.f;c.abl_nmodes=100;
    c.abl_sigu_ratio=2.5f;c.abl_sigv_ratio=1.9f;c.abl_sigw_ratio=1.25f;
    lbm::Solver s(c);
    s.load_geometry(g.type.data(),g.perm.data(),g.inh.data(),g.dep_vel.data());
    auto res=s.run();
    s.export_tiac("tiac_vanleer_z1.bin",1);                  // production van Leer field
    std::vector<float> ux,uy,uz,nut; s.copy_mean_flow_to_host(ux,uy,uz,nut);

    // linear-upwind forward from the SAME point source on the frozen flow
    std::vector<float> vdep(N3,0.f);
    Field F{NX,NY,NZ,N3,g.type.data(),ux.data(),uy.data(),uz.data(),nut.data(),vdep.data(),
            1e-3f,0.7f,s.settling_velocity_lattice(),0.f}; F.dt=stable_dt(F);
    std::vector<float> sfield(N3,0.f);
    int sx=(int)std::lround(p.source_x/city::CELL), sy=NY/2, sz=1;
    sfield[(size_t)(sz*NY+sy)*NX+sx]=1.f;
    int nstep=(int)std::min(4000.0,std::max(400.0,3.0*NX/0.05/F.dt));
    std::vector<float> tiac; std::vector<float> dummyw(N3,0.f);
    forward_objective(F,sfield,dummyw,nstep,&tiac);          // returns TIAC field
    { std::vector<float> o(NX*NY); for(int k=0;k<NX*NY;++k)o[k]=tiac[(size_t)1*NY*NX+k];
      FILE* ff=fopen("tiac_upwind_z1.bin","wb"); int h[2]={NX,NY};
      fwrite(h,4,2,ff);fwrite(o.data(),4,NX*NY,ff);fclose(ff); }
    printf("[city] wrote tiac_vanleer_z1.bin (production) and tiac_upwind_z1.bin (objective).\n");
    printf("[city] plot: python3 plot_diffusion.py city %d %d\n",NX,NY);
    return 0;
}
#endif

int main(int argc,char**argv){
    std::string mode=(argc>1)?argv[1]:"bench";
    if(mode=="bench") return run_bench();
#ifdef WITH_LBM
    if(mode=="city")  return run_city(argc,argv);
#endif
    fprintf(stderr,"usage: %s bench            (self-contained, fast)\n",argv[0]);
    fprintf(stderr,"       %s city <m> <s> <w> (production van Leer vs upwind; needs WITH_LBM build)\n",argv[0]);
    return 1;
}
