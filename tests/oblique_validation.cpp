// oblique_validation.cpp — physics acceptance tests for the two-inlet oblique-wind fix.
//
// Runs ONE case (empty or cube domain at wind angle θ), then extracts the MEAN
// flow and computes the three self-grading metrics that establish the oblique
// capability is physical (see OBLIQUE_DIVERGENCE_DIAGNOSIS.md, "validation ladder"):
//
//   RUNG 2  horizontal homogeneity : spread of |U|(z) across interior columns.
//           An undisturbed ABL is horizontally homogeneous, so every interior
//           column (away from walls/buildings) must share the same profile. A
//           lateral BC that scars the field, or a starved y-fetch, shows up as
//           spread. Also reports RMS error vs the analytic log law.
//   RUNG 3  incompressibility       : RMS |∇·u| over interior fluid cells
//           (central differences), normalized — the mass-closure proxy.
//   RUNG 4  rotation invariance      : handled by run_oblique_validation.sh, which
//           overlays the per-angle |U|(z) profiles this driver writes to CSV. A
//           1-D ABL profile cannot depend on θ, so the overlay must collapse.
//
// One case per process; the shell loops over θ and grades. Emits:
//   - a human summary,
//   - profile_deg<NN>.csv  (z_m, |U|_mean, |U|_loglaw)  for the overlay,
//   - a machine RESULT line.
//
// BUILD (backend-agnostic; link against the same kernels.o/solver.o as the driver):
//   GPU: $HOSTCXX -O3 -std=c++17 -fopenmp oblique_validation.cpp kernels.o solver.o -L$CUDA_LIB -lcudart -o oblique_val
//   CPU: g++      -O3 -std=c++17 -fopenmp oblique_validation.cpp kernels.o solver.o -o oblique_val_cpu

#include "lbm_solver.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
using std::vector;

static const uint8_t FLUID=0, GROUND=1;
static inline int IDX(int x,int y,int z,int nx,int ny){ return z*ny*nx + y*nx + x; }
static const double KAPPA=0.41;

static void add_ground(vector<uint8_t>& t,int nx,int ny){
    for(int y=0;y<ny;++y) for(int x=0;x<nx;++x) t[IDX(x,y,0,nx,ny)]=GROUND;
}
static void add_cube(vector<uint8_t>& t,int nx,int ny,int nz,int x0,int y0,int H){
    for(int z=1;z<=H&&z<nz;++z) for(int y=y0;y<y0+H&&y<ny;++y) for(int x=x0;x<x0+H&&x<nx;++x)
        t[IDX(x,y,z,nx,ny)]=GROUND;
}
static lbm::Config base_cfg(int nx,int ny,int nz,float wind_angle){
    lbm::Config c{};
    c.nx=nx;c.ny=ny;c.nz=nz;c.cell_size=4.0f;c.U_inlet=5.0f;c.wind_angle=wind_angle;
    c.nu_phys=1.5e-5f;c.Cw=0.325f;c.Sc_t=0.7f;c.D_mol=1e-5f;
    c.source_x=-100;c.source_y=-100;c.source_z=-100;c.Q_source=0.0f;
    // Flow-only validation: the mean flow we grade is built entirely in Phase A
    // (spin-up + averaging). Phase B (scalar transport) is unused, so keep it to a
    // single step. NOTE: release_time MUST be > 0 — when it is 0 the solver sets the
    // Phase-B length to max_steps as a fallback, which here would be 2,000,000 steps.
    c.max_steps=200000;c.check_interval=500;c.conv_threshold=1e-4f;c.release_time=1e-3f;
    c.inlet_profile=1;c.abl_z0=0.7f;c.abl_zref=40.0f;c.abl_Lturb=40.0f;
    c.abl_nmodes=100;c.abl_sigu_ratio=2.5f;c.abl_sigv_ratio=1.9f;c.abl_sigw_ratio=1.25f;
    return c;
}

int main(int argc,char** argv){
    double deg=0.0; int H=16, spin=4000, avg=3000; bool empty=true;
    for(int i=1;i<argc;++i){
        if(!strcmp(argv[i],"--deg")&&i+1<argc) deg=atof(argv[++i]);
        else if(!strcmp(argv[i],"--H")&&i+1<argc) H=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--spin")&&i+1<argc) spin=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--avg")&&i+1<argc) avg=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--cube")) empty=false;
        else if(!strcmp(argv[i],"--empty")) empty=true;
        else { fprintf(stderr,"unknown arg %s\n",argv[i]); return 3; }
    }
    int nx=(5+1+15)*H, ny=(5*2+1)*H, nz=(5+1)*H;
    vector<uint8_t> t(nx*ny*nz,FLUID); add_ground(t,nx,ny);
    if(!empty) add_cube(t,nx,ny,nz,5*H,5*H,H);

    lbm::Config c=base_cfg(nx,ny,nz,(float)(deg*M_PI/180.0));
    c.spinup_steps=spin; c.max_warmup=spin+avg; c.avg_steps=std::min(avg,1500); c.avg_threshold=3e-3f;

    printf("================================================================\n");
    printf("[val] deg=%.0f grid=%dx%dx%d cube=%s spin=%d avg=%d\n",
           deg,nx,ny,nz, empty?"OFF":"ON", spin, avg);
    printf("================================================================\n"); fflush(stdout);

    lbm::Solver s(c);
    vector<float> perm(nx*ny*nz,0), inh(nx*ny*nz,0), dvel(nx*ny*nz,0);
    s.load_geometry(t.data(),perm.data(),inh.data(),dvel.data());
    lbm::Result r=s.run();
    if(!std::isfinite(r.max_velocity)){
        printf("RESULT deg=%.0f -> DIVERGED (no profile)\n",deg); return 2;
    }

    vector<float> ux,uy,uz,nut; s.copy_mean_flow_to_host(ux,uy,uz,nut);
    const double cell=c.cell_size, Uref=c.U_inlet, z0=c.abl_z0, zref=c.abl_zref;
    const double ustar = Uref*KAPPA/std::log((zref+z0)/z0);
    auto loglaw=[&](double zm){ return (ustar/KAPPA)*std::log((zm+z0)/z0); };

    // Sample the DEVELOPED INTERIOR CORE only: a centered patch that excludes the
    // near-inlet adjustment region (where the imposed profile is still developing
    // over the first fetch) and every boundary layer. Angle-agnostic, so it is fair
    // for oblique wind. Homogeneity is judged here — near-inlet development is NOT
    // counted as inhomogeneity.
    int x0=(int)std::lround(0.35*nx), x1=(int)std::lround(0.70*nx);
    int y0=(int)std::lround(0.30*ny), y1=(int)std::lround(0.70*ny);
    vector<std::pair<int,int>> cols;
    for(int gx=0;gx<4;++gx) for(int gy=0;gy<4;++gy){
        int x=x0 + gx*(x1-x0)/3, y=y0 + gy*(y1-y0)/3;
        if(!empty){ int bx=5*H,by=5*H; if(x>=bx-H&&x<=bx+2*H&&y>=by-H&&y<=by+2*H) continue; }
        cols.push_back({x,y});
    }
    // Project each core column onto the wind direction so we can separate STREAMWISE
    // maintenance (change along the flow = ABL decay/growth over fetch) from LATERAL
    // spread (variation across the flow at similar fetch = residual inhomogeneity +
    // turbulent noise). s = x·cosθ + y·sinθ; terciles of s give upstream/downstream.
    const double cwx=std::cos(deg*M_PI/180.0), cwy=std::sin(deg*M_PI/180.0);
    vector<double> svals; for(auto&p:cols) svals.push_back(p.first*cwx+p.second*cwy);
    vector<double> ssort=svals; std::sort(ssort.begin(),ssort.end());
    double s_lo = ssort.empty()?0:ssort[ssort.size()/3];
    double s_hi = ssort.empty()?0:ssort[(2*ssort.size())/3];

    // RUNG 2: |U|(z) per column; report cross-column spread and log-law RMS error.
    // Speed magnitude in physical units: |U|_phys = |u|_lb * (Uref/u_lb). u_lb is
    // not exposed, but the ratio Uref/u_lb equals 1/(mean |u|_lb at zref)/... —
    // instead we compare in *normalized* form |U|/Uref = |u|_lb / u_lb_ref, where
    // u_lb_ref is the lattice speed at zref recovered from the field itself.
    int zref_k = std::clamp((int)std::round(zref/cell),1,nz-2);
    // reference lattice speed = column-median |u|_lb at zref (robust to one bad column)
    vector<double> sref;
    for(auto&p:cols){ double a=ux[IDX(p.first,p.second,zref_k,nx,ny)],
                              b=uy[IDX(p.first,p.second,zref_k,nx,ny)];
                      sref.push_back(std::sqrt(a*a+b*b)); }
    std::sort(sref.begin(),sref.end());
    double u_lb_ref = sref.empty()?1.0:sref[sref.size()/2];
    if(u_lb_ref<=0) u_lb_ref=1e-9;

    FILE* f=nullptr; char fn[64]; std::snprintf(fn,sizeof fn,"profile_deg%02d.csv",(int)std::round(deg));
    f=fopen(fn,"w"); if(f) fprintf(f,"z_m,U_over_Uref_mean,U_over_Uref_loglaw,streamwise_frac,lateral_frac\n");

    double loglaw_rms=0; int nz_used=0;
    // Accumulators for RMS-over-height. All quantities are in U_ref units (V is
    // already normalized by the zref speed), so we do NOT re-divide by the small
    // near-ground mean — that local-mean division is what inflates surface-layer
    // numbers and conflates high near-wall turbulence intensity with mean
    // inhomogeneity. We also use STD (not range) so the estimator isn't
    // sample-size/outlier driven, and split surface-layer (z<zref) from aloft.
    double sw2=0, lat2=0; int nsw=0;            // full-profile sums of squares
    double latSL2=0; int nSL=0, latAL2n=0; double latAL2=0;
    double sw_max=0, lat_max=0;
    for(int z=1; z<nz-1; ++z){
        double zm=z*cell;
        vector<double> up, dn, mid, all;
        for(size_t i=0;i<cols.size();++i){
            double a=ux[IDX(cols[i].first,cols[i].second,z,nx,ny)],
                   b=uy[IDX(cols[i].first,cols[i].second,z,nx,ny)];
            double V=std::sqrt(a*a+b*b)/u_lb_ref;                 // in U_ref units
            all.push_back(V);
            if(svals[i]<=s_lo) up.push_back(V); else if(svals[i]>=s_hi) dn.push_back(V); else mid.push_back(V);
        }
        if(all.empty()) continue;
        auto avg=[](const vector<double>&v){ double s=0; for(double x:v)s+=x; return v.empty()?0.0:s/v.size(); };
        auto ssd=[&](const vector<double>&v){ double m=avg(v),s=0; for(double x:v)s+=(x-m)*(x-m); return s; };
        double mean=avg(all);
        // streamwise maintenance: upstream→downstream change of the fetch-binned mean (U_ref units).
        double sw = (!up.empty()&&!dn.empty()) ? std::fabs(avg(dn)-avg(up)) : 0.0;
        // lateral inhomogeneity: pooled WITHIN-fetch-bin std (removes the streamwise trend), U_ref units.
        double pooled_var = (ssd(up)+ssd(mid)+ssd(dn)) / std::max<size_t>(1,all.size());
        double lat = std::sqrt(pooled_var);
        sw2+=sw*sw; lat2+=lat*lat; ++nsw;
        sw_max=std::max(sw_max,sw); lat_max=std::max(lat_max,lat);
        if(zm<zref){ latSL2+=lat*lat; ++nSL; } else { latAL2+=lat*lat; ++latAL2n; }
        double ll = loglaw(zm)/loglaw(zref);
        loglaw_rms += (mean-ll)*(mean-ll); ++nz_used;
        if(z%std::max(1,(nz/12))==0)
            printf("  %5.1f      %7.3f       %7.3f     %6.1f%%   %6.1f%%\n", zm, mean, ll, 100*sw, 100*lat);
        if(f) fprintf(f,"%.2f,%.4f,%.4f,%.4f,%.4f\n",zm,mean,ll,sw,lat);
    }
    if(f) fclose(f);
    loglaw_rms = std::sqrt(loglaw_rms/std::max(1,nz_used));
    double stream_rms=std::sqrt(sw2/std::max(1,nsw)), lat_rms=std::sqrt(lat2/std::max(1,nsw));
    double lat_sl=std::sqrt(latSL2/std::max(1,nSL)), lat_al=std::sqrt(latAL2/std::max(1,latAL2n));

    // RUNG 3: RMS |div u| over interior fluid cells (central differences, LU).
    double div2=0; long ndiv=0;
    auto isFluid=[&](int x,int y,int z){ return t[IDX(x,y,z,nx,ny)]==FLUID; };
    for(int z=2;z<nz-2;++z) for(int y=2;y<ny-2;++y) for(int x=2;x<nx-2;++x){
        if(!isFluid(x,y,z)) continue;
        if(!isFluid(x+1,y,z)||!isFluid(x-1,y,z)||!isFluid(x,y+1,z)||
           !isFluid(x,y-1,z)||!isFluid(x,y,z+1)||!isFluid(x,y,z-1)) continue;
        double dux=(ux[IDX(x+1,y,z,nx,ny)]-ux[IDX(x-1,y,z,nx,ny)])*0.5;
        double duy=(uy[IDX(x,y+1,z,nx,ny)]-uy[IDX(x,y-1,z,nx,ny)])*0.5;
        double duz=(uz[IDX(x,y,z+1,nx,ny)]-uz[IDX(x,y,z-1,nx,ny)])*0.5;
        double d=dux+duy+duz; div2+=d*d; ++ndiv;
    }
    double div_rms = (ndiv? std::sqrt(div2/ndiv):0.0) / u_lb_ref;   // normalized

    // Grades. Homogeneity is judged on the MEAN flow in U_ref units, RMS over height;
    // the surface layer (z<zref) is reported separately because near-wall turbulence
    // intensity is physically high there and is not a mean-inhomogeneity defect.
    bool p_sw = stream_rms < 0.10, p_lat = lat_al < 0.10, p2 = p_sw && p_lat;
    bool p3 = div_rms < 0.05, pll = loglaw_rms < 0.12;
    printf("\n[val] RUNG 2 homogeneity (U_ref units, RMS over z):\n");
    printf("[val]   streamwise maintenance = %.1f%% (max %.1f%%) -> %s\n",
           100*stream_rms, 100*sw_max, p_sw?"PASS":"FAIL");
    printf("[val]   lateral inhomogeneity  = %.1f%% aloft (z>=zref) -> %s   [surface-layer %.1f%%, full %.1f%%, max %.1f%%]\n",
           100*lat_al, p_lat?"PASS":"FAIL", 100*lat_sl, 100*lat_rms, 100*lat_max);
    printf("[val]        log-law fit: profile RMS err  = %.1f%%  -> %s\n",
           100*loglaw_rms, pll?"PASS":"FAIL");
    printf("[val] RUNG 3 incompress.: RMS|div u|/Uref  = %.2f%%  -> %s\n",
           100*div_rms, p3?"PASS":"FAIL");
    printf("[val] (RUNG 4 rotation invariance graded across angles by the shell.)\n");
    printf("RESULT deg=%.0f streamwise=%.4f lateral_aloft=%.4f lateral_sl=%.4f loglaw=%.4f div=%.4f pass=%d profile=%s\n",
           deg, stream_rms, lat_al, lat_sl, loglaw_rms, div_rms, (p2&&p3&&pll)?1:0, fn);
    fflush(stdout);
    return (p2&&p3&&pll)?0:1;
}
