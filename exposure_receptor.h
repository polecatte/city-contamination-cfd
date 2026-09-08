#pragma once
// exposure_receptor.h — building-anchored microenvironmental exposure (EXPOSURE_METRIC.md).
// Population is anchored to buildings (connected components of SOLID cells), NOT to
// cells. Each building b: occupancy N_b, envelope cells (indoor intake, weight
// f_in·F_inf) and adjacent pedestrian-level street cells (outdoor, weight f_out).
// From the outdoor dosage field Θ=∫C dt (one forward run), reduces to the three
// standard quantities: E_indoor, E_outdoor, E_total = E_in + E_out.
#include <vector>
#include <cstdint>
#include <cmath>
#include <cstdio>

namespace expo {
// cell types (shared with must_geom / adj): FLUID=0, GROUND=1, SOLID/SHELL=2
enum { C_FLUID=0, C_GROUND=1, C_SOLID=2 };

struct Geom { int nx,ny,nz; const uint8_t* tp; int ax0,ay0; double dx; };
static inline int GX(int x,int y,int z,int nx,int ny){ return (z*ny+y)*nx+x; }

// 6-connected labelling of SOLID cells -> building id (>=0); -1 elsewhere.
inline std::vector<int> label_buildings(const Geom& g){
    int nx=g.nx,ny=g.ny,nz=g.nz,N=nx*ny*nz;
    std::vector<int> lab(N,-1); std::vector<int> stk; int next=0;
    const int dof[6][3]={{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1}};
    for(int s=0;s<N;++s){
        if(g.tp[s]!=C_SOLID || lab[s]>=0) continue;
        int id=next++; lab[s]=id; stk.clear(); stk.push_back(s);
        while(!stk.empty()){ int c=stk.back(); stk.pop_back();
            int z=c/(nx*ny), y=(c/nx)%ny, x=c%nx;
            for(auto&d:dof){ int xx=x+d[0],yy=y+d[1],zz=z+d[2];
                if(xx<0||xx>=nx||yy<0||yy>=ny||zz<0||zz>=nz) continue;
                int n=GX(xx,yy,zz,nx,ny);
                if(g.tp[n]==C_SOLID && lab[n]<0){ lab[n]=id; stk.push_back(n); } } }
    }
    return lab; // next = #buildings
}

struct Weights { std::vector<float> w_in, w_out; int n_buildings; long n_env, n_street; };

// Build receptor weights. occ[b] = occupancy of building b (from city_builder7 eff_inh,
// or a floor-area proxy). F_inf may be per-building (finf.size()==nB) or scalar via finf0.
inline Weights build_receptors(const Geom& g, const std::vector<int>& lab, int nB,
                               const std::vector<double>& occ,
                               double f_in, double f_out, double finf0,
                               const std::vector<double>* finf /*=nullptr*/, double z_ped_m){
    int nx=g.nx,ny=g.ny,nz=g.nz,N=nx*ny*nz;
    int zped=std::max(1,(int)std::llround(z_ped_m/g.dx));
    auto solidAt=[&](int x,int y,int z){ return (x<0||x>=nx||y<0||y>=ny||z<0||z>=nz)?-2
                        : (g.tp[GX(x,y,z,nx,ny)]==C_SOLID? lab[GX(x,y,z,nx,ny)] : -1); };
    // pass 1: count env/street cells per building (for the 1/|env|, 1/|street| means)
    std::vector<long> nenv(nB,0), nstr(nB,0);
    const int dof[6][3]={{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1}};
    auto adjBuilding=[&](int x,int y,int z)->int{               // building id this outdoor cell touches
        for(auto&d:dof){ int b=solidAt(x+d[0],y+d[1],z+d[2]); if(b>=0) return b; } return -1; };
    for(int z=1;z<nz-1;++z)for(int y=1;y<ny-1;++y)for(int x=1;x<nx-1;++x){
        if(g.tp[GX(x,y,z,nx,ny)]!=C_FLUID) continue;
        int b=adjBuilding(x,y,z); if(b<0) continue;
        ++nenv[b]; if(z==zped) ++nstr[b];
    }
    Weights W; W.w_in.assign(N,0.f); W.w_out.assign(N,0.f); W.n_buildings=nB; W.n_env=W.n_street=0;
    for(int z=1;z<nz-1;++z)for(int y=1;y<ny-1;++y)for(int x=1;x<nx-1;++x){
        int id=GX(x,y,z,nx,ny); if(g.tp[id]!=C_FLUID) continue;
        int b=adjBuilding(x,y,z); if(b<0) continue;
        double Fb = (finf&&(int)finf->size()==nB)? (*finf)[b] : finf0;
        if(nenv[b]>0){ W.w_in[id] += (float)(occ[b]*f_in*Fb/nenv[b]); ++W.n_env; }         // indoor intake
        if(z==zped && nstr[b]>0){ W.w_out[id] += (float)(occ[b]*f_out/nstr[b]); ++W.n_street; } // outdoor
    }
    return W;
}

struct Exposure { double indoor, outdoor, total; };
// n_source = number of active source cells whose bursts were summed (by linearity)
// into Theta. Dividing by it turns J into the mean exposure PER SOURCE LOCATION, so
// that considering more or fewer candidate release cells does not shift the metric —
// it stays an intensive per-source quantity. Pass 1 for a single-cell burst (default).
// This matches exposure_solve.cpp's  J/|Omega|  and reverse_objective.h's  s=Q/tot.
inline Exposure reduce(const Weights& W, const std::vector<float>& Theta, long n_source=1){
    double ei=0,eo=0; for(size_t i=0;i<Theta.size();++i){ ei+=(double)W.w_in[i]*Theta[i]; eo+=(double)W.w_out[i]*Theta[i]; }
    double inv = (n_source>0) ? 1.0/(double)n_source : 1.0;
    ei*=inv; eo*=inv;
    return { ei, eo, ei+eo };
}
// floor-area proxy occupancy: N_b ∝ building footprint area × density (until city_builder7 supplies eff_inh)
inline std::vector<double> occupancy_proxy(const Geom& g, const std::vector<int>& lab, int nB, double per_m2){
    std::vector<double> foot(nB,0.0); int nx=g.nx,ny=g.ny;
    for(int y=0;y<ny;++y)for(int x=0;x<nx;++x){ int b=lab[GX(x,y,1,nx,ny)]; if(b>=0) foot[b]+=1.0; } // z=1 footprint
    for(auto&f:foot) f *= g.dx*g.dx*per_m2;   // cells -> m² -> people
    return foot;
}
} // namespace expo
