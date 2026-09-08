// reverse_objective.h — build the receptor field w(x) and source mask s(x) for
// the effective-exposure objective, and evaluate it via the reverse solver.
//
// Objective (linear in concentration, as agreed):
//     J = Σ_x w(x)·C(x) = Σ_x F(x)·s(x)
//   w(x) — effective-inhabitance receptor weight (people present × where they
//          sample the air): building eff_inh over facade-ring fluid cells, street
//          weights on road cells, park eff_inh over park ground — all at z=1.
//   s(x) — source mask: UNIFORM over open city spaces (roads + parks) at ground
//          level (z=1 fluid cells inside the city rectangle), zero over buildings,
//          buffer fetch, and above ground; normalized so Σs = 1, scaled × Q.
//   F(x) — exposure footprint from the reverse solver, forced by w.
//
// By construction (exact discrete adjoint, see adjoint_transport.h) the forward
// value Σ w·C and the reverse value Σ F·s agree to machine precision; the driver
// returns both so the caller can assert the reciprocity regression.

#pragma once
#include "adjoint_transport.h"
#include "city_builder7.h"
#include "occupancy.h"
#include <vector>
#include <cstdint>
#include <cstdio>

namespace adj {

// Receptor field w(x): the §-5 effective-exposure weighting, as a 3-D field whose
// support is the z=1 breathing/facade level. Weights live on FLUID cells only
// (the adjoint field is defined on fluid), so eff_inh is distributed over the
// actual outdoor-air facade cells.
inline void build_receptor_field(const city::Result& r, const uint8_t* tp,
                                 int nx, int ny, int nz,
                                 const city::StreetField& sf,
                                 std::vector<float>& w) {
    int N=nx*ny*nz; w.assign(N,0.f);
    auto IDX=[&](int x,int y,int z){ return (z*ny+y)*nx+x; };
    auto in =[&](int x,int y){ return x>=0&&x<nx&&y>=0&&y<ny; };
    auto fluid_z1=[&](int x,int y){ return in(x,y) && tp[IDX(x,y,1)]==FLUID; };

    // buildings: distribute eff_inh over the one-cell facade ring (fluid cells)
    for(const auto& b : r.blocks){
        if(b.usage==city::PARK || b.eff_inh<=0) continue;
        int x0=b.bx0-1, x1=b.bx1, y0=b.by0-1, y1=b.by1;
        std::vector<int> ring;
        for(int x=x0;x<=x1;++x){ if(fluid_z1(x,y0)) ring.push_back(IDX(x,y0,1));
                                 if(fluid_z1(x,y1)) ring.push_back(IDX(x,y1,1)); }
        for(int y=y0;y<=y1;++y){ if(fluid_z1(x0,y)) ring.push_back(IDX(x0,y,1));
                                 if(fluid_z1(x1,y)) ring.push_back(IDX(x1,y,1)); }
        if(ring.empty()) continue;
        float per=(float)(b.eff_inh/ring.size());
        for(int id:ring) w[id]+=per;
    }
    // street occupants: per-cell weights already on the z=1 road network
    for(int y=0;y<ny;++y) for(int x=0;x<nx;++x){
        float sw=sf.weight[(size_t)y*nx+x];
        if(sw>0.f && fluid_z1(x,y)) w[IDX(x,y,1)]+=sw;
    }
    // parks: distribute eff_inh over the park footprint (fluid/porous ground)
    for(const auto& b : r.blocks){
        if(b.usage!=city::PARK || b.eff_inh<=0) continue;
        std::vector<int> cells;
        for(int y=b.by0;y<b.by1;++y) for(int x=b.bx0;x<b.bx1;++x)
            if(fluid_z1(x,y)) cells.push_back(IDX(x,y,1));
        if(cells.empty()) continue;
        float per=(float)(b.eff_inh/cells.size());
        for(int id:cells) w[id]+=per;
    }
}

// Source mask s(x): uniform over z=1 fluid cells inside the city rectangle
// (roads + parks; buildings are solid → auto-excluded; buffer fetch is outside
// the rectangle → excluded). Normalized to Σ=1, then scaled by Q_total.
inline void build_source_mask(const city::Result& r, const city::Params& p,
                              const uint8_t* tp, int nx, int ny, int nz,
                              double Q_total, std::vector<float>& s) {
    int N=nx*ny*nz; s.assign(N,0.f);
    auto IDX=[&](int x,int y,int z){ return (z*ny+y)*nx+x; };
    int Wc=(int)std::lround(p.city_w/city::CELL);
    int Hc=(int)std::lround(p.city_h/city::CELL);
    int x0=r.offset_x, x1=std::min(nx, r.offset_x+Wc);
    int y0=r.offset_y, y1=std::min(ny, r.offset_y+Hc);
    double tot=0.0;
    for(int y=y0;y<y1;++y) for(int x=x0;x<x1;++x){
        int id=IDX(x,y,1);
        if(tp[id]==FLUID){ s[id]=1.f; tot+=1.0; }
    }
    // Normalize by the source-cell count `tot`: s(x)=Q_total/tot spreads the SAME
    // total release Q_total over however many open cells exist, so J=Σ Foot·s =
    // Q_total·mean(Foot over source cells). This is the per-source-location mean —
    // adding/removing candidate release cells (finer grid, bigger open-space region)
    // does not change the metric by count alone. Same intent as J/|Omega| elsewhere.
    if(tot>0.0){ float k=(float)(Q_total/tot); for(int i=0;i<N;++i) s[i]*=k; }
}

// Evaluate the objective via the reverse solver and (optionally) the forward
// check. Returns J_reverse; sets *J_forward if requested.
inline double effective_exposure_reverse(const Field& Fld,
                                         const std::vector<float>& w,
                                         const std::vector<float>& s,
                                         int n_steps,
                                         double* J_forward=nullptr,
                                         std::vector<float>* footprint=nullptr) {
    std::vector<float> Foot;
    reverse_footprint(Fld, w, n_steps, Foot);
    double Jr = dot(Foot, s);
    if(J_forward) *J_forward = forward_objective(Fld, s, w, n_steps);
    if(footprint) *footprint = Foot;
    return Jr;
}

} // namespace adj
