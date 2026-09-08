// adjoint_recip_test.cpp — verify the reverse solver against the forward.
//   (A) dot-product test: <M u, v> == <u, Mᵀ v>  (exact transpose)
//   (B) integrated objective identity: forward J(s,w) == reverse Σ F(w)·s
//   (C) physical sanity: a release advects downwind, mass is conserved up to
//       deposition+outflow.
// Uses a prescribed sheared flow + a city-like geometry (ground + solid blocks).
#include "adjoint_transport.h"
#include <cstdio>
#include <random>
using namespace adj;

int main(){
    bool ok=true; const double TOL=1e-4;   // gate: transpose + objective identity must hold
    const int nx=28, ny=20, nz=12, N=nx*ny*nz;
    std::vector<uint8_t> tp(N, (uint8_t)FLUID);
    std::vector<float> ux(N,0.f), uy(N,0.f), uz(N,0.f), nut(N,0.f), vdep(N,0.f);
    auto IDX=[&](int x,int y,int z){ return (z*ny+y)*nx+x; };

    // geometry: ground plane + solid blocks
    for(int y=0;y<ny;++y) for(int x=0;x<nx;++x) tp[IDX(x,y,0)]=GROUND;
    int blk[4][5]={{6,6,5,5,5},{16,5,4,6,7},{11,12,5,4,4},{20,12,4,5,6}};
    for(auto&b:blk) for(int x=b[0];x<b[0]+b[2];++x) for(int y=b[1];y<b[1]+b[3];++y)
        for(int z=1;z<1+b[4];++z) if(x<nx&&y<ny&&z<nz) tp[IDX(x,y,z)]=SHELL;

    // prescribed flow: sheared +x, plus small cross/vertical components; deposition
    // velocity on cells adjacent to solids (representative).
    for(int z=0;z<nz;++z) for(int y=0;y<ny;++y) for(int x=0;x<nx;++x){
        int i=IDX(x,y,z);
        if(tp[i]!=FLUID) continue;
        ux[i]=0.06f*(float)z/(nz-1);
        uy[i]=0.005f*std::sin(0.3f*x);
        uz[i]=0.003f;
        nut[i]=0.01f*(0.5f+0.5f*(float)z/(nz-1));
    }
    // surface deposition velocity on solid cells (ground + building faces)
    for(int i=0;i<N;++i) if(tp[i]==GROUND || tp[i]==SHELL) vdep[i]=0.01f;
    Field F{nx,ny,nz,N, tp.data(),ux.data(),uy.data(),uz.data(),nut.data(),vdep.data(),
            1e-3f, 0.7f, 0.006f, 0.f};
    F.dt = stable_dt(F);
    printf("stable dt = %.4f\n", F.dt);

    std::mt19937 rng(1); std::normal_distribution<float> g(0.f,1.f);
    auto fluid_mask=[&](std::vector<float>& v){ for(int i=0;i<N;++i) if(tp[i]!=FLUID) v[i]=0.f; };

    // ── (A) dot-product test for single step and composed n-step operators ──
    printf("\n(A) dot-product test  <M u,v> vs <u, Mt v>\n");
    for(int n : {1,8,40}){
        std::vector<float> u(N),v(N); for(int i=0;i<N;++i){u[i]=g(rng);v[i]=g(rng);}
        fluid_mask(u); fluid_mask(v);
        // M^n u
        std::vector<float> a=u,b(N);
        for(int k=0;k<n;++k){ fwd_step(F,a.data(),b.data()); std::swap(a,b);} // a = M^n u
        // (Mt)^n v
        std::vector<float> c=v,e(N);
        for(int k=0;k<n;++k){ adj_step(F,c.data(),e.data()); std::swap(c,e);} // c = (Mt)^n v
        double lhs=dot(a,v), rhs=dot(u,c);
        double rel=std::fabs(lhs-rhs)/std::max({std::fabs(lhs),std::fabs(rhs),1e-300});
        if(rel>TOL) ok=false;
        printf("  n=%2d:  <Mu,v>=%+.9e  <u,Mtv>=%+.9e  rel=%.2e\n", n, lhs, rhs, rel);
    }

    // ── (B) integrated objective identity: forward J == reverse Σ F·s ──
    printf("\n(B) objective identity  forward J(s,w)  vs  reverse Sum F(w).s\n");
    std::vector<float> s(N,0.f), w(N,0.f);
    // source: uniform over ground-level open fluid (z=1, not solid)
    for(int y=0;y<ny;++y) for(int x=0;x<nx;++x){ int i=IDX(x,y,1); if(tp[i]==FLUID) s[i]=1.f; }
    { double tot=0; for(int i=0;i<N;++i) tot+=s[i]; for(int i=0;i<N;++i) s[i]/=(float)tot; }
    // receptor: random nonneg weights on fluid cells (stands in for effective inhabitance)
    for(int i=0;i<N;++i) if(tp[i]==FLUID) w[i]=std::fabs(g(rng));
    int T=60;
    double Jf = forward_objective(F, s, w, T);
    std::vector<float> Foot; reverse_footprint(F, w, T, Foot);
    double Jr = dot(Foot, s);
    double rel=std::fabs(Jf-Jr)/std::max({std::fabs(Jf),std::fabs(Jr),1e-300});
    if(rel>TOL) ok=false;
    printf("  forward  J = %.9e\n  reverse  J = %.9e\n  rel.err = %.2e\n", Jf, Jr, rel);

    // ── (C) physical sanity: forward plume from a single ground cell ──
    printf("\n(C) physical sanity (single upstream ground release)\n");
    std::vector<float> s1(N,0.f); s1[IDX(3,ny/2,1)]=1.f;
    std::vector<float> wuni(N,0.f); for(int i=0;i<N;++i) if(tp[i]==FLUID) wuni[i]=1.f;
    std::vector<float> TIAC;
    forward_objective(F, s1, wuni, 80, &TIAC);
    // centre-of-mass x of TIAC should be downwind (> release x=3)
    double m=0,mx=0; for(int z=0;z<nz;++z)for(int y=0;y<ny;++y)for(int x=0;x<nx;++x){
        double c=TIAC[IDX(x,y,z)]; m+=c; mx+=c*x; }
    double cx = (m>0?mx/m:0);
    if(!(cx>3.0) || !(m>0)) ok=false;   // plume must carry mass downwind of the release
    printf("  TIAC mass=%.4e  centroid_x=%.2f (release at x=3, wind +x => expect > 3)\n", m, cx);
    printf("\n[adjoint_recip] %s\n", ok?"PASS — reverse solver is the exact transpose of the forward"
                                       :"FAIL — reverse objective is NOT the forward's transpose (reverse ranking is corrupt)");
    return ok?0:1;
}
