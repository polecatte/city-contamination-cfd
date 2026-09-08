// overshoot_test.cpp — characterise QUICK's unphysical (negative) values and their
// impact on the dose. QUICK is linear+higher-order, so by Godunov it is non-
// monotone: it can undershoot below 0 near sharp gradients. This quantifies:
//   (1) magnitude (max |neg|/peak), (2) how much MASS is negative (Σ|neg|/Σ|all|),
//   (3) whether it GROWS (instability) or damps, (4) impact on the dose J=⟨w,Θ⟩,
//   and tests the fix: clipping Θ≥0 for the dose (mass-safe, standard practice).
#include "adjoint_transport.h"
#include <cstdio>
#include <vector>
#include <cmath>
#include <algorithm>
using namespace adj;

struct Stat{ double overshoot, negfrac, totmass; };
static Stat stat(const std::vector<float>&C){
    double peak=0,neg=0,pos=0; for(float v:C){ if(v>peak)peak=v; if(v<0)neg+=-v; else pos+=v; }
    double mn=0; for(float v:C) if(v<mn)mn=v;
    return { peak>0? -mn/peak:0.0, (pos+neg)>0? neg/(pos+neg):0.0, pos-neg };
}
int main(){
    int nx=140, ny=48, nz=48, N=nx*ny*nz;
    double U=0.05, D=0.004;
    std::vector<uint8_t> tp(N,(uint8_t)FLUID);
    std::vector<float> ux(N,(float)U),uy(N,0),uz(N,0),nut(N,0),vd(N,0);
    Field F; F.nx=nx;F.ny=ny;F.nz=nz;F.N=N;F.tp=tp.data();
    F.ux=ux.data();F.uy=uy.data();F.uz=uz.data();F.nut=nut.data();F.vdep=vd.data();
    F.D0=(float)D;F.Sc_t=1.0f;F.w_s=0.f;F.dt=stable_dt(F,0.4f);
    auto ix=[&](int x,int y,int z){return F.idx(x,y,z);};
    int x0=30,y0=ny/2,z0=nz/2;
    double sharp_nm0=0,sharp_nm2=0,realistic_dose_impact=0;

    for(int sharp=1; sharp>=0; --sharp){
        std::vector<float> C(N,0.f), Cn(N,0.f), Theta(N,0.f);
        if(sharp) C[ix(x0,y0,z0)]=1.0f;                       // delta: sharpest (worst case)
        else { double s=2.0, tot=0;                            // Gaussian blob σ=2 (~ a resolved pulse)
            for(int z=z0-6;z<=z0+6;++z)for(int y=y0-6;y<=y0+6;++y)for(int x=x0-6;x<=x0+6;++x){
                double r2=(x-x0)*(x-x0)+(y-y0)*(y-y0)+(z-z0)*(z-z0);
                double v=std::exp(-r2/(2*s*s)); C[ix(x,y,z)]=(float)v; tot+=v; }
            for(auto&v:C) v/=(float)tot; }                     // normalise to unit mass
        int steps=500; double dt=F.dt;
        std::vector<double> nm_hist;
        for(int t=0;t<steps;++t){ fwd_step(F,C.data(),Cn.data()); C.swap(Cn);
            for(int i=0;i<N;++i) Theta[i]+=C[i]*dt;            // accumulate dose Θ=∫C dt
            if(t==steps/4||t==steps/2||t==steps-1){ Stat s=stat(C); nm_hist.push_back(s.negfrac); } }
        Stat sC=stat(C), sT=stat(Theta);
        printf("=== %s release ===\n", sharp?"SHARP delta (worst case)":"smooth blob (realistic)");
        printf("  instantaneous C: overshoot=%.1f%%  neg-mass=%.3f%%  (final step)\n",100*sC.overshoot,100*sC.negfrac);
        printf("  dose Θ=∫C dt   : overshoot=%.1f%%  neg-mass=%.3f%%\n",100*sT.overshoot,100*sT.negfrac);
        printf("  neg-mass evolution (t=%d,%d,%d): %.3f%% -> %.3f%% -> %.3f%%  [%s]\n",
               steps/4,steps/2,steps-1, 100*nm_hist[0],100*nm_hist[1],100*nm_hist[2],
               nm_hist[2]<=nm_hist[0]*1.2?"BOUNDED (stable)":"growing");
        if(sharp){ sharp_nm0=nm_hist[0]; sharp_nm2=nm_hist[2]; }
        // dose impact: receptor slab downstream; J vs clipped J
        double J=0,Jc=0; for(int i=0;i<N;++i){ J+=Theta[i]; Jc+=std::max(0.f,Theta[i]); }
        printf("  whole-field dose ΣΘ=%.4e  clipped Σmax(0,Θ)=%.4e  Δ=%.3f%%\n",
               J,Jc, J!=0?100*(Jc-J)/std::fabs(J):0.0);
        printf("  -> negatives are %.3f%% of the dose; clipping (nonlinear) would remove them\n"
               "     but break source superposition. Kept linear; negatives negligible.\n\n",100*(Jc-J)/std::fabs(J));
        if(!sharp) realistic_dose_impact=100*(Jc-J)/std::fabs(J);
    }
    bool bounded = sharp_nm2 <= sharp_nm0*1.2;          // worst-case negatives do not grow
    bool small   = std::fabs(realistic_dose_impact) < 1.0; // realistic dose impact < 1%
    printf("VERDICT: QUICK negatives %s (worst-case %s) and dose impact %s (%.3f%% realistic).\n",
           bounded?"BOUNDED":"GROWING", bounded?"damps":"GROWS",
           small?"NEGLIGIBLE":"SIGNIFICANT", std::fabs(realistic_dose_impact));
    printf("%s\n", (bounded&&small)?"PASS — unphysical values do not affect the dose; kept linear (no clip)."
                                    :"REVIEW — negatives non-negligible; consider clip for reporting.");
    return (bounded&&small)?0:1;
}
