#include "psd.h"
#include "deposition.h"
#include <cstdio>
int main(){
    psd::Lognormal L{1.0e-6, 2.0, 1800.0};   // MMAD=1um, GSD=2.0, rho_p=1800 (param_space FIXED)
    int N=6;                                   // n_size_bins
    auto bins=psd::discretize(L,N);            // default n_sig=3.0
    printf("bin  d_ae(um)  d_phys(um)  activity%%   w_settle(m/s)  v_dep_PARK   v_dep_BIZ\n");
    double fsum=0,wmin=1e9,wmax=0;
    for(size_t i=0;i<bins.size();++i){
        auto&b=bins[i];
        double w=dep::settling_velocity(b.d_phys,1800.0);
        double vp=dep::deposition_velocity(city::PARK,b.d_phys,1800.0,0.5);
        double vb=dep::deposition_velocity(city::BUSINESS,b.d_phys,1800.0,0.5);
        printf("%2zu   %7.3f   %8.3f   %7.2f   %.3e   %.3e  %.3e\n",
               i+1, b.d_ae*1e6, b.d_phys*1e6, b.frac*100, w, vp, vb);
        fsum+=b.frac; if(w<wmin)wmin=w; if(w>wmax&&w>0)wmax=w;
    }
    printf("sum activity = %.4f\n", fsum);
    printf("settling span (max/min over bins) = %.0fx\n", wmax/ (wmin>0?wmin:1e-9));
    return 0;
}
