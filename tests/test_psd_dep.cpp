#include "psd.h"
#include "deposition.h"
#include <cstdio>
using namespace city;
int main(){
    // PSD: typical environmental release, MMAD=1um, GSD=2, rho=1800
    psd::Lognormal L{1.0e-6, 2.0, 1800.0};
    auto bins=psd::discretize(L, 6);
    FILE*fb=fopen("psd_bins.csv","w"); fprintf(fb,"bin,d_ae_um,d_phys_um,frac,w_s_mm_s,vd_park_cm_s,vd_biz_cm_s\n");
    fprintf(stderr,"PSD MMAD=1um GSD=2 -> 6 size bins:\n");
    double sum=0;
    for(size_t i=0;i<bins.size();++i){
        auto&b=bins[i];
        double ws=dep::settling_velocity(b.d_phys,L.rho_p);
        double vdp=dep::deposition_velocity(PARK,b.d_phys,L.rho_p);
        double vdb=dep::deposition_velocity(BUSINESS,b.d_phys,L.rho_p);
        fprintf(fb,"%zu,%.3f,%.3f,%.4f,%.4f,%.4f,%.4f\n",i,b.d_ae*1e6,b.d_phys*1e6,b.frac,
                ws*1e3,vdp*100,vdb*100);
        fprintf(stderr,"  bin %zu: d_ae=%.2f um  frac=%.3f  w_s=%.3g mm/s  vd_park=%.3g cm/s  vd_biz=%.3g cm/s\n",
                i,b.d_ae*1e6,b.frac,ws*1e3,vdp*100,vdb*100);
        sum+=b.frac;
    }
    fclose(fb); fprintf(stderr,"  (mass fractions sum=%.3f)\n",sum);
    // v_d(dp) curves
    FILE*fc=fopen("vd_curve.csv","w"); fprintf(fc,"dp_um,vd_park,vd_res_low,vd_biz\n");
    for(double dp=0.01; dp<=30; dp*=1.25){
        double m=dp*1e-6;
        fprintf(fc,"%.4f,%.5f,%.5f,%.5f\n",dp,
            dep::deposition_velocity(PARK,m,1800)*100,
            dep::deposition_velocity(RES_LOW,m,1800)*100,
            dep::deposition_velocity(BUSINESS,m,1800)*100);
    }
    fclose(fc);
    return 0;
}
