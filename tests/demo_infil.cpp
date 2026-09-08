#include "infiltration.h"
#include <cstdio>
int main(){
    printf("dp_um,P,Finf_a0.2,Finf_a0.5,Finf_a1.0\n");
    for(double dp=0.01; dp<=12; dp*=1.3){
        double m=dp*1e-6;
        printf("%.3f,%.3f,%.3f,%.3f,%.3f\n", dp, infil::penetration_factor(m),
            infil::infiltration_factor(m,0.2), infil::infiltration_factor(m,0.5),
            infil::infiltration_factor(m,1.0));
    }
    // PM10 (the project's particle) at typical urban AER
    double f10=infil::infiltration_factor(10e-6,0.5);
    double f25=infil::infiltration_factor(2.5e-6,0.5);
    fprintf(stderr,"PM10 F_inf(a=0.5/h)=%.2f ; PM2.5 F_inf=%.2f (lit. PM2.5 range 0.3-0.82)\n",f10,f25);
    return 0;
}
