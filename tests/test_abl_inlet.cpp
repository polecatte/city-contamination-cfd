#include "abl_inlet.h"
#include <cstdio>
using namespace abl;
int main(){
    ABLInlet in; in.z0=0.7; in.L_turb=40.0; in.n_modes=120; in.wind_angle=0.0;
    in.init(/*U_ref*/5.0, /*z_ref*/40.0);
    fprintf(stderr,"u_star=%.4f m/s  gains=(%.3f,%.3f,%.3f)\n", in.u_star, in.gx,in.gy,in.gz);

    // (1) mean profile + realized sigmas vs height (sample over y,t)
    FILE*fp=fopen("abl_profile.csv","w"); fprintf(fp,"z,Umean,sig_u,sig_v,sig_w\n");
    for(double z=2; z<=120; z+=2){
        double su=0,sv=0,sw=0; int M=3000;
        for(int m=0;m<M;++m){
            double y=(m*37.0); double t=(m*0.05);
            double ux,uy,uz; in.velocity(0,y,z,t,ux,uy,uz);
            double U=in.mean_speed(z);
            double up=ux-U*std::cos(in.wind_angle), vp=uy, wp=uz;
            su+=up*up; sv+=vp*vp; sw+=wp*wp;
        }
        fprintf(fp,"%.1f,%.4f,%.4f,%.4f,%.4f\n",z,in.mean_speed(z),
                std::sqrt(su/M),std::sqrt(sv/M),std::sqrt(sw/M));
    }
    fclose(fp);

    // (2) time series at z_ref for autocorrelation + spectrum
    FILE*ft=fopen("abl_tseries.csv","w"); fprintf(ft,"t,up\n");
    double dt=0.02; for(int i=0;i<8192;++i){ double t=i*dt;
        double ux,uy,uz; in.velocity(0,250,40,t,ux,uy,uz);
        fprintf(ft,"%.4f,%.5f\n", t, ux-in.mean_speed(40)); }
    fclose(ft);

    // (3) divergence check: |div u'| * dx / sigma over random stencils
    double dx=1.0, acc=0, accu=0; int M=2000;
    for(int m=0;m<M;++m){
        double X=(m*13.7), Y=(m*7.3), Z=10+std::fmod(m*3.1,100.0), T=(m*0.11);
        double uxp,uyp,uzp,a,b,c;
        auto fl=[&](double x,double y,double z,double t,double&fx,double&fy,double&fz){
            in.fluct(x,y,z,t,fx,fy,fz);
            fx*=in.sigma_u_ratio*in.u_star; fy*=in.sigma_v_ratio*in.u_star; fz*=in.sigma_w_ratio*in.u_star; };
        double fxp,fyp,fzp,fxm,fym,fzm,t1,t2,t3;
        fl(X+dx,Y,Z,T,fxp,t1,t2); fl(X-dx,Y,Z,T,fxm,t1,t2);
        fl(X,Y+dx,Z,T,t1,fyp,t2);  fl(X,Y-dx,Z,T,t1,fym,t2);
        fl(X,Y,Z+dx,T,t1,t2,fzp);  fl(X,Y,Z-dx,T,t1,t2,fzm);
        double div=(fxp-fxm)/(2*dx)+(fyp-fym)/(2*dx)+(fzp-fzm)/(2*dx);
        in.fluct(X,Y,Z,T,a,b,c);
        double up=in.sigma_u_ratio*in.u_star*a, vp=in.sigma_v_ratio*in.u_star*b, wp=in.sigma_w_ratio*in.u_star*c;
        double mag=std::sqrt(up*up+vp*vp+wp*wp)+1e-9;
        acc+=std::fabs(div); accu+=mag/dx;
    }
    fprintf(stderr,"mean|div u'|/(sigma/dx) = %.4f  (0=perfectly solenoidal)\n", acc/accu);
    return 0;
}
