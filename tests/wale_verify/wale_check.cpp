// stock = OpenLB 1.8.1 collisionLES.h detail::WaleEffectiveOmega::computeEffectiveOmega, tauTurb part, verbatim
#include <cstdio>
#include <cmath>
#include <random>
typedef double V;
V stock(const V vg[9], V pre){
    V g[3][3]; for(unsigned i=0;i<3;i++)for(unsigned j=0;j<3;j++) g[i][j]=vg[i*3+j];
    V s[3][3]; for(unsigned i=0;i<3;i++)for(unsigned j=0;j<3;j++) s[i][j]=(g[i][j]+g[j][i])/V{2};
    V G[3][3]; for(unsigned i=0;i<3;i++)for(unsigned j=0;j<3;j++) G[i][j]=V{0};
    for(unsigned i=0;i<3;i++)for(unsigned j=0;j<3;j++)for(unsigned k=0;k<3;k++) G[i][j]+=(g[i][k]*g[k][j]+g[j][k]*g[k][i])/V{2};
    V trace{}; for(unsigned i=0;i<3;i++) trace+=V{1}/V{3}*g[i][i]*g[i][i];
    for(unsigned i=0;i<3;i++) G[i][i]-=trace;
    V G_ip{}; for(unsigned i=0;i<3;i++)for(unsigned j=0;j<3;j++) G_ip=G[i][j]*G[i][j];
    V s_ip{}; for(unsigned i=0;i<3;i++)for(unsigned j=0;j<3;j++) s_ip=s[i][j]*s[i][j];
    V tauTurb=V{3}*pre*(std::pow(G_ip,1.5)/(std::pow(s_ip,2.5)+std::pow(G_ip,1.25)));
    if((std::pow(s_ip,2.5)+std::pow(G_ip,1.25))==0) tauTurb=0; if(tauTurb<0) tauTurb=0;
    return tauTurb;
}
int main(int argc,char**argv){
    // read gradient tensors from stdin, print stock tauTurb
    V vg[9]; V pre=0.325*0.325;
    while(true){ for(int k=0;k<9;k++) if(scanf("%lf",&vg[k])!=1) return 0; printf("%.17g\n",stock(vg,pre)); }
}
