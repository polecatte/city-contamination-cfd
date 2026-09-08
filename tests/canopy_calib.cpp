// 1-D canopy drag calibration. A D1Q3 lattice-BGK channel drives flow through a
// region of SHELL cells applying the EXACT partial-bounce-back closure
// (fd[i]=sigma*fp[OPP[i]]+(1-sigma)*fp[i], sigma=1-perm). In 1-D, continuity keeps
// rho*u constant, so the canopy manifests as a PRESSURE DROP; Dp = cs2*(rho_up -
// rho_dn) across the canopy is the integrated drag. The dimensionless loss
// coefficient Dp/(rho*u^2) is matched to the physical canopy value C_d*a*L.
#include <cstdio>
#include <cmath>
#include <vector>
#include <initializer_list>
int main(int argc,char**argv){
    // D1Q3
    const int Q=3; int C[3]={0,1,-1}, OPP[3]={0,2,1}; double W[3]={2./3,1./6,1./6};
    const double cs2=1./3;
    int nx=400; int xc0=150; int Ncan=(argc>3)?atoi(argv[3]):5; int xc1=xc0+Ncan;
    double u0 = (argc>1)? atof(argv[1]) : 0.05;  // inlet lattice velocity
    double perm=(argc>2)? atof(argv[2]) : 0.8;
    double omega=1.0;                            // BGK relaxation (nu=1/6); robust
    std::vector<double> f(nx*Q), ft(nx*Q);
    auto feq=[&](int i,double r,double u){ double cu=C[i]*u; return W[i]*r*(1+3*cu+4.5*cu*cu-1.5*u*u); };
    for(int x=0;x<nx;++x)for(int i=0;i<Q;++i) f[x*Q+i]=feq(i,1.0,u0);

    auto macro=[&](int x,double&r,double&u){ r=f[x*Q]+f[x*Q+1]+f[x*Q+2];
                                             u=(f[x*Q+1]-f[x*Q+2])/r; };
    double rprev=0; int it=0; const int MAXIT=200000;
    for(it=0; it<MAXIT; ++it){
        // collide
        for(int x=0;x<nx;++x){
            double r,u; macro(x,r,u);
            bool shell=(x>=xc0 && x<xc1);
            if(shell){ double sigma=1.0-perm;
                for(int i=0;i<Q;++i) ft[x*Q+i]=sigma*f[x*Q+OPP[i]]+(1.0-sigma)*f[x*Q+i]; } // partial BB
            else for(int i=0;i<Q;++i){ double fe=feq(i,r,u); ft[x*Q+i]=f[x*Q+i]-omega*(f[x*Q+i]-fe); }
        }
        // stream (periodic buffer, overwritten by BCs)
        for(int x=0;x<nx;++x)for(int i=0;i<Q;++i){ int xn=(x+C[i]+nx)%nx; f[xn*Q+i]=ft[x*Q+i]; }
        // BC: inlet x=0 equilibrium (rho=1,u=u0); outlet x=nx-1 zero-gradient u, rho=1
        for(int i=0;i<Q;++i) f[0*Q+i]=feq(i,1.0,u0);
        double ro,uo; macro(nx-2,ro,uo); for(int i=0;i<Q;++i) f[(nx-1)*Q+i]=feq(i,1.0,uo);
        if(it%500==0){ double r,u; macro(nx/2,r,u); if(it>0 && std::fabs(r-rprev)/r<1e-9) break; rprev=r; }
    }
    // measure pressure drop across the canopy (a few cells clear of the edges)
    double r_up,u_up,r_dn,u_dn; macro(xc0-5,r_up,u_up); macro(xc1+5,r_dn,u_dn);
    double Dp = cs2*(r_up - r_dn);
    double umid; { double rr; macro((xc0+xc1)/2,rr,umid); }
    double loss = Dp/(r_up*u0*u0);               // dimensionless loss coefficient Dp/(rho u^2)
    printf("perm=%.4f u0=%.4f  iters=%d  rho_up=%.6f rho_dn=%.6f  u_mid=%.5f\n",perm,u0,it,r_up,r_dn,umid);
    printf("  Dp=%.4e  loss Dp/(rho*u^2)=%.4f  (over %d canopy cells)\n",Dp,loss,Ncan);
    printf("%.6f %.6f %.6f\n", perm, u0, loss);   // machine-readable: perm u0 loss
    return 0;
}
