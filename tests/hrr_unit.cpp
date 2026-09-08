#include <cstdio>
#include <cmath>
// D3Q19 constants (standard) + verbatim mirror of rr_collide's math to unit-test invariants.
static constexpr int CX[19]={0, 1,-1,0,0,0,0, 1,-1,1,-1, 1,-1,1,-1, 0,0,0,0};
static constexpr int CY[19]={0, 0,0,1,-1,0,0, 1,1,-1,-1, 0,0,0,0, 1,-1,1,-1};
static constexpr int CZ[19]={0, 0,0,0,0,1,-1, 0,0,0,0, 1,1,-1,-1, 1,1,-1,-1};
static const float W19[19]={1.f/3.f,
  1.f/18.f,1.f/18.f,1.f/18.f,1.f/18.f,1.f/18.f,1.f/18.f,
  1.f/36.f,1.f/36.f,1.f/36.f,1.f/36.f,1.f/36.f,1.f/36.f,
  1.f/36.f,1.f/36.f,1.f/36.f,1.f/36.f,1.f/36.f,1.f/36.f};

void rr_collide(const float* fl,float r,float u,float v,float w,float omega,float,float* fp){
    const float cs2=1.f/3.f; float usq=u*u+v*v+w*w; float feq[19],fneq[19];
    for(int i=0;i<19;++i){float cu=CX[i]*u+CY[i]*v+CZ[i]*w;
        feq[i]=W19[i]*r*(1.f+3.f*cu+4.5f*cu*cu-1.5f*usq); fneq[i]=fl[i]-feq[i];}
    float axx=0,ayy=0,azz=0,axy=0,axz=0,ayz=0;
    for(int i=0;i<19;++i){float fn=fneq[i];float cx=CX[i],cy=CY[i],cz=CZ[i];
        axx+=cx*cx*fn;ayy+=cy*cy*fn;azz+=cz*cz*fn;axy+=cx*cy*fn;axz+=cx*cz*fn;ayz+=cy*cz*fn;}
    float axxy=2*u*axy+v*axx,axxz=2*u*axz+w*axx,ayyx=2*v*axy+u*ayy,
          ayyz=2*v*ayz+w*ayy,azzx=2*w*axz+u*azz,azzy=2*w*ayz+v*azz;
    for(int i=0;i<19;++i){float cx=CX[i],cy=CY[i],cz=CZ[i];
        float H2a2=axx*(cx*cx-cs2)+ayy*(cy*cy-cs2)+azz*(cz*cz-cs2)+2*(axy*cx*cy+axz*cx*cz+ayz*cy*cz);
        float H3a3=axxy*cy*(cx*cx-cs2)+axxz*cz*(cx*cx-cs2)+ayyx*cx*(cy*cy-cs2)
                  +ayyz*cz*(cy*cy-cs2)+azzx*cx*(cz*cz-cs2)+azzy*cy*(cz*cz-cs2);
        fp[i]=feq[i]+(1.f-omega)*(W19[i]*(4.5f*H2a2+13.5f*H3a3));}
}
static void feq_of(float r,float u,float v,float w,float*f){
    float usq=u*u+v*v+w*w; for(int i=0;i<19;++i){float cu=CX[i]*u+CY[i]*v+CZ[i]*w;
        f[i]=W19[i]*r*(1.f+3.f*cu+4.5f*cu*cu-1.5f*usq);}}
static void moms(const float*f,float&r,float&mx,float&my,float&mz){
    r=mx=my=mz=0; for(int i=0;i<19;++i){r+=f[i];mx+=CX[i]*f[i];my+=CY[i]*f[i];mz+=CZ[i]*f[i];}}
int main(){
    int fails=0;
    // case set: (rho,u,v,w) plus a random non-eq perturbation
    float cases[][4]={{1.f,0,0,0},{1.f,0.06f,0,0},{1.02f,0.05f,-0.03f,0.02f},{0.98f,0.08f,0.04f,-0.05f}};
    for(auto&c:cases){
        float r=c[0],u=c[1],v=c[2],w=c[3];
        float feq[19]; feq_of(r,u,v,w,feq);
        // (1) equilibrium fixed point: fl=feq -> fp=feq
        float fp[19]; rr_collide(feq,r,u,v,w,1.2f,1.f,fp);
        float dmax=0; for(int i=0;i<19;++i) dmax=fmaxf(dmax,fabsf(fp[i]-feq[i]));
        printf("case r=%.2f u=(%.2f,%.2f,%.2f): eq-fixed-point dev=%.2e %s\n",r,u,v,w,dmax,dmax<1e-6?"OK":"FAIL");
        if(dmax>=1e-6) fails++;
        // (2) conservation on a perturbed state (add a shear-like non-eq)
        float fl[19]; for(int i=0;i<19;++i){ float pert=0.01f*(CX[i]*CY[i]) + 0.005f*(CX[i]*CX[i]-1.f/3.f);
            fl[i]=feq[i]+pert*W19[i]; }
        float r0,mx0,my0,mz0,r1,mx1,my1,mz1;
        moms(fl,r0,mx0,my0,mz0); rr_collide(fl,r,u,v,w,1.2f,1.f,fp); moms(fp,r1,mx1,my1,mz1);
        float dmass=fabsf(r1-r0), dmom=fabsf(mx1-mx0)+fabsf(my1-my0)+fabsf(mz1-mz0);
        printf("            conservation: dmass=%.2e dmom=%.2e %s\n",dmass,dmom,
               (dmass<1e-6&&dmom<1e-6)?"OK":"FAIL");
        if(dmass>=1e-6||dmom>=1e-6) fails++;
    }
    printf("\n%s (%d failure(s))\n", fails?"UNIT TEST FAILED":"ALL UNIT TESTS PASSED", fails);
    return fails?1:0;
}
