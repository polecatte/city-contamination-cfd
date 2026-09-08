// ═══════════════════════════════════════════════════════════════════════
// lbm_kernels.cu — Device code: constants, kernels, GPU wrappers
//
// Compile: nvcc -O3 -arch=sm_XX --extended-lambda -c lbm_kernels.cu
//
// PHYSICS/MATH REFERENCE: this backend is a line-for-line GPU port of
// lbm_kernels_cpu.cpp — the D3Q19 MRT collision, WALE eddy viscosity, D3Q7
// advection-diffusion (TRT diffusion + van Leer/upwind TVD advection), settling
// and surface deposition are identical. See lbm_kernels_cpu.cpp for the formula
// derivations and design rationale; comments here focus on CUDA-specific
// concerns (constant memory, streams, the two-pass scalar launch).
// ═══════════════════════════════════════════════════════════════════════

#include "lbm_gpu.h"
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/execution_policy.h>
#include <cstdio>
#include <cmath>
#include <vector>

// ─── Helpers ─────────────────────────────────────────────────────────

#define CU(call) do { cudaError_t e=(call); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1); }} while(0)

static constexpr int NTHREADS = 256;

// D3Q7 settling↔sticking closure factor = 1/W7_axial = 1/0.125 = 8.
// A gravitational settling velocity ws (lattice units) onto an upward-facing
// surface becomes a per-link capture fraction ws/W7_axial. Kept as a named
// constant (not a bare 8) so it stays consistent with W7 and with the
// deposition-velocity conversion in upload_geometry().
static constexpr float INV_W7_AXIAL = 8.0f;

__host__ __device__ __forceinline__
int idx3(int x,int y,int z,int nx,int ny){ return z*ny*nx+y*nx+x; }

enum : uint8_t { FLUID=0, GROUND=1, SHELL=2, INDOOR=3 };

// ─── D3Q19 lattice (constant memory) ────────────────────────────────

__constant__ int CX[19]={0, 1,-1,0,0,0,0, 1,-1,1,-1, 1,-1,1,-1, 0,0,0,0};
__constant__ int CY[19]={0, 0,0,1,-1,0,0, 1,1,-1,-1, 0,0,0,0, 1,-1,1,-1};
__constant__ int CZ[19]={0, 0,0,0,0,1,-1, 0,0,0,0, 1,1,-1,-1, 1,1,-1,-1};
__constant__ int OPP[19]={0, 2,1,4,3,6,5, 10,9,8,7, 14,13,12,11, 18,17,16,15};
// Axis-mirror maps for SPECULAR (free-slip) reflection: flip ONE component only
// (YMIR flips e_y, ZMIR flips e_z; tangential preserved). Unlike OPP (all three =
// bounce-back/no-slip), these give zero shear at the wall. Mirrors lbm_kernels_cpu.cpp.
__constant__ int YMIR[19]={0,1,2,4,3,5,6,9,10,7,8,11,12,13,14,16,15,18,17};
__constant__ int ZMIR[19]={0,1,2,3,4,6,5,7,8,9,10,13,14,11,12,17,18,15,16};
__constant__ float W19[19]={
    1.f/3,
    1.f/18,1.f/18,1.f/18,1.f/18,1.f/18,1.f/18,
    1.f/36,1.f/36,1.f/36,1.f/36,1.f/36,1.f/36,
    1.f/36,1.f/36,1.f/36,1.f/36,1.f/36,1.f/36};
__constant__ float MNRM2[19]={
    19,2394,252, 10,40,10,40,10,40, 36,72,12,24, 4,4,4, 8,8,8};
__constant__ float S_GHOST[19]={
    0,1.19f,1.4f, 0,1.2f, 0,1.2f, 0,1.2f,
    0,1.4f, 0,1.4f, 0,0,0, 1.98f,1.98f,1.98f};

// ─── D3Q7 lattice ───────────────────────────────────────────────────

__constant__ int EX[7]={0,1,-1,0,0,0,0};
__constant__ int EY[7]={0,0,0,1,-1,0,0};
__constant__ int EZ[7]={0,0,0,0,0,1,-1};
__constant__ int GOPP[7]={0,2,1,4,3,6,5};
__constant__ float W7[7]={.25f,.125f,.125f,.125f,.125f,.125f,.125f};

// ─── MRT forward:  f → m  (d'Humières 2002 Table 1) ────────────────

__device__ __forceinline__
void mrt_fwd(const float f[19], float m[19])
{
    float fxp=f[1]+f[2], fxm=f[1]-f[2];
    float fyp=f[3]+f[4], fym=f[3]-f[4];
    float fzp=f[5]+f[6], fzm=f[5]-f[6];
    float fp=fxp+fyp+fzp;
    float exy=f[7]+f[8]+f[9]+f[10];
    float exz=f[11]+f[12]+f[13]+f[14];
    float eyz=f[15]+f[16]+f[17]+f[18];
    float ep=exy+exz+eyz;
    float exym=f[7]-f[8]+f[9]-f[10], exyp=f[7]+f[8]-f[9]-f[10];
    float exzm=f[11]-f[12]+f[13]-f[14], exzp=f[11]+f[12]-f[13]-f[14];
    float eyzm=f[15]-f[16]+f[17]-f[18], eyzp=f[15]+f[16]-f[17]-f[18];

    m[0] =f[0]+fp+ep;
    m[1] =-30.f*f[0]-11.f*fp+8.f*ep;
    m[2] = 12.f*f[0]- 4.f*fp+ep;
    m[3] =fxm+exym+exzm;            m[4]=-4.f*fxm+exym+exzm;
    m[5] =fym+exyp+eyzm;            m[6]=-4.f*fym+exyp+eyzm;
    m[7] =fzm+exzp+eyzp;            m[8]=-4.f*fzm+exzp+eyzp;
    m[9] = 2.f*fxp-fyp-fzp + exy+exz-2.f*eyz;
    m[10]=-4.f*fxp+2.f*fyp+2.f*fzp + exy+exz-2.f*eyz;
    m[11]= fyp-fzp + exy-exz;
    m[12]=-2.f*fyp+2.f*fzp + exy-exz;
    m[13]=f[7]-f[8]-f[9]+f[10];
    m[14]=f[15]-f[16]-f[17]+f[18];
    m[15]=f[11]-f[12]-f[13]+f[14];
    m[16]=f[7]-f[8]+f[9]-f[10]-f[11]+f[12]-f[13]+f[14];
    m[17]=-f[7]-f[8]+f[9]+f[10]+f[15]-f[16]+f[17]-f[18];
    m[18]=f[11]+f[12]-f[13]-f[14]-f[15]-f[16]+f[17]+f[18];
}

// ─── MRT inverse:  m → f  via  f[j] = Σ M[i][j]·m[i]/‖row_i‖² ────

__device__ __forceinline__
void mrt_inv(const float m[19], float f[19])
{
    float s[19];
    #pragma unroll
    for(int i=0;i<19;++i) s[i]=m[i]/MNRM2[i];

    f[0] =s[0]-30.f*s[1]+12.f*s[2];
    f[1] =s[0]-11.f*s[1]-4.f*s[2]+s[3]-4.f*s[4]+2.f*s[9]-4.f*s[10];
    f[2] =s[0]-11.f*s[1]-4.f*s[2]-s[3]+4.f*s[4]+2.f*s[9]-4.f*s[10];
    f[3] =s[0]-11.f*s[1]-4.f*s[2]+s[5]-4.f*s[6]-s[9]+2.f*s[10]+s[11]-2.f*s[12];
    f[4] =s[0]-11.f*s[1]-4.f*s[2]-s[5]+4.f*s[6]-s[9]+2.f*s[10]+s[11]-2.f*s[12];
    f[5] =s[0]-11.f*s[1]-4.f*s[2]+s[7]-4.f*s[8]-s[9]+2.f*s[10]-s[11]+2.f*s[12];
    f[6] =s[0]-11.f*s[1]-4.f*s[2]-s[7]+4.f*s[8]-s[9]+2.f*s[10]-s[11]+2.f*s[12];
    f[7] =s[0]+8.f*s[1]+s[2]+s[3]+s[4]+s[5]+s[6]+s[9]+s[10]+s[11]+s[12]+s[13]+s[16]-s[17];
    f[8] =s[0]+8.f*s[1]+s[2]-s[3]-s[4]+s[5]+s[6]+s[9]+s[10]+s[11]+s[12]-s[13]-s[16]-s[17];
    f[9] =s[0]+8.f*s[1]+s[2]+s[3]+s[4]-s[5]-s[6]+s[9]+s[10]+s[11]+s[12]-s[13]+s[16]+s[17];
    f[10]=s[0]+8.f*s[1]+s[2]-s[3]-s[4]-s[5]-s[6]+s[9]+s[10]+s[11]+s[12]+s[13]-s[16]+s[17];
    f[11]=s[0]+8.f*s[1]+s[2]+s[3]+s[4]+s[7]+s[8]+s[9]+s[10]-s[11]-s[12]+s[15]-s[16]+s[18];
    f[12]=s[0]+8.f*s[1]+s[2]-s[3]-s[4]+s[7]+s[8]+s[9]+s[10]-s[11]-s[12]-s[15]+s[16]+s[18];
    f[13]=s[0]+8.f*s[1]+s[2]+s[3]+s[4]-s[7]-s[8]+s[9]+s[10]-s[11]-s[12]-s[15]-s[16]-s[18];
    f[14]=s[0]+8.f*s[1]+s[2]-s[3]-s[4]-s[7]-s[8]+s[9]+s[10]-s[11]-s[12]+s[15]+s[16]-s[18];
    f[15]=s[0]+8.f*s[1]+s[2]+s[5]+s[6]+s[7]+s[8]-2.f*s[9]-2.f*s[10]+s[14]+s[17]-s[18];
    f[16]=s[0]+8.f*s[1]+s[2]-s[5]-s[6]+s[7]+s[8]-2.f*s[9]-2.f*s[10]-s[14]-s[17]-s[18];
    f[17]=s[0]+8.f*s[1]+s[2]+s[5]+s[6]-s[7]-s[8]-2.f*s[9]-2.f*s[10]-s[14]+s[17]+s[18];
    f[18]=s[0]+8.f*s[1]+s[2]-s[5]-s[6]-s[7]-s[8]-2.f*s[9]-2.f*s[10]+s[14]-s[17]+s[18];
}

// ─── Equilibrium moments ────────────────────────────────────────────

__device__ __forceinline__
void meq_compute(float rho,float ux,float uy,float uz,float meq[19])
{
    float jx=rho*ux,jy=rho*uy,jz=rho*uz;
    float jsq=jx*jx+jy*jy+jz*jz, ri=1.f/rho;
    meq[0]=rho;
    meq[1]=-11.f*rho+19.f*jsq*ri;
    meq[2]=3.f*rho-5.5f*jsq*ri;
    meq[3]=jx;  meq[4]=-2.f/3.f*jx;
    meq[5]=jy;  meq[6]=-2.f/3.f*jy;
    meq[7]=jz;  meq[8]=-2.f/3.f*jz;
    meq[9]=(2.f*jx*jx-jy*jy-jz*jz)*ri;  meq[10]=-.5f*meq[9];
    meq[11]=(jy*jy-jz*jz)*ri;            meq[12]=-.5f*meq[11];
    meq[13]=jx*jy*ri; meq[14]=jy*jz*ri; meq[15]=jx*jz*ri;
    meq[16]=0; meq[17]=0; meq[18]=0;
}

// ─── WALE eddy viscosity (Nicoud & Ducros 1999) ─────────────────────

__device__ __forceinline__
float wale_nut(float dudx,float dudy,float dudz,
               float dvdx,float dvdy,float dvdz,
               float dwdx,float dwdy,float dwdz,
               float Cw)
{
    float S00=dudx,S11=dvdy,S22=dwdz;
    float S01=.5f*(dudy+dvdx),S02=.5f*(dudz+dwdx),S12=.5f*(dvdz+dwdy);
    float Ssq=S00*S00+S11*S11+S22*S22+2.f*(S01*S01+S02*S02+S12*S12);

    float g200=dudx*dudx+dudy*dvdx+dudz*dwdx;
    float g211=dvdx*dudy+dvdy*dvdy+dvdz*dwdy;
    float g222=dwdx*dudz+dwdy*dvdz+dwdz*dwdz;
    float g201=dudx*dudy+dudy*dvdy+dudz*dwdy;
    float g210=dvdx*dudx+dvdy*dvdx+dvdz*dwdx;
    float g202=dudx*dudz+dudy*dvdz+dudz*dwdz;
    float g220=dwdx*dudx+dwdy*dvdx+dwdz*dwdx;
    float g212=dvdx*dudz+dvdy*dvdz+dvdz*dwdz;
    float g221=dwdx*dudy+dwdy*dvdy+dwdz*dwdy;

    float tr3=(g200+g211+g222)/3.f;
    float Sd00=g200-tr3, Sd11=g211-tr3, Sd22=g222-tr3;
    float Sd01=.5f*(g201+g210), Sd02=.5f*(g202+g220), Sd12=.5f*(g212+g221);
    float A=Sd00*Sd00+Sd11*Sd11+Sd22*Sd22+2.f*(Sd01*Sd01+Sd02*Sd02+Sd12*Sd12);

    const float eps=1e-30f;
    float Af=fmaxf(A,eps), Bf=fmaxf(Ssq,eps);
    float A32=Af*sqrtf(Af);            // A^1.5
    float B52=Bf*Bf*sqrtf(Bf);         // B^2.5
    float A54=Af*sqrtf(sqrtf(Af));     // A^1.25
    float den=B52+A54;
    return (den<1e-20f) ? 0.f : Cw*Cw*A32/den;   // dx=1 in LU, absorbed into Cw²
}

// ═════════════════════════════════════════════════════════════════════
//  KERNELS
// ═════════════════════════════════════════════════════════════════════

// ─── K1: macroscopic from f (for WALE neighbors + scalar advection) ─

__global__ void kern_macro(
    const float* __restrict__ f, int N, int nx, int ny,
    float* __restrict__ rho, float* __restrict__ ux,
    float* __restrict__ uy,  float* __restrict__ uz,
    const uint8_t* __restrict__ tp)
{
    int id=blockIdx.x*blockDim.x+threadIdx.x;
    if(id>=N) return;
    if(__ldg(&tp[id])==GROUND){rho[id]=1;ux[id]=uy[id]=uz[id]=0;return;}
    float r=0,u=0,v=0,w=0;
    #pragma unroll
    for(int i=0;i<19;++i){float fi=f[i*N+id]; r+=fi; u+=CX[i]*fi; v+=CY[i]*fi; w+=CZ[i]*fi;}
    float ri=1.f/r;
    rho[id]=r; ux[id]=u*ri; uy[id]=v*ri; uz[id]=w*ri;
}

// ─── K1b: WALE eddy viscosity → I.nut ───────────────────────────────
//     Computed in its own kernel on the default stream AFTER kern_macro and
//     BEFORE the flow/scalar kernels launch. Previously WALE was computed
//     inside kern_flow (which wrote I.nut) while kern_scalar read I.nut on a
//     concurrent stream — a read/write data race. With nut finalized here,
//     both kern_flow and kern_scalar only READ it. Gradients use the stored
//     macroscopic velocity (central differences), so this is one step lagged
//     by construction — but now deterministically so, not race-dependent.
__global__ void kern_nut(
    const float* __restrict__ Ux, const float* __restrict__ Uy, const float* __restrict__ Uz,
    const uint8_t* __restrict__ tp, float* __restrict__ nut_out,
    int N,int nx,int ny,int nz, float Cw)
{
    int id=blockIdx.x*blockDim.x+threadIdx.x;
    if(id>=N) return;
    int z=id/(ny*nx), yx=id-z*ny*nx, y=yx/nx, x=yx-y*nx;
    float nut=0.f;
    if(__ldg(&tp[id])!=GROUND && Cw>0.f &&
       x>0 && x<nx-1 && y>0 && y<ny-1 && z>0 && z<nz-1){
        int xp=idx3(x+1,y,z,nx,ny),xm=idx3(x-1,y,z,nx,ny);
        int yp=idx3(x,y+1,z,nx,ny),ym=idx3(x,y-1,z,nx,ny);
        int zp=idx3(x,y,z+1,nx,ny),zm=idx3(x,y,z-1,nx,ny);
        nut=wale_nut(
            .5f*(__ldg(&Ux[xp])-__ldg(&Ux[xm])),
            .5f*(__ldg(&Ux[yp])-__ldg(&Ux[ym])),
            .5f*(__ldg(&Ux[zp])-__ldg(&Ux[zm])),
            .5f*(__ldg(&Uy[xp])-__ldg(&Uy[xm])),
            .5f*(__ldg(&Uy[yp])-__ldg(&Uy[ym])),
            .5f*(__ldg(&Uy[zp])-__ldg(&Uy[zm])),
            .5f*(__ldg(&Uz[xp])-__ldg(&Uz[xm])),
            .5f*(__ldg(&Uz[yp])-__ldg(&Uz[ym])),
            .5f*(__ldg(&Uz[zp])-__ldg(&Uz[zm])),
            Cw);
    }
    nut_out[id]=nut;
}

// ─── Recursive-Regularized (RR) collision — device mirror of the CPU
//     rr_collide (lbm_kernels_cpu.cpp). COLLISION=hrr (colReg==2). See that
//     file for the derivation; math is identical (float32). sigma accepted but
//     not yet applied (RR core, sigma=1); FD hybrid is the next increment.
__device__ __forceinline__ void rr_collide(
        const float* fl,float r,float u,float v,float w,float omega,float sigma,
        float Sxx,float Syy,float Szz,float Sxy,float Sxz,float Syz,float* fp){
    const float cs2=1.f/3.f; float usq=u*u+v*v+w*w; float feq[19],fneq[19];
    #pragma unroll
    for(int i=0;i<19;++i){ float cu=CX[i]*u+CY[i]*v+CZ[i]*w;
        feq[i]=W19[i]*r*(1.f+3.f*cu+4.5f*cu*cu-1.5f*usq); fneq[i]=fl[i]-feq[i]; }
    // projected a2 = Pi_neq (used for the 2nd-order term => exact NS viscosity)
    float axx=0,ayy=0,azz=0,axy=0,axz=0,ayz=0;
    #pragma unroll
    for(int i=0;i<19;++i){ float fn=fneq[i]; float cx=CX[i],cy=CY[i],cz=CZ[i];
        axx+=cx*cx*fn;ayy+=cy*cy*fn;azz+=cz*cz*fn;axy+=cx*cy*fn;axz+=cx*cz*fn;ayz+=cy*cz*fn; }
    // HRR hybrid: stress for the 3rd-order recursion = sigma*a2 + (1-sigma)*a2_FD,
    // a2_FD = -(2 rho/(3 omega)) S (FD strain). Hyperviscosity via 3rd order only.
    float bxx=axx,byy=ayy,bzz=azz,bxy=axy,bxz=axz,byz=ayz;
    if(sigma<1.f){ float k=-2.f*r/(3.f*omega),s1=1.f-sigma;
        bxx=sigma*axx+s1*k*Sxx;byy=sigma*ayy+s1*k*Syy;bzz=sigma*azz+s1*k*Szz;
        bxy=sigma*axy+s1*k*Sxy;bxz=sigma*axz+s1*k*Sxz;byz=sigma*ayz+s1*k*Syz; }
    float axxy=2.f*u*bxy+v*bxx,axxz=2.f*u*bxz+w*bxx,ayyx=2.f*v*bxy+u*byy,
          ayyz=2.f*v*byz+w*byy,azzx=2.f*w*bxz+u*bzz,azzy=2.f*w*byz+v*bzz;
    #pragma unroll
    for(int i=0;i<19;++i){ float cx=CX[i],cy=CY[i],cz=CZ[i];
        float H2a2=axx*(cx*cx-cs2)+ayy*(cy*cy-cs2)+azz*(cz*cz-cs2)+2.f*(axy*cx*cy+axz*cx*cz+ayz*cy*cz);
        float H3a3=axxy*cy*(cx*cx-cs2)+axxz*cz*(cx*cx-cs2)+ayyx*cx*(cy*cy-cs2)
                  +ayyz*cz*(cy*cy-cs2)+azzx*cx*(cz*cz-cs2)+azzy*cy*(cz*cz-cs2);
        fp[i]=feq[i]+(1.f-omega)*(W19[i]*(4.5f*H2a2+13.5f*H3a3)); }
}

// ─── K2: flow — pull · MRT-WALE / HRR collision · write ──────────────
//     __ldg() for read-only geometry; streams for concurrency

__global__ void kern_flow(
    const float* __restrict__ fs, float* __restrict__ fd,
    const float* __restrict__ Ux, const float* __restrict__ Uy, const float* __restrict__ Uz,
    const uint8_t* __restrict__ tp, const float* __restrict__ pm,
    const float* __restrict__ nut_in,
    int N,int nx,int ny,int nz,
    float nu0,float Cw,
    float uIx,float uIy,float uIz,
    int inAx,int inS,int outAx,int outS,
    const float* __restrict__ inlet_plane,int inlet_stride,int inlet_profile,
    int colReg,float hrrSigma,int wallM,float uwX,float uwY,int latBC,
    int perFace,int fxm,int fxp,int fym,int fyp,
    const float* __restrict__ inlet_plane_y,const float* __restrict__ mean_plane)
{
    int id=blockIdx.x*blockDim.x+threadIdx.x;
    if(id>=N) return;
    uint8_t t=__ldg(&tp[id]);
    if(t==GROUND){
        #pragma unroll
        for(int i=0;i<19;++i) fd[i*N+id]=fs[OPP[i]*N+id];
        return;
    }
    int z=id/(ny*nx), yx=id-z*ny*nx, y=yx/nx, x=yx-y*nx;

    // ── Pull ──
    float fl[19];
    #pragma unroll
    for(int i=0;i<19;++i){
        int sx=x-CX[i],sy=y-CY[i],sz=z-CZ[i];
        sx=max(0,min(nx-1,sx)); sy=max(0,min(ny-1,sy)); sz=max(0,min(nz-1,sz));
        int nb=idx3(sx,sy,sz,nx,ny);
        if(__ldg(&tp[nb])==GROUND){
            float bb=fs[OPP[i]*N+id];
            // Moving-wall bounce-back on the DOMAIN FLOOR only (sz==0): rough-wall
            // log-law slip u_w (mirrors lbm_kernels_cpu.cpp). Building walls (sz==z)
            // and roofs (sz=roof≠0) keep no-slip. +6 w_i (e_i·u_w) with ρ≈1.
            if(wallM && sz==0) bb += 6.f*W19[i]*(CX[i]*uwX + CY[i]*uwY);
            fl[i]=bb;
        } else fl[i]=fs[i*N+nb];
    }

    // ── Local macroscopic (from pulled distributions) ──
    float r=0,u=0,v=0,w=0;
    #pragma unroll
    for(int i=0;i<19;++i){r+=fl[i]; u+=CX[i]*fl[i]; v+=CY[i]*fl[i]; w+=CZ[i]*fl[i];}
    float ri=1.f/r; u*=ri; v*=ri; w*=ri;

    // ── WALE eddy viscosity: read the value finalized by kern_nut (no race) ──
    float nut=__ldg(&nut_in[id]);
    // nu0 (= I.nu_lb) is already floored to nu_floor in lbm_solver.cpp — the
    // single source of truth for the stability floor. nut>=0 and nu0>=nu_floor>0,
    // so nu_eff>0 (tau>0.5) is guaranteed without a separate hardcoded floor here.
    // (The old `if(nu_eff<1e-3) nu_eff=1e-3` silently capped the maximum Re and
    // overrode NU_FLOOR — removed.)
    float nu_eff=nu0+nut;
    float snu=1.f/(3.f*nu_eff+.5f);

    // ── collision: HRR/RR (colReg==2) or MRT (0=plain, 1=projected-reg) ──
    float fp[19];
    if(colReg==2){
        // HRR/RR: strain rate from central-difference velocity gradients (interior
        // only; boundaries pass sigma=1 => pure RR). Mirrors the CPU backend.
        float Sxx=0,Syy=0,Szz=0,Sxy=0,Sxz=0,Syz=0, sig=hrrSigma;
        if(hrrSigma<1.f && x>0&&x<nx-1 && y>0&&y<ny-1 && z>0&&z<nz-1){
            int xp=idx3(x+1,y,z,nx,ny),xm=idx3(x-1,y,z,nx,ny);
            int yp=idx3(x,y+1,z,nx,ny),ym=idx3(x,y-1,z,nx,ny);
            int zp=idx3(x,y,z+1,nx,ny),zm=idx3(x,y,z-1,nx,ny);
            float dux=.5f*(__ldg(&Ux[xp])-__ldg(&Ux[xm])),duy=.5f*(__ldg(&Ux[yp])-__ldg(&Ux[ym])),duz=.5f*(__ldg(&Ux[zp])-__ldg(&Ux[zm]));
            float dvx=.5f*(__ldg(&Uy[xp])-__ldg(&Uy[xm])),dvy=.5f*(__ldg(&Uy[yp])-__ldg(&Uy[ym])),dvz=.5f*(__ldg(&Uy[zp])-__ldg(&Uy[zm]));
            float dwx=.5f*(__ldg(&Uz[xp])-__ldg(&Uz[xm])),dwy=.5f*(__ldg(&Uz[yp])-__ldg(&Uz[ym])),dwz=.5f*(__ldg(&Uz[zp])-__ldg(&Uz[zm]));
            Sxx=dux; Syy=dvy; Szz=dwz;
            Sxy=.5f*(duy+dvx); Sxz=.5f*(duz+dwx); Syz=.5f*(dvz+dwy);
        } else sig=1.f;
        rr_collide(fl, r,u,v,w, snu, sig, Sxx,Syy,Szz,Sxy,Sxz,Syz, fp);
    }else{
        float m[19],meq[19];
        mrt_fwd(fl,m);
        meq_compute(r,u,v,w,meq);
        float sr[19];
        #pragma unroll
        for(int i=0;i<19;++i) sr[i]=S_GHOST[i];
        sr[9]=sr[11]=sr[13]=sr[14]=sr[15]=snu;
        // Projected regularization (mirrors lbm_kernels_cpu.cpp): relax ghost moments
        // fully to (zero) equilibrium — filters the aliased modes that blow up at τ→0.5.
        // Conserved + shear moments untouched ⇒ recovered viscosity unchanged.
        if(colReg){ sr[1]=sr[2]=sr[4]=sr[6]=sr[8]=sr[10]=sr[12]=sr[16]=sr[17]=sr[18]=1.f; }
        #pragma unroll
        for(int i=0;i<19;++i) m[i]-=sr[i]*(m[i]-meq[i]);
        mrt_inv(m,fp);
    }

    // ── Write ──
    if(t==SHELL){
        float sigma=1.f-__ldg(&pm[id]);
        #pragma unroll
        for(int i=0;i<19;++i) fd[i*N+id]=sigma*fp[OPP[i]]+(1.f-sigma)*fp[i];
    }else{
        #pragma unroll
        for(int i=0;i<19;++i) fd[i*N+id]=fp[i];
    }

    // ── BC: inlet ──
    // Sheared turbulent ABL inlet (Richards & Hoxey 1993 mean + RFG turbulence,
    // Kraichnan 1970 / Smirnov et al. 2001): when inlet_profile==1, read the
    // per-cell velocity (lattice units) from the host-filled plane uploaded each
    // step to d_inlet_plane; otherwise fall back to the uniform plug (uIx,uIy,uIz).
    // Mirrors lbm_kernels_cpu.cpp; indexing matches fill_inlet_plane (z*stride+y
    // for an x-normal inlet, z*stride+x for y-normal).
    if(perFace){
        // ── Per-face inflow/outflow/symmetry (oblique-wind fix) ─────────────────
        // Each vertical face is INLET/OUTFLOW/SYM (host-classified by mean normal
        // flux; passed as fxm/fxp/fym/fyp). Oblique wind => TWO inlet faces; their
        // shared vertical edge is written from the RFG-free MEAN (mean_plane) so it
        // is continuous — removing the inlet-corner discontinuity that blew up.
        // Aligned wind keeps ±y as symmetry, reproducing the legacy result exactly.
        int fx=(x==0)?fxm:(x==nx-1)?fxp:-1;
        int fy=(y==0)?fym:(y==ny-1)?fyp:-1;
        bool xin=(fx==1), yin=(fy==1);       // 1=INLET 2=OUTFLOW 0=SYM (lbm::bc_mode)
        bool xout=(fx==2), yout=(fy==2);
        if(t!=GROUND){
            if(xin||yin){
                float ix,iy,iz;
                if(inlet_profile!=1){ ix=uIx; iy=uIy; iz=uIz; }
                else if(xin&&yin){ ix=mean_plane[3*z]; iy=mean_plane[3*z+1]; iz=mean_plane[3*z+2]; }
                else if(xin){ int k=z*inlet_stride+y; ix=inlet_plane[3*k]; iy=inlet_plane[3*k+1]; iz=inlet_plane[3*k+2]; }
                else        { int k=z*inlet_stride+x; ix=inlet_plane_y[3*k]; iy=inlet_plane_y[3*k+1]; iz=inlet_plane_y[3*k+2]; }
                float usq=ix*ix+iy*iy+iz*iz;
                #pragma unroll
                for(int i=0;i<19;++i){ float cu=CX[i]*ix+CY[i]*iy+CZ[i]*iz;
                    fd[i*N+id]=W19[i]*r*(1.f+3.f*cu+4.5f*cu*cu-1.5f*usq); }
            } else if(xout||yout){           // convective/zero-gradient outflow
                int ix=x,iy=y; if(xout) ix+=(x==0)?1:-1; else iy+=(y==0)?1:-1;
                int src=idx3(ix,iy,z,nx,ny);
                #pragma unroll
                for(int i=0;i<19;++i) fd[i*N+id]=fs[i*N+src];
            } else {                         // symmetry faces + top (+z)
                if(z==nz-1){ for(int i=0;i<19;++i) if(CZ[i]<0) fd[i*N+id]=fd[ZMIR[i]*N+id]; }
                if(fy==0){
                    if(y==ny-1){ for(int i=0;i<19;++i) if(CY[i]<0) fd[i*N+id]=fd[YMIR[i]*N+id]; }
                    if(y==0)   { for(int i=0;i<19;++i) if(CY[i]>0) fd[i*N+id]=fd[YMIR[i]*N+id]; }
                }
            }
        }
    } else {   // ===== legacy single-inlet path (WIND_BC=legacy) =====
    bool inlet=(inAx==0)?((inS==0&&x==0)||(inS==1&&x==nx-1))
              :(inAx==1)?((inS==0&&y==0)||(inS==1&&y==ny-1)):false;
    if(inlet&&t!=GROUND){
        float ix=uIx,iy=uIy,iz=uIz;
        if(inlet_profile==1){
            int kpl=(inAx==0)?(z*inlet_stride+y):(z*inlet_stride+x);
            ix=inlet_plane[3*kpl]; iy=inlet_plane[3*kpl+1]; iz=inlet_plane[3*kpl+2];
        }
        float usq=ix*ix+iy*iy+iz*iz;
        #pragma unroll
        for(int i=0;i<19;++i){
            float cu=CX[i]*ix+CY[i]*iy+CZ[i]*iz;
            fd[i*N+id]=W19[i]*r*(1.f+3.f*cu+4.5f*cu*cu-1.5f*usq);
        }
    }
    // ── BC: outlet ──
    bool outlet=(outAx==0)?((outS==0&&x==0)||(outS==1&&x==nx-1))
               :(outAx==1)?((outS==0&&y==0)||(outS==1&&y==ny-1)):false;
    if(outlet&&!inlet&&t!=GROUND){
        int ix=x,iy=y;
        if(outAx==0) ix+=(outS==1)?-1:1;
        if(outAx==1) iy+=(outS==1)?-1:1;
        int src=idx3(ix,iy,z,nx,ny);
        #pragma unroll
        for(int i=0;i<19;++i) fd[i*N+id]=fs[i*N+src];
    }
    // ── BC: free-slip (symmetry) on top (+z) and lateral (±y) ──
    // Specular reflection of the outward-normal populations (tangential preserved).
    // ±y supersedes the clamped self-pull (accelerated near-boundary flow, T5).
    // COST-732/AIJ symmetry BC (Franke et al. 2007; Tominaga et al. 2008); specular
    // reflection per Krüger et al. 2017, "The Lattice Boltzmann Method" §5.3.4.
    if(t!=GROUND&&!inlet&&!outlet){
        // Set the UNKNOWN post-stream populations (upstream node outside the domain)
        // to their axis-MIRROR partner — flips only the wall-normal component, giving
        // zero normal flux + zero shear (true symmetry). OPP would flip tangential too
        // (bounce-back) and inject spurious near-wall momentum that blew up large
        // domains (~1e4 steps). Axis-mirror fixes it. Mirrors lbm_kernels_cpu.cpp.
        if(z==nz-1){ for(int i=0;i<19;++i) if(CZ[i]<0) fd[i*N+id]=fd[ZMIR[i]*N+id]; } // top +z
        if(latBC==1){
            // Experiment #6: OPEN zero-gradient ±y (mirror of the outlet). Copy
            // populations from the interior y-neighbor (source buffer fs → race-free)
            // so a mean cross-stream advects OUT of ±y instead of being reflected.
            if(y==ny-1){ int s=idx3(x,y-1,z,nx,ny); for(int i=0;i<19;++i) fd[i*N+id]=fs[i*N+s]; }
            if(y==0)   { int s=idx3(x,y+1,z,nx,ny); for(int i=0;i<19;++i) fd[i*N+id]=fs[i*N+s]; }
        } else {
            if(y==ny-1){ for(int i=0;i<19;++i) if(CY[i]<0) fd[i*N+id]=fd[YMIR[i]*N+id]; } // +y lateral
            if(y==0)   { for(int i=0;i<19;++i) if(CY[i]>0) fd[i*N+id]=fd[YMIR[i]*N+id]; } // −y lateral
        }
    }
    } // end legacy WIND_BC path
}

// ─── K3 helpers: flux-limited (TVD) advection of the macroscopic scalar ──────
// Mirrors the CPU implementation in lbm_kernels_cpu.cpp (advective form −u·∇C;
// open faces allow outflow; solid faces block; 2-pass consistent C field).
__device__ __forceinline__ float vanleer_psi(float r){ return r<=0.f?0.f:(r+r)/(1.f+r); }
__device__ __forceinline__ float faceflux(float vf,float Cl,float Cr,float Clu,float Cru,int mode,
                                          bool luValid=true, bool ruValid=true){
    float Cface;
    if(vf>=0.f){ Cface=Cl;
        if(mode==0){ float den=Cr-Cl; if(fabsf(den)>1e-12f) Cface=Cl+0.5f*vanleer_psi((Cl-Clu)/den)*den; }
        else if(mode==3 && luValid){ Cface=0.75f*Cl+0.375f*Cr-0.125f*Clu; } }   // QUICK
    else       { Cface=Cr;
        if(mode==0){ float den=Cl-Cr; if(fabsf(den)>1e-12f) Cface=Cr+0.5f*vanleer_psi((Cr-Cru)/den)*den; }
        else if(mode==3 && ruValid){ Cface=0.75f*Cr+0.375f*Cl-0.125f*Cru; } }   // QUICK (mirror)
    return vf*Cface;
}
__device__ float advect_dC(int x,int y,int z,int nx,int ny,int nz,
    const float* C,const float* UX,const float* UY,const float* UZ,
    const uint8_t* tp,float ws,int mode)
{
    auto inb=[&](int xx,int yy,int zz){ return xx>=0&&xx<nx&&yy>=0&&yy<ny&&zz>=0&&zz<nz; };
    auto solid=[&](int xx,int yy,int zz){ if(!inb(xx,yy,zz)) return false; uint8_t t=tp[idx3(xx,yy,zz,nx,ny)]; return (t==GROUND||t==SHELL); };
    auto Cat=[&](int xx,int yy,int zz){ if(!inb(xx,yy,zz)||solid(xx,yy,zz)) return 0.f; return C[idx3(xx,yy,zz,nx,ny)]; };
    auto Vat=[&](int xx,int yy,int zz,int ax){ if(!inb(xx,yy,zz)) return 0.f; int id=idx3(xx,yy,zz,nx,ny); return ax==0?UX[id]:ax==1?UY[id]:(UZ[id]-ws); };
    float dC=0.f, divu=0.f;
    const int off[3][3]={{1,0,0},{0,1,0},{0,0,1}};
    for(int ax=0;ax<3;++ax){
        int dx=off[ax][0],dy=off[ax][1],dz=off[ax][2];
        float Cm2=Cat(x-2*dx,y-2*dy,z-2*dz),Cm1=Cat(x-dx,y-dy,z-dz),C0=Cat(x,y,z),
              Cp1=Cat(x+dx,y+dy,z+dz),Cp2=Cat(x+2*dx,y+2*dy,z+2*dz);
        float V0=Vat(x,y,z,ax);
        float Fp=0.f,vfp=0.f;
        if(!solid(x+dx,y+dy,z+dz)){ float vR=inb(x+dx,y+dy,z+dz)?Vat(x+dx,y+dy,z+dz,ax):V0; vfp=0.5f*(V0+vR);
            bool luP=inb(x-dx,y-dy,z-dz)&&!solid(x-dx,y-dy,z-dz);
            bool ruP=inb(x+2*dx,y+2*dy,z+2*dz)&&!solid(x+2*dx,y+2*dy,z+2*dz);
            Fp=faceflux(vfp,C0,Cp1,Cm1,Cp2,mode,luP,ruP); }
        float Fm=0.f,vfm=0.f;
        if(!solid(x-dx,y-dy,z-dz)){ float vL=inb(x-dx,y-dy,z-dz)?Vat(x-dx,y-dy,z-dz,ax):V0; vfm=0.5f*(vL+V0);
            bool luM=inb(x-2*dx,y-2*dy,z-2*dz)&&!solid(x-2*dx,y-2*dy,z-2*dz);
            bool ruM=inb(x+dx,y+dy,z+dz)&&!solid(x+dx,y+dy,z+dz);
            Fm=faceflux(vfm,Cm1,C0,Cm2,Cp1,mode,luM,ruM); }
        dC+=(Fm-Fp); divu+=(vfp-vfm);
    }
    return dC + Cat(x,y,z)*divu;        // advective form −u·∇C (passive scalar)
}

// ─── K3 pass 1: current post-stream macroscopic C (TVD/upwind only) ──────────
__global__ void kern_cfield(const float* __restrict__ gs, const uint8_t* __restrict__ tp,
    const float* __restrict__ dvel, float* __restrict__ cmac,
    int N,int nx,int ny,int nz,float ws)
{
    int id=blockIdx.x*blockDim.x+threadIdx.x; if(id>=N) return;
    float sa=ws*INV_W7_AXIAL; if(sa>1.f) sa=1.f;
    uint8_t t=__ldg(&tp[id]);
    if(t==GROUND){ cmac[id]=0.f; return; }
    int z=id/(ny*nx),yx=id-z*ny*nx,y=yx/nx,x=yx-y*nx;
    float C=0.f;
    #pragma unroll
    for(int k=0;k<7;++k){
        int sx=x-EX[k],sy=y-EY[k],sz=z-EZ[k];
        if(sx<0||sx>=nx||sy<0||sy>=ny||sz<0||sz>=nz) continue;
        int nb=idx3(sx,sy,sz,nx,ny);
        if(__ldg(&tp[nb])==GROUND){
            float a=__ldg(&dvel[nb]); if(EZ[k]>0) a+=sa; if(a>1.f) a=1.f;
            C+=(1.f-a)*gs[GOPP[k]*N+id];
        }else C+=gs[k*N+nb];
    }
    cmac[id]=C;
}

// ─── K3 pass 2: scalar — pull · (advect) · diffuse · deposit · write ─────────
__global__ void kern_scalar(
    const float* __restrict__ gs, float* __restrict__ gd,
    const float* __restrict__ Ux, const float* __restrict__ Uy, const float* __restrict__ Uz,
    const uint8_t* __restrict__ tp, const float* __restrict__ pm,
    const float* __restrict__ nut_in,
    const float* __restrict__ dvel, float* __restrict__ dep, float* __restrict__ cint,
    const float* __restrict__ cmac0,
    int N,int nx,int ny,int nz,
    float D0,float Sc_t,int srcIdx,float Qdt,float ws,int adv,
    int inAx,int inS, const uint8_t* __restrict__ srcmask)
{
    int id=blockIdx.x*blockDim.x+threadIdx.x;
    if(id>=N) return;
    float sa=ws*INV_W7_AXIAL; if(sa>1.f) sa=1.f;
    uint8_t t=__ldg(&tp[id]);
    if(t==GROUND){
        #pragma unroll
        for(int k=0;k<7;++k) gd[k*N+id]=gs[GOPP[k]*N+id];
        return;
    }
    int z=id/(ny*nx),yx=id-z*ny*nx,y=yx/nx,x=yx-y*nx;

    // Pull. Far-field zero inflow out of domain (conservative outflow); absorbing
    // bounce-back at GROUND links (settling flux deposits on upward-facing walls).
    float gl[7]; float dep_here=0.f;
    #pragma unroll
    for(int k=0;k<7;++k){
        int sx=x-EX[k],sy=y-EY[k],sz=z-EZ[k];
        if(sx<0||sx>=nx||sy<0||sy>=ny||sz<0||sz>=nz){ gl[k]=0.f; continue; }
        int nb=idx3(sx,sy,sz,nx,ny);
        if(__ldg(&tp[nb])==GROUND){
            float gin=gs[GOPP[k]*N+id];
            float a=__ldg(&dvel[nb]); if(EZ[k]>0) a+=sa; if(a>1.f) a=1.f;
            gl[k]=(1.f-a)*gin; dep_here+=a*gin;
        }else gl[k]=gs[k*N+nb];
    }
    float C=0; for(int k=0;k<7;++k) C+=gl[k];
    cint[id]+=C;
    float u=__ldg(&Ux[id]),v=__ldg(&Uy[id]),w=__ldg(&Uz[id]);
    // D0 (= I.D_lb) is pre-floored to D_floor in lbm_solver.cpp; nut>=0, so
    // D_eff>0 without a separate hardcoded floor (which would override D_FLOOR).
    float D_eff=D0+__ldg(&nut_in[id])/Sc_t;

    float gp[7];
    if(adv==2){
        // Legacy: TRT with central advection in the equilibrium (c_s²=1/3).
        float inv_tau=1.f/(3.f*D_eff+0.5f), wadv=w-ws;
        float geq[7];
        for(int k=0;k<7;++k){ float cu=EX[k]*u+EY[k]*v+EZ[k]*wadv; geq[k]=W7[k]*C*(1.f+3.f*cu); }
        float wm=inv_tau, wp=1.f/(0.5f+1.f/(12.f*D_eff));
        for(int k=0;k<7;++k){ int ko=GOPP[k];
            float g_s=0.5f*(gl[k]+gl[ko]),g_a=0.5f*(gl[k]-gl[ko]);
            float e_s=0.5f*(geq[k]+geq[ko]),e_a=0.5f*(geq[k]-geq[ko]);
            gp[k]=gl[k]-wp*(g_s-e_s)-wm*(g_a-e_a); }
    }else{
        // Sequential split: advect (van Leer/upwind) the streamed field FIRST,
        // then TRT diffusion (isotropic eq, correct c_s²=1/4). Stable, positive.
        if(t==FLUID||t==INDOOR){
            float dCadv=advect_dC(x,y,z,nx,ny,nz,cmac0,Ux,Uy,Uz,tp,ws,adv);
            for(int k=0;k<7;++k) gl[k]+=W7[k]*dCadv;
            C+=dCadv;
        }
        float inv_tau=1.f/(4.f*D_eff+0.5f);
        float wm=inv_tau, wp=1.f/(0.5f+1.f/(16.f*D_eff));
        for(int k=0;k<7;++k){ int ko=GOPP[k];
            float g_s=0.5f*(gl[k]+gl[ko]),g_a=0.5f*(gl[k]-gl[ko]);
            float e_s=0.5f*(W7[k]+W7[ko])*C;
            gp[k]=gl[k]-wp*(g_s-e_s)-wm*g_a; }
    }
    // Source injection: per-cell MASK (source distribution over Ω) if provided,
    // else the single srcIdx cell. Null mask ⇒ exactly the original behavior.
    if(srcmask ? (srcmask[id]!=0) : (id==srcIdx))
        for(int k=0;k<7;++k) gp[k]+=Qdt*W7[k];

    if(t==SHELL){
        float sigma=1.f-__ldg(&pm[id]);
        #pragma unroll
        for(int k=0;k<7;++k){
            float a=__ldg(&dvel[id]); if(EZ[k]>0) a+=sa; if(a>1.f) a=1.f;
            float refl=sigma*gp[GOPP[k]];
            gd[k*N+id]=(1.f-a)*refl+(1.f-sigma)*gp[k];
            dep_here+=a*refl;
        }
    }else{
        #pragma unroll
        for(int k=0;k<7;++k) gd[k*N+id]=gp[k];
    }
    if(dep_here>0.f) dep[id]+=dep_here;
    bool inlet=(inAx==0)?((inS==0&&x==0)||(inS==1&&x==nx-1))
              :(inAx==1)?((inS==0&&y==0)||(inS==1&&y==ny-1)):false;
    if(inlet&&t!=GROUND) for(int k=0;k<7;++k) gd[k*N+id]=0;
}

// ═════════════════════════════════════════════════════════════════════
//  GPU WRAPPER FUNCTIONS  (called from lbm_solver.cpp)
// ═════════════════════════════════════════════════════════════════════

namespace lbm { namespace gpu {

void alloc(Solver::Impl& I) {
    int N=I.N; size_t Nf=N*sizeof(float);
    // ── Memory pre-flight (every run): abort BEFORE allocating if it won't fit.
    // Dynamic + system-adaptive: queries the actual free memory on this device.
    {
        size_t need = bytes_per_cell()*(size_t)N
                    + (size_t)3*std::max(I.cfg.nx,I.cfg.ny)*I.cfg.nz*sizeof(float); // inlet plane
        size_t freeB=0, totalB=0; cudaMemGetInfo(&freeB, &totalB);
        double GB=1.0/(1024*1024*1024);
        printf("[LBM] Memory pre-flight: need %.2f GB (%zu B/cell × %.1fM cells) | "
               "free %.2f / %.2f GB on device\n",
               need*GB, bytes_per_cell(), N/1e6, freeB*GB, totalB*GB);
        if (need > freeB) {
            fprintf(stderr,
                "[LBM] FATAL: insufficient device memory — need %.2f GB but only %.2f GB free "
                "(%.2f GB total). Grid %d×%d×%d = %.1fM cells at %zu B/cell.\n"
                "       Reduce the domain (smaller OUTFLOW_H / city_m) or cell count; cell size is fixed.\n",
                need*GB, freeB*GB, totalB*GB, I.cfg.nx, I.cfg.ny, I.cfg.nz, N/1e6, bytes_per_cell());
            std::fflush(stderr); std::exit(3);
        }
    }
    for(int i=0;i<2;++i){ CU(cudaMalloc(&I.f[i],19*Nf)); CU(cudaMalloc(&I.g[i],7*Nf)); }
    CU(cudaMalloc(&I.rho,Nf)); CU(cudaMalloc(&I.ux,Nf));
    CU(cudaMalloc(&I.uy,Nf));  CU(cudaMalloc(&I.uz,Nf));
    CU(cudaMalloc(&I.ux0,Nf)); CU(cudaMalloc(&I.uy0,Nf)); CU(cudaMalloc(&I.uz0,Nf));
    CU(cudaMalloc(&I.tp,N*sizeof(uint8_t)));
    CU(cudaMalloc(&I.pm,Nf)); CU(cudaMalloc(&I.inh,Nf));
    CU(cudaMalloc(&I.nut,Nf));
    CU(cudaMemset(I.nut,0,Nf));
    // ABL inlet device mirror: 3 components * nz * max(nx,ny) (= inlet_stride,
    // set on the host after alloc). Sized from cfg here; always allocated.
    CU(cudaMalloc(&I.d_inlet_plane,
        sizeof(float)*3*(size_t)I.cfg.nz*(I.cfg.nx>I.cfg.ny?I.cfg.nx:I.cfg.ny)));
    // Oblique-wind per-face BC: second inlet plane (y-inlet face) + RFG-free mean
    // profile (shared-edge tie-break). Same sizing as d_inlet_plane; mean is 3*nz.
    CU(cudaMalloc(&I.d_inlet_plane_y,
        sizeof(float)*3*(size_t)I.cfg.nz*(I.cfg.nx>I.cfg.ny?I.cfg.nx:I.cfg.ny)));
    CU(cudaMalloc(&I.d_mean_plane, sizeof(float)*3*(size_t)I.cfg.nz));
    CU(cudaMalloc(&I.dvel,Nf)); CU(cudaMemset(I.dvel,0,Nf));
    CU(cudaMalloc(&I.dep,Nf));  CU(cudaMemset(I.dep,0,Nf));
    CU(cudaMalloc(&I.cint,Nf)); CU(cudaMemset(I.cint,0,Nf));
    CU(cudaMalloc(&I.cmac[0],Nf)); CU(cudaMemset(I.cmac[0],0,Nf));
    CU(cudaMalloc(&I.cmac[1],Nf)); CU(cudaMemset(I.cmac[1],0,Nf));
    I.warm_pending=false;
    CU(cudaMalloc(&I.ux_sum,Nf)); CU(cudaMalloc(&I.uy_sum,Nf)); CU(cudaMalloc(&I.rho_sum,Nf));
    CU(cudaMalloc(&I.uz_sum,Nf)); CU(cudaMalloc(&I.C_sum,Nf));
    I.n_avg=0;

    // Streams + event for overlapping flow and scalar kernels
    cudaStream_t sf,ss; cudaEvent_t ev;
    CU(cudaStreamCreate(&sf)); CU(cudaStreamCreate(&ss));
    CU(cudaEventCreate(&ev));
    I.stream_flow   = (void*)sf;
    I.stream_scalar = (void*)ss;
    I.evt_macro_done= (void*)ev;
}

void free_all(Solver::Impl& I) {
    for(int i=0;i<2;++i){ cudaFree(I.f[i]); cudaFree(I.g[i]); }
    cudaFree(I.rho); cudaFree(I.ux); cudaFree(I.uy); cudaFree(I.uz);
    cudaFree(I.ux0); cudaFree(I.uy0); cudaFree(I.uz0);
    cudaFree(I.tp); cudaFree(I.pm); cudaFree(I.inh);
    cudaFree(I.nut);
    cudaFree(I.d_inlet_plane); cudaFree(I.d_inlet_plane_y); cudaFree(I.d_mean_plane);
    cudaFree(I.dvel); cudaFree(I.dep); cudaFree(I.cint);
    cudaFree(I.cmac[0]); cudaFree(I.cmac[1]);
    cudaFree(I.ux_sum); cudaFree(I.uy_sum); cudaFree(I.rho_sum);
    cudaFree(I.uz_sum); cudaFree(I.C_sum);
    if(I.d_srcmask){ cudaFree(I.d_srcmask); I.d_srcmask=nullptr; }
    cudaStreamDestroy((cudaStream_t)I.stream_flow);
    cudaStreamDestroy((cudaStream_t)I.stream_scalar);
    cudaEventDestroy((cudaEvent_t)I.evt_macro_done);
}

// Upload the host source-distribution mask (set via Solver::load_source_mask →
// Impl::srcmask) to device memory so kern_scalar can read it. No-op if there is no
// mask or it is already resident. Called from Solver::run() before Phase B.
void sync_source_mask(Solver::Impl& I) {
    if(I.srcmask && !I.d_srcmask){
        CU(cudaMalloc(&I.d_srcmask, (size_t)I.N * sizeof(uint8_t)));
        CU(cudaMemcpy(I.d_srcmask, I.srcmask, (size_t)I.N * sizeof(uint8_t), cudaMemcpyHostToDevice));
    } else if(!I.srcmask && I.d_srcmask){
        cudaFree(I.d_srcmask); I.d_srcmask=nullptr;
    }
}

void upload_geometry(Solver::Impl& I,
                     const uint8_t* h_tp, const float* h_pm, const float* h_inh,
                     const float* h_dvel)
{
    int N=I.N;
    CU(cudaMemcpy(I.tp,h_tp,N*sizeof(uint8_t),cudaMemcpyHostToDevice));
    CU(cudaMemcpy(I.pm,h_pm,N*sizeof(float),cudaMemcpyHostToDevice));
    CU(cudaMemcpy(I.inh,h_inh,N*sizeof(float),cudaMemcpyHostToDevice));

    // Per-cell deposition velocity (m/s) → per-link sticking probability α∈[0,1].
    // Leading-order D3Q7 closure: v_dep_lb ≈ α/8 ⇒ α = 8·v_dep_lb. (See
    // test_deposition.cpp for calibration of the realized v_d.)
    const float vfac=I.dt_phys/I.cfg.cell_size;
    const float W7_AXIAL=0.125f;
    std::vector<float> alpha(N);
    int n_clamped=0;
    for(int i=0;i<N;++i){
        float a=h_dvel[i]*vfac/W7_AXIAL;
        if(a<0.f) a=0.f;
        if(a>1.f){ a=1.f; ++n_clamped; }
        alpha[i]=a;
    }
    CU(cudaMemcpy(I.dvel,alpha.data(),N*sizeof(float),cudaMemcpyHostToDevice));
    if(n_clamped>0)
        printf("[LBM] NOTE: %d surface cells clamped to perfect absorption "
               "(deposition velocity exceeds %.3g m/s lattice ceiling)\n",
               n_clamped, W7_AXIAL/vfac);
}


void init_equilibrium(Solver::Impl& I) {
    int N=I.N;
    float hw[19]={1.f/3,
        1.f/18,1.f/18,1.f/18,1.f/18,1.f/18,1.f/18,
        1.f/36,1.f/36,1.f/36,1.f/36,1.f/36,1.f/36,
        1.f/36,1.f/36,1.f/36,1.f/36,1.f/36,1.f/36};
    std::vector<float> hf(19*N);
    for(int i=0;i<19;++i) for(int j=0;j<N;++j) hf[i*N+j]=hw[i];
    CU(cudaMemcpy(I.f[0],hf.data(),19*N*sizeof(float),cudaMemcpyHostToDevice));
    CU(cudaMemcpy(I.f[1],hf.data(),19*N*sizeof(float),cudaMemcpyHostToDevice));
    CU(cudaMemset(I.g[0],0,7*N*sizeof(float)));
    CU(cudaMemset(I.g[1],0,7*N*sizeof(float)));
    CU(cudaMemset(I.dep,0,N*sizeof(float)));   // reset deposited-mass map
    CU(cudaMemset(I.cint,0,N*sizeof(float)));  // reset time-integrated air conc.
    CU(cudaMemset(I.cmac[0],0,N*sizeof(float)));
    CU(cudaMemset(I.cmac[1],0,N*sizeof(float)));
}

void step(Solver::Impl& I, int cur, int nxt, bool with_scalar) {
    int N=I.N, nx=I.cfg.nx, ny=I.cfg.ny, nz=I.cfg.nz;
    int nblk=(N+NTHREADS-1)/NTHREADS;

    cudaStream_t s0=0;  // default stream for macroscopic
    cudaStream_t sf=(cudaStream_t)I.stream_flow;
    cudaStream_t ss=(cudaStream_t)I.stream_scalar;
    cudaEvent_t  ev=(cudaEvent_t)I.evt_macro_done;

    // K1: macroscopic (default stream)
    kern_macro<<<nblk,NTHREADS,0,s0>>>(
        I.f[cur],N,nx,ny, I.rho,I.ux,I.uy,I.uz, I.tp);
    // K1b: WALE eddy viscosity (default stream, after macro). Finalizing I.nut
    // here — before the event below — lets the concurrent flow and scalar
    // kernels both READ it without a data race.
    kern_nut<<<nblk,NTHREADS,0,s0>>>(
        I.ux,I.uy,I.uz, I.tp, I.nut, N,nx,ny,nz, I.cfg.Cw);
    CU(cudaEventRecord(ev, s0));

    // K2: flow (stream_flow, waits on macroscopic)
    CU(cudaStreamWaitEvent(sf, ev, 0));
    // Upload the host-filled ABL inlet plane (filled by fill_inlet_plane each
    // step in lbm_solver.cpp) before the flow kernel reads it; ordered on sf.
    if(I.cfg.inlet_profile==1){
        CU(cudaMemcpyAsync(I.d_inlet_plane, I.inlet_plane.data(),
            sizeof(float)*I.inlet_plane.size(), cudaMemcpyHostToDevice, sf));
        // Per-face BC (default): also upload the y-inlet plane + mean profile.
        if(I.per_face && !I.inlet_plane_y.empty())
            CU(cudaMemcpyAsync(I.d_inlet_plane_y, I.inlet_plane_y.data(),
                sizeof(float)*I.inlet_plane_y.size(), cudaMemcpyHostToDevice, sf));
        if(I.per_face && !I.mean_plane.empty())
            CU(cudaMemcpyAsync(I.d_mean_plane, I.mean_plane.data(),
                sizeof(float)*I.mean_plane.size(), cudaMemcpyHostToDevice, sf));
    }
    kern_flow<<<nblk,NTHREADS,0,sf>>>(
        I.f[cur],I.f[nxt], I.ux,I.uy,I.uz, I.tp,I.pm, I.nut,
        N,nx,ny,nz, I.nu_lb,I.cfg.Cw,
        I.uInX,I.uInY,I.uInZ,
        I.inAx,I.inSide,I.outAx,I.outSide,
        I.d_inlet_plane,I.inlet_stride,I.cfg.inlet_profile,
        I.cfg.collision_mode,I.cfg.hrr_sigma,I.cfg.wall_model,I.u_wallX,I.u_wallY,I.lateral_bc,
        I.per_face,I.face_bc[0],I.face_bc[1],I.face_bc[2],I.face_bc[3],
        I.d_inlet_plane_y,I.d_mean_plane);

    // K3: scalar (stream_scalar, waits on macroscopic, concurrent with flow).
    // Skipped during flow-only warm-up (Phase A).
    if(with_scalar){
        CU(cudaStreamWaitEvent(ss, ev, 0));
        int adv=I.cfg.scalar_advection;
        if(adv!=2)                                  // pass 1: consistent post-stream C
            kern_cfield<<<nblk,NTHREADS,0,ss>>>(I.g[cur], I.tp, I.dvel, I.cmac[0],
                N,nx,ny,nz, I.w_settle_lb);
        kern_scalar<<<nblk,NTHREADS,0,ss>>>(        // pass 2
            I.g[cur],I.g[nxt], I.ux,I.uy,I.uz, I.tp,I.pm, I.nut,
            I.dvel, I.dep, I.cint, I.cmac[0],
            N,nx,ny,nz, I.D_lb, I.cfg.Sc_t, I.srcIdx, I.cfg.Q_source*I.dt_phys, I.w_settle_lb, adv,
            I.inAx,I.inSide, I.d_srcmask);
    }

    // Sync both streams before next step
    CU(cudaStreamSynchronize(sf));
    if(with_scalar) CU(cudaStreamSynchronize(ss));
}

void save_checkpoint(Solver::Impl& I) {
    int N=I.N; size_t Nf=N*sizeof(float);
    CU(cudaMemcpy(I.ux0,I.ux,Nf,cudaMemcpyDeviceToDevice));
    CU(cudaMemcpy(I.uy0,I.uy,Nf,cudaMemcpyDeviceToDevice));
    CU(cudaMemcpy(I.uz0,I.uz,Nf,cudaMemcpyDeviceToDevice));
}

float rms_velocity_delta(Solver::Impl& I) {
    int N=I.N;
    float *ux=I.ux,*uy=I.uy,*uz=I.uz;
    float *ux0=I.ux0,*uy0=I.uy0,*uz0=I.uz0;
    uint8_t *tp=I.tp;

    double sum_sq=thrust::transform_reduce(
        thrust::counting_iterator<int>(0),
        thrust::counting_iterator<int>(N),
        [ux,uy,uz,ux0,uy0,uz0,tp] __device__(int i)->double{
            if(tp[i]==GROUND) return 0.0;
            float dx=ux[i]-ux0[i],dy=uy[i]-uy0[i],dz=uz[i]-uz0[i];
            return (double)(dx*dx+dy*dy+dz*dz);
        },
        0.0, thrust::plus<double>());

    int count=thrust::transform_reduce(
        thrust::counting_iterator<int>(0),
        thrust::counting_iterator<int>(N),
        [tp] __device__(int i)->int{ return (tp[i]!=GROUND)?1:0; },
        0, thrust::plus<int>());

    return (count>0)?sqrtf((float)(sum_sq/count)):0.f;
}

void compute_results(Solver::Impl& I, int gfin,
                     float& max_vel, float& max_conc)
{
    int N=I.N;
    // C = Σ g_k on device
    float* d_C; CU(cudaMalloc(&d_C,N*sizeof(float)));
    CU(cudaMemset(d_C,0,N*sizeof(float)));
    for(int k=0;k<7;++k)
        thrust::transform(
            thrust::device_ptr<float>(d_C),
            thrust::device_ptr<float>(d_C+N),
            thrust::device_ptr<float>(I.g[gfin]+k*N),
            thrust::device_ptr<float>(d_C),
            thrust::plus<float>());

    // max |u|
    float *ux=I.ux,*uy=I.uy,*uz=I.uz;
    max_vel=thrust::transform_reduce(
        thrust::counting_iterator<int>(0),
        thrust::counting_iterator<int>(N),
        [ux,uy,uz] __device__(int i){return sqrtf(ux[i]*ux[i]+uy[i]*uy[i]+uz[i]*uz[i]);},
        0.f, thrust::maximum<float>());

    // max C
    max_conc=thrust::reduce(
        thrust::device_ptr<float>(d_C),
        thrust::device_ptr<float>(d_C+N),
        0.f, thrust::maximum<float>());

    cudaFree(d_C);
}

void export_velocity_slice(Solver::Impl& I, int z_slice,
                           float* h_ux, float* h_uy, int nx, int ny)
{
    size_t off=(size_t)z_slice*ny*nx;
    CU(cudaMemcpy(h_ux,I.ux+off,nx*ny*sizeof(float),cudaMemcpyDeviceToHost));
    CU(cudaMemcpy(h_uy,I.uy+off,nx*ny*sizeof(float),cudaMemcpyDeviceToHost));
}

void download_mean_flow(Solver::Impl& I, float* h_ux, float* h_uy,
                        float* h_uz, float* h_nut)
{
    int N=I.N; size_t nb=(size_t)N*sizeof(float);
    if(I.n_avg>0){
        // copy the velocity sums, then scale by 1/n_avg on the host
        CU(cudaMemcpy(h_ux,I.ux_sum,nb,cudaMemcpyDeviceToHost));
        CU(cudaMemcpy(h_uy,I.uy_sum,nb,cudaMemcpyDeviceToHost));
        CU(cudaMemcpy(h_uz,I.uz_sum,nb,cudaMemcpyDeviceToHost));
        float inv=1.f/I.n_avg;
        for(int i=0;i<N;++i){ h_ux[i]*=inv; h_uy[i]*=inv; h_uz[i]*=inv; }
    } else {
        CU(cudaMemcpy(h_ux,I.ux,nb,cudaMemcpyDeviceToHost));
        CU(cudaMemcpy(h_uy,I.uy,nb,cudaMemcpyDeviceToHost));
        CU(cudaMemcpy(h_uz,I.uz,nb,cudaMemcpyDeviceToHost));
    }
    CU(cudaMemcpy(h_nut,I.nut,nb,cudaMemcpyDeviceToHost));
}

void download_density(Solver::Impl& I, float* h_rho){
    int N=I.N; size_t nb=(size_t)N*sizeof(float);
    if(I.n_avg>0){
        CU(cudaMemcpy(h_rho,I.rho_sum,nb,cudaMemcpyDeviceToHost));
        float inv=1.f/I.n_avg; for(int i=0;i<N;++i) h_rho[i]*=inv;
    } else {
        CU(cudaMemcpy(h_rho,I.rho,nb,cudaMemcpyDeviceToHost));
    }
}
void download_tiac(Solver::Impl& I, float* h_cint){   // full ∫C dt (Θ) field, device->host
    CU(cudaMemcpy(h_cint, I.cint, (size_t)I.N*sizeof(float), cudaMemcpyDeviceToHost));
}

// Instantaneous (live) velocity — for the divergence-location diagnostic (exp #5).
void download_instant_velocity(Solver::Impl& I, float* h_ux, float* h_uy, float* h_uz){
    size_t nb=(size_t)I.N*sizeof(float);
    CU(cudaMemcpy(h_ux,I.ux,nb,cudaMemcpyDeviceToHost));
    CU(cudaMemcpy(h_uy,I.uy,nb,cudaMemcpyDeviceToHost));
    CU(cudaMemcpy(h_uz,I.uz,nb,cudaMemcpyDeviceToHost));
}

// ── Particulate deposition + mass budget ────────────────────────────

double total_deposited(Solver::Impl& I) {
    return (double)thrust::reduce(
        thrust::device_ptr<float>(I.dep),
        thrust::device_ptr<float>(I.dep+I.N), 0.0f, thrust::plus<float>());
}

double total_airborne(Solver::Impl& I, int gbuf) {
    int N=I.N; double s=0;
    for(int k=0;k<7;++k)
        s += (double)thrust::reduce(
            thrust::device_ptr<float>(I.g[gbuf]+k*N),
            thrust::device_ptr<float>(I.g[gbuf]+(k+1)*N), 0.0f, thrust::plus<float>());
    return s;
}

void export_deposition_slice(Solver::Impl& I, int z_slice,
                             float* h_dep, int nx, int ny)
{
    size_t off=(size_t)z_slice*ny*nx;
    CU(cudaMemcpy(h_dep,I.dep+off,nx*ny*sizeof(float),cudaMemcpyDeviceToHost));
}

void export_tiac_slice(Solver::Impl& I, int z_slice,
                       float* h_cint, int nx, int ny)
{
    size_t off=(size_t)z_slice*ny*nx;
    CU(cudaMemcpy(h_cint,I.cint+off,nx*ny*sizeof(float),cudaMemcpyDeviceToHost));
}

void download_flow(Solver::Impl& I, int buf, float* h_f){
    CU(cudaMemcpy(h_f, I.f[buf], (size_t)19*I.N*sizeof(float), cudaMemcpyDeviceToHost));
}
void upload_flow(Solver::Impl& I, const float* h_f){
    CU(cudaMemcpy(I.f[0], h_f, (size_t)19*I.N*sizeof(float), cudaMemcpyHostToDevice));
    CU(cudaMemcpy(I.f[1], h_f, (size_t)19*I.N*sizeof(float), cudaMemcpyHostToDevice));
}

void export_scalar_slice(Solver::Impl& I, int z_slice,
                         float* h_C, int nx, int ny, int gbuf)
{
    int N=I.N; size_t off=(size_t)z_slice*ny*nx;
    std::vector<float> tmp(nx*ny);
    for(int i=0;i<nx*ny;++i) h_C[i]=0.f;
    for(int k=0;k<7;++k){
        CU(cudaMemcpy(tmp.data(),I.g[gbuf]+k*N+off,nx*ny*sizeof(float),cudaMemcpyDeviceToHost));
        for(int i=0;i<nx*ny;++i) h_C[i]+=tmp[i];
    }
}

void export_avg_scalar_slice(Solver::Impl& I, int z_slice,
                             float* h_C, int nx, int ny)
{
    size_t off=(size_t)z_slice*ny*nx;
    CU(cudaMemcpy(h_C,I.C_sum+off,nx*ny*sizeof(float),cudaMemcpyDeviceToHost));
    float inv_n=(I.n_avg>0)?1.f/I.n_avg:0.f;
    for(int i=0;i<nx*ny;++i) h_C[i]*=inv_n;
}

// ── Time-averaging ──────────────────────────────────────────────────

void reset_averages(Solver::Impl& I) {
    int N=I.N; size_t Nf=N*sizeof(float);
    CU(cudaMemset(I.ux_sum,0,Nf)); CU(cudaMemset(I.rho_sum,0,Nf));
    CU(cudaMemset(I.uy_sum,0,Nf));
    CU(cudaMemset(I.uz_sum,0,Nf));
    CU(cudaMemset(I.C_sum, 0,Nf));
    I.n_avg=0;
}

// Device kernel: accumulate velocity + concentration
__global__ void kern_accumulate(
    const float* __restrict__ ux, const float* __restrict__ uy, const float* __restrict__ uz,
    const float* __restrict__ rho, const float* __restrict__ g, int N,
    float* __restrict__ ux_sum, float* __restrict__ uy_sum, float* __restrict__ uz_sum,
    float* __restrict__ rho_sum, float* __restrict__ C_sum)
{
    int id=blockIdx.x*blockDim.x+threadIdx.x;
    if(id>=N) return;
    ux_sum[id]+=ux[id];
    uy_sum[id]+=uy[id];
    uz_sum[id]+=uz[id];
    rho_sum[id]+=rho[id];
    float C=0;
    #pragma unroll
    for(int k=0;k<7;++k) C+=g[k*N+id];
    C_sum[id]+=C;
}

void accumulate(Solver::Impl& I, int g_buf) {
    int N=I.N, nblk=(N+NTHREADS-1)/NTHREADS;
    kern_accumulate<<<nblk,NTHREADS>>>(
        I.ux,I.uy,I.uz, I.rho, I.g[g_buf], N,
        I.ux_sum,I.uy_sum,I.uz_sum, I.rho_sum, I.C_sum);
    I.n_avg++;
}

void compute_averaged_results(Solver::Impl& I,
                              float& max_vel, float& max_conc)
{
    int N=I.N;
    float inv_n=1.f/I.n_avg;
    float *uxs=I.ux_sum, *uys=I.uy_sum, *uzs=I.uz_sum, *cs=I.C_sum;

    // max |u_avg|
    max_vel=thrust::transform_reduce(
        thrust::counting_iterator<int>(0),
        thrust::counting_iterator<int>(N),
        [uxs,uys,uzs,inv_n] __device__(int i){
            float ux=uxs[i]*inv_n, uy=uys[i]*inv_n, uz=uzs[i]*inv_n;
            return sqrtf(ux*ux+uy*uy+uz*uz);
        },
        0.f, thrust::maximum<float>());

    // max C_avg
    max_conc=thrust::transform_reduce(
        thrust::counting_iterator<int>(0),
        thrust::counting_iterator<int>(N),
        [cs,inv_n] __device__(int i){return cs[i]*inv_n;},
        0.f, thrust::maximum<float>());
}

void export_avg_velocity_slice(Solver::Impl& I, int z_slice,
                               float* h_ux, float* h_uy, int nx, int ny)
{
    size_t off=(size_t)z_slice*ny*nx;
    size_t sz=nx*ny*sizeof(float);
    float inv_n=1.f/I.n_avg;
    // Copy sums to host, then normalize
    CU(cudaMemcpy(h_ux,I.ux_sum+off,sz,cudaMemcpyDeviceToHost));
    CU(cudaMemcpy(h_uy,I.uy_sum+off,sz,cudaMemcpyDeviceToHost));
    for(int i=0;i<nx*ny;++i){ h_ux[i]*=inv_n; h_uy[i]*=inv_n; }
}

}} // namespace lbm::gpu
