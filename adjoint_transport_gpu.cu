// adjoint_transport_gpu.cu — CUDA mirror of adjoint_transport.h.
//
// Same gather-form forward M and adjoint Mᵀ as the CPU header, one CUDA thread
// per cell. Because both kernels are pure gathers (each thread writes only its
// own cell), the adjoint is race-free — no atomics, no scatter. The per-cell
// math is identical to the CPU version, so CPU and GPU agree bit-for-bit up to
// float reduction order; the reciprocity test guards the transpose.
//
// UNTESTED ON HARDWARE: no nvcc in the dev environment. Device-code syntax was
// checked with CUDA keywords neutralized. Rebuild on the target GPU and run the
// reciprocity check (adjoint_recip_test logic) before production use.

#include <cuda_runtime.h>
#include <vector>
#include <cstdint>
#include <cmath>
#include <cstdio>

namespace adjgpu {

enum { FLUID=0, GROUND=1, SHELL=2, INDOOR=3 };
#define CUA(x) do{ cudaError_t e=(x); if(e){ printf("CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); } }while(0)

__constant__ int c_off[6][3] = {{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1}};
__constant__ int c_axis[6]   = {0,0,1,1,2,2};
__constant__ int c_sign[6]   = {1,-1,1,-1,1,-1};

struct GField {
    int nx,ny,nz,N;
    const uint8_t* tp;
    const float *ux,*uy,*uz,*nut,*vdep;
    float D0, Sc_t, w_s, dt;
};

__device__ __forceinline__ int gidx(const GField& F,int x,int y,int z){ return (z*F.ny+y)*F.nx+x; }
__device__ __forceinline__ float fvel(const GField& F,int i,int j,int ax,int s){
    float ui,uj;
    if(ax==0){ ui=F.ux[i]; uj=F.ux[j]; }
    else if(ax==1){ ui=F.uy[i]; uj=F.uy[j]; }
    else { ui=F.uz[i]-F.w_s; uj=F.uz[j]-F.w_s; }
    return 0.5f*(ui+uj)*(float)s;
}
__device__ __forceinline__ float fvelb(const GField& F,int i,int ax,int s){
    float ui = (ax==0)?F.ux[i] : (ax==1)?F.uy[i] : (F.uz[i]-F.w_s);
    return ui*(float)s;
}
__device__ __forceinline__ float fdiff(const GField& F,int i,int j){
    return F.D0 + 0.5f*(F.nut[i]+F.nut[j])/F.Sc_t;
}

// Forward propagator Cn = M·C (gather; mirrors adj::fwd_step exactly)
__global__ void kern_fwd_step(GField F, const float* __restrict__ C, float* __restrict__ Cn){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i>=F.N) return;
    uint8_t ti=F.tp[i];
    if(ti!=FLUID && ti!=INDOOR){ Cn[i]=0.f; return; }
    int z=i/(F.ny*F.nx), r=i-z*F.ny*F.nx, y=r/F.nx, x=r-y*F.nx;
    const float ka=F.dt;
    float acc=C[i];
    #pragma unroll
    for(int d=0; d<6; ++d){
        int xn=x+c_off[d][0], yn=y+c_off[d][1], zn=z+c_off[d][2];
        int ax=c_axis[d], s=c_sign[d];
        bool out=(xn<0||xn>=F.nx||yn<0||yn>=F.ny||zn<0||zn>=F.nz);
        if(!out){
            int j=gidx(F,xn,yn,zn); uint8_t tj=F.tp[j];
            if(tj==FLUID || tj==INDOOR){
                float vfn=fvel(F,i,j,ax,s);
                float Cup=(vfn>=0.f)?C[i]:C[j];
                acc += -ka*vfn*Cup;
                float kd=ka*fdiff(F,i,j);
                acc += kd*(C[j]-C[i]);
            } else {
                float vd=F.vdep[j] + ((d==5)?F.w_s:0.f);   // surface cell's deposition vel
                acc += -ka*vd*C[i];
            }
        } else if(ax==0){
            float vfn=fvelb(F,i,ax,s);
            if(vfn>=0.f) acc += -ka*vfn*C[i];
        }
    }
    Cn[i]=acc;
}

// Adjoint propagator Pn = Mᵀ·P (gather; mirrors adj::adj_step exactly)
__global__ void kern_adj_step(GField F, const float* __restrict__ P, float* __restrict__ Pn){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i>=F.N) return;
    uint8_t ti=F.tp[i];
    if(ti!=FLUID && ti!=INDOOR){ Pn[i]=0.f; return; }
    int z=i/(F.ny*F.nx), r=i-z*F.ny*F.nx, y=r/F.nx, x=r-y*F.nx;
    const float ka=F.dt;
    float acc=P[i];
    #pragma unroll
    for(int d=0; d<6; ++d){
        int xn=x+c_off[d][0], yn=y+c_off[d][1], zn=z+c_off[d][2];
        int ax=c_axis[d], s=c_sign[d];
        bool out=(xn<0||xn>=F.nx||yn<0||yn>=F.ny||zn<0||zn>=F.nz);
        if(!out){
            int j=gidx(F,xn,yn,zn); uint8_t tj=F.tp[j];
            if(tj==FLUID || tj==INDOOR){
                float vfn=fvel(F,i,j,ax,s);
                if(vfn>=0.f){ acc += -ka*vfn*P[i]; acc += ka*vfn*P[j]; }
                float kd=ka*fdiff(F,i,j);
                acc += kd*(P[j]-P[i]);
            } else {
                float vd=F.vdep[j] + ((d==5)?F.w_s:0.f);   // surface cell's deposition vel
                acc += -ka*vd*P[i];
            }
        } else if(ax==0){
            float vfn=fvelb(F,i,ax,s);
            if(vfn>=0.f) acc += -ka*vfn*P[i];
        }
    }
    Pn[i]=acc;
}

// fused elementwise helpers
__global__ void kern_axpy(float* a, const float* b, float c, int N){    // a += c*b
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<N) a[i]+=c*b[i];
}
__global__ void kern_inject(float* C, const float* Cn, const float* s, float dt, int N){ // C = Cn + dt*s
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<N) C[i]=Cn[i]+dt*s[i];
}

// ── device drivers (device pointers in; manage scratch internally) ──
// Reverse footprint: force with w over n_steps → F_out (device). Exact reverse-
// mode adjoint of forward_objective (mirrors adj::reverse_footprint).
void reverse_footprint(const GField& F, const float* d_w, int n_steps, float* d_Fout){
    int N=F.N, nb=(N+255)/256;
    float *Cb,*Cbn,*sb;
    CUA(cudaMalloc(&Cb ,N*sizeof(float))); CUA(cudaMemset(Cb ,0,N*sizeof(float)));
    CUA(cudaMalloc(&Cbn,N*sizeof(float)));
    CUA(cudaMalloc(&sb ,N*sizeof(float))); CUA(cudaMemset(sb ,0,N*sizeof(float)));
    for(int t=0;t<n_steps;++t){
        kern_axpy<<<nb,256>>>(Cb, d_w, F.dt, N);   // Cb += dt*w
        kern_axpy<<<nb,256>>>(sb, Cb , F.dt, N);   // sb += dt*Cb
        kern_adj_step<<<nb,256>>>(F, Cb, Cbn);     // Cb = Mᵀ Cb
        float* tmp=Cb; Cb=Cbn; Cbn=tmp;
    }
    CUA(cudaMemcpy(d_Fout, sb, N*sizeof(float), cudaMemcpyDeviceToDevice));
    cudaFree(Cb); cudaFree(Cbn); cudaFree(sb);
}

// Forward objective accumulator: returns TIAC (device) for the reciprocity check.
void forward_tiac(const GField& F, const float* d_s, int n_steps, float* d_TIAC){
    int N=F.N, nb=(N+255)/256;
    float *C,*Cn;
    CUA(cudaMalloc(&C ,N*sizeof(float))); CUA(cudaMemset(C,0,N*sizeof(float)));
    CUA(cudaMalloc(&Cn,N*sizeof(float)));
    CUA(cudaMemset(d_TIAC,0,N*sizeof(float)));
    for(int t=0;t<n_steps;++t){
        kern_fwd_step<<<nb,256>>>(F, C, Cn);       // Cn = M C
        kern_inject  <<<nb,256>>>(C, Cn, d_s, F.dt, N); // C = Cn + dt*s
        kern_axpy    <<<nb,256>>>(d_TIAC, C, F.dt, N);  // TIAC += dt*C
    }
    cudaFree(C); cudaFree(Cn);
}

} // namespace adjgpu
