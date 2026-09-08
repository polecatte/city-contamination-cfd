// ═══════════════════════════════════════════════════════════════════════
// lbm_kernels_cpu.cpp — CPU backend (drop-in replacement for lbm_kernels.cu)
//
// Same wrapper API as the CUDA version. Link against this OR lbm_kernels.cu.
//
// Build (macOS):
//   clang++ -O3 -std=c++17 -Xpreprocessor -fopenmp -I$(brew --prefix libomp)/include \
//           -c lbm_kernels_cpu.cpp -o kernels.o
//   clang++ -O3 -std=c++17 -c lbm_solver.cpp -o solver.o
//   clang++ kernels.o solver.o -L$(brew --prefix libomp)/lib -lomp -o solver
//
// Build (Linux):
//   g++ -O3 -std=c++17 -fopenmp -c lbm_kernels_cpu.cpp -o kernels.o
//   g++ -O3 -std=c++17 -c lbm_solver.cpp -o solver.o
//   g++ kernels.o solver.o -fopenmp -o solver
// ═══════════════════════════════════════════════════════════════════════

#include "lbm_gpu.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>
#include <sys/sysinfo.h>   // host RAM query for the memory pre-flight guard

#ifdef _OPENMP
#include <omp.h>
#endif

// ─── Cell types ──────────────────────────────────────────────────────
// FLUID  : open outdoor air (full LBM)
// GROUND : solid floor / wall  (full bounce-back)
// SHELL  : permeable building/park envelope (partial bounce-back, β = perm)
// INDOOR : interior air enclosed by a SHELL (normal fluid; carries inhabitance)
enum : uint8_t { FLUID=0, GROUND=1, SHELL=2, INDOOR=3 };

// ─── D3Q19 lattice (flow) ───────────────────────────────────────────
// 19 discrete velocities e_i: rest, 6 face (|e|=1), 12 edge (|e|²=2).
// CX/CY/CZ are the e_i components; OPP[i] is the index of −e_i (bounce-back).
// W19 are the equilibrium weights (Σ w_i = 1, Σ w_i e_i e_i = c_s² I, c_s²=1/3).
static constexpr int CX[19]={0, 1,-1,0,0,0,0, 1,-1,1,-1, 1,-1,1,-1, 0,0,0,0};
static constexpr int CY[19]={0, 0,0,1,-1,0,0, 1,1,-1,-1, 0,0,0,0, 1,-1,1,-1};
static constexpr int CZ[19]={0, 0,0,0,0,1,-1, 0,0,0,0, 1,1,-1,-1, 1,1,-1,-1};
static constexpr int OPP[19]={0, 2,1,4,3,6,5, 10,9,8,7, 14,13,12,11, 18,17,16,15};
// Axis-mirror maps for SPECULAR (free-slip) reflection: flip ONE component only.
// YMIR flips e_y, ZMIR flips e_z (tangential components preserved). Unlike OPP
// (which flips all three = bounce-back/no-slip), these give zero shear at the wall.
static constexpr int YMIR[19]={0,1,2,4,3,5,6,9,10,7,8,11,12,13,14,16,15,18,17};
static constexpr int ZMIR[19]={0,1,2,3,4,6,5,7,8,9,10,13,14,11,12,17,18,15,16};
static constexpr float W19[19]={
    1.f/3,
    1.f/18,1.f/18,1.f/18,1.f/18,1.f/18,1.f/18,
    1.f/36,1.f/36,1.f/36,1.f/36,1.f/36,1.f/36,
    1.f/36,1.f/36,1.f/36,1.f/36,1.f/36,1.f/36};
// MRT collision (moment space). MNRM2[a] = <m_a|m_a> are the squared norms of the
// 19 raw moment basis vectors, used to map relaxed moments back to populations:
//   f_i = Σ_a (M_ai / MNRM2[a]) m_a.  The hydrodynamic moments (ρ, momentum) are
// conserved; S_GHOST holds the fixed relaxation rates for the non-hydrodynamic
// ("ghost") moments. The shear moments instead relax at the flow rate ω(ν_eff)
// computed per cell from WALE (set inside the collision, not here).
static constexpr float MNRM2[19]={
    19,2394,252, 10,40,10,40,10,40, 36,72,12,24, 4,4,4, 8,8,8};
static constexpr float S_GHOST[19]={
    0,1.19f,1.4f, 0,1.2f, 0,1.2f, 0,1.2f,
    0,1.4f, 0,1.4f, 0,0,0, 1.98f,1.98f,1.98f};

// ─── D3Q7 lattice (scalar advection-diffusion) ──────────────────────
// 7 velocities: rest + 6 faces. Weights {1/4, 1/8×6}. GOPP[k] = −e_k.
static constexpr int EX[7]={0,1,-1,0,0,0,0};
static constexpr int EY[7]={0,0,0,1,-1,0,0};
static constexpr int EZ[7]={0,0,0,0,0,1,-1};
static constexpr int GOPP[7]={0,2,1,4,3,6,5};
static constexpr float W7[7]={.25f,.125f,.125f,.125f,.125f,.125f,.125f};

// ─── Index helper ───────────────────────────────────────────────────

inline int idx3(int x,int y,int z,int nx,int ny){ return z*ny*nx+y*nx+x; }

// ── Flux-limited (TVD) advection of the macroscopic scalar field ─────────────
// Returns ΔC = −∇·(u C) at cell (x,y,z) in conservative finite-volume form.
// mode: 0 = van Leer TVD, 1 = first-order upwind.
//   • SOLID face (neighbour GROUND/SHELL): no advective flux (LBM bounce-back
//     handles the wall).
//   • OPEN face (neighbour out of domain): advective OUTFLOW allowed (carries
//     interior C out; clean zero inflow), velocity extrapolated from the cell.
// Positivity-preserving for CFL = Σ|u_axis| < 1 (always true here, |u_lb|≈0.06).
static inline float vanleer_psi(float r){
    if(r<=0.f) return 0.f;
    return (r+r)/(1.f+r);              // 2r/(1+r), capped <2
}
// face value × face velocity, given the 4-point stencil around the face.
// mode: 0 = van Leer TVD, 1 = first-order upwind, 2 = legacy central,
//       3 = QUICK (Leonard 1979) — linear 3rd-order upstream, the LIVE-FLOW
//           FORWARD production scheme (matches the adjoint's transport operator).
// QUICK needs the far-upstream cell; where it is blocked (solid/off-domain) the
// scheme drops to first-order upwind (luValid/ruValid flags), exactly as the
// adjoint QUICK does, so the two stay consistent.
static inline float faceflux(float vf,float Cl,float Cr,float Clu,float Cru,int mode,
                             bool luValid=true, bool ruValid=true){
    float Cface;
    if(vf>=0.f){
        Cface=Cl;                       // upwind (mode 1 / fallback)
        if(mode==0){ float den=Cr-Cl; if(fabsf(den)>1e-12f) Cface=Cl+0.5f*vanleer_psi((Cl-Clu)/den)*den; }
        else if(mode==3 && luValid){ Cface = 0.75f*Cl + 0.375f*Cr - 0.125f*Clu; } // QUICK
    }else{
        Cface=Cr;
        if(mode==0){ float den=Cl-Cr; if(fabsf(den)>1e-12f) Cface=Cr+0.5f*vanleer_psi((Cr-Cru)/den)*den; }
        else if(mode==3 && ruValid){ Cface = 0.75f*Cr + 0.375f*Cl - 0.125f*Cru; } // QUICK (mirror)
    }
    return vf*Cface;
}
static inline float advect_dC(int x,int y,int z,int nx,int ny,int nz,
                              const float* C, const float* UX,const float* UY,
                              const float* UZ, const uint8_t* tp, float ws, int mode)
{
    auto inb=[&](int xx,int yy,int zz)->bool{
        return xx>=0&&xx<nx&&yy>=0&&yy<ny&&zz>=0&&zz<nz; };
    auto solid=[&](int xx,int yy,int zz)->bool{
        if(!inb(xx,yy,zz)) return false;          // out-of-domain is OPEN, not solid
        uint8_t t=tp[idx3(xx,yy,zz,nx,ny)];
        return (t==GROUND||t==SHELL); };
    auto Cat=[&](int xx,int yy,int zz)->float{
        if(!inb(xx,yy,zz)||solid(xx,yy,zz)) return 0.f;
        return C[idx3(xx,yy,zz,nx,ny)]; };
    auto Vat=[&](int xx,int yy,int zz,int ax)->float{
        if(!inb(xx,yy,zz)) return 0.f;
        int id=idx3(xx,yy,zz,nx,ny);
        return ax==0?UX[id] : ax==1?UY[id] : (UZ[id]-ws); };

    float dC=0.f, divu=0.f;             // divu accumulates the discrete ∇·u
    const int off[3][3]={{1,0,0},{0,1,0},{0,0,1}};
    for(int ax=0;ax<3;++ax){
        int dx=off[ax][0],dy=off[ax][1],dz=off[ax][2];
        float Cm2=Cat(x-2*dx,y-2*dy,z-2*dz), Cm1=Cat(x-dx,y-dy,z-dz);
        float C0 =Cat(x,y,z);
        float Cp1=Cat(x+dx,y+dy,z+dz), Cp2=Cat(x+2*dx,y+2*dy,z+2*dz);
        float V0=Vat(x,y,z,ax);
        // +1/2 face (cell ↔ +neighbour); wall faces carry zero velocity & flux
        float Fp=0.f, vfp=0.f;
        if(!solid(x+dx,y+dy,z+dz)){
            float vR = inb(x+dx,y+dy,z+dz)? Vat(x+dx,y+dy,z+dz,ax) : V0; // extrapolate at open edge
            vfp=0.5f*(V0+vR);
            // QUICK far-upstream validity: Clu=Cm1 at (x-dx), Cru=Cp2 at (x+2dx)
            bool luP = inb(x-dx,y-dy,z-dz)     && !solid(x-dx,y-dy,z-dz);
            bool ruP = inb(x+2*dx,y+2*dy,z+2*dz)&& !solid(x+2*dx,y+2*dy,z+2*dz);
            Fp=faceflux(vfp, C0,Cp1, Cm1,Cp2, mode, luP, ruP);
        }
        // −1/2 face (−neighbour ↔ cell)
        float Fm=0.f, vfm=0.f;
        if(!solid(x-dx,y-dy,z-dz)){
            float vL = inb(x-dx,y-dy,z-dz)? Vat(x-dx,y-dy,z-dz,ax) : V0;
            vfm=0.5f*(vL+V0);
            // Clu=Cm2 at (x-2dx), Cru=Cp1 at (x+dx)
            bool luM = inb(x-2*dx,y-2*dy,z-2*dz)&& !solid(x-2*dx,y-2*dy,z-2*dz);
            bool ruM = inb(x+dx,y+dy,z+dz)     && !solid(x+dx,y+dy,z+dz);
            Fm=faceflux(vfm, Cm1,C0, Cm2,Cp1, mode, luM, ruM);
        }
        dC   += (Fm-Fp);                 // conservative −∂(uC)/∂x
        divu += (vfp-vfm);               // discrete ∂u/∂x (same face velocities)
    }
    // Advective form −u·∇C = −∇·(uC) + C∇·u. Dropping the C∇·u term is correct
    // for a passive scalar in (near-)incompressible flow and removes the spurious
    // amplification from the weakly-compressible LBM velocity field.
    return dC + Cat(x,y,z)*divu;
}


// ─── MRT forward transform  m = M·f  (d'Humières D3Q19 moment basis) ──
// Maps the 19 populations to 19 moments: m[0]=ρ; m[1]=energy e; m[2]=energy²;
// m[3,5,7]=momentum jx,jy,jz; m[4,6,8]=energy flux; m[9..15]=stress tensor
// (m9,m11=normal stresses pxx,pww; m13,14,15=shear pxy,pyz,pxz); m16..18=ghost.
// The intermediate sums below are the rows of M evaluated by symmetry grouping
// (face/edge populations) to avoid a full 19×19 matrix multiply.
inline void mrt_fwd(const float f[19], float m[19])
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
    m[9] = 2.f*fxp-fyp-fzp+exy+exz-2.f*eyz;
    m[10]=-4.f*fxp+2.f*fyp+2.f*fzp+exy+exz-2.f*eyz;
    m[11]= fyp-fzp+exy-exz;
    m[12]=-2.f*fyp+2.f*fzp+exy-exz;
    m[13]=f[7]-f[8]-f[9]+f[10];
    m[14]=f[15]-f[16]-f[17]+f[18];
    m[15]=f[11]-f[12]-f[13]+f[14];
    m[16]=f[7]-f[8]+f[9]-f[10]-f[11]+f[12]-f[13]+f[14];
    m[17]=-f[7]-f[8]+f[9]+f[10]+f[15]-f[16]+f[17]-f[18];
    m[18]=f[11]+f[12]-f[13]-f[14]-f[15]-f[16]+f[17]+f[18];
}

// ─── MRT inverse transform  f = M⁻¹·m ────────────────────────────────
// M⁻¹ = Mᵀ·diag(1/<m_a|m_a>) because the moment basis is orthogonal; hence
// first rescale by MNRM2 (the squared norms) then apply Mᵀ. Reconstructs the
// post-collision populations from the relaxed moments.
inline void mrt_inv(const float m[19], float f[19])
{
    float s[19];
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

// ─── Equilibrium moments m^eq(ρ,u) ──────────────────────────────────
// The target moments toward which MRT relaxes. Conserved moments (ρ, momentum
// j=ρu) equal their actual values; the energy and stress equilibria are the
// standard quadratic-in-velocity Navier–Stokes forms; ghost equilibria are 0.
inline void meq_compute(float rho,float ux,float uy,float uz,float meq[19])
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

// ─── WALE subgrid-scale eddy viscosity (Nicoud & Ducros 1999) ────────
// ν_t = (Cw Δ)² · (Sᵈ:Sᵈ)^{3/2} / [ (S:S)^{5/2} + (Sᵈ:Sᵈ)^{5/4} ],  Δ=1 cell.
// S   = symmetric strain rate = ½(∂u_i/∂x_j + ∂u_j/∂x_i)         (→ Ssq = S:S)
// Sᵈ  = traceless symmetric part of g² where g = ∇u (g2ij below)  (→ A = Sᵈ:Sᵈ)
// The g² construction makes ν_t vanish as y³ at walls and in pure shear, which
// is why WALE is preferred over Smagorinsky for the near-building boundary layers.
inline float wale_nut(float dudx,float dudy,float dudz,
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
    float Sd00=g200-tr3,Sd11=g211-tr3,Sd22=g222-tr3;
    float Sd01=.5f*(g201+g210),Sd02=.5f*(g202+g220),Sd12=.5f*(g212+g221);
    float A=Sd00*Sd00+Sd11*Sd11+Sd22*Sd22+2.f*(Sd01*Sd01+Sd02*Sd02+Sd12*Sd12);

    const float eps=1e-30f;
    float Af=std::max(A,eps),Bf=std::max(Ssq,eps);
    float A32=Af*sqrtf(Af);
    float B52=Bf*Bf*sqrtf(Bf);
    float A54=Af*sqrtf(sqrtf(Af));
    float den=B52+A54;
    return (den<1e-20f)?0.f:Cw*Cw*A32/den;
}

// ─── Recursive-Regularized (RR) collision, Hermite/population space ──────────
// COLLISION=hrr (collision_mode==2). Reconstructs the non-equilibrium populations
// from Hermite coefficients: the 2nd-order coefficient a2 = Pi_neq (measured), and
// the 3rd-order coefficient a3 built RECURSIVELY from a2 and u (Malaspinas 2015)
// rather than measured - which removes the spurious 3rd-order modes that
// destabilise the tau->1/2 limit. Then BGK-relax: f = feq + (1-omega) f_neq_reg.
//
// D3Q19 admissibility is automatic: the inadmissible 3rd-order Hermite polynomials
// vanish on this velocity set (H3_xxx(c)=c_x(1-3cs^2)=0 since cs^2=1/3 and c^3=c;
// H3_xyz(c)=c_x c_y c_z=0 since no D3Q19 velocity has all three components), so
// only the six a_iij terms contribute and no hand-picked subset can be got wrong.
// Reconstruction weights: 2nd order 1/(2 cs^4)=4.5; the six 3rd-order terms carry
// 1/(2 cs^6)=13.5 (the 3x multiplicity of each a_iij multiset folded into 1/3!).
//
// sigma (HRR hybrid FD blend) is accepted but NOT yet applied here: this is the RR
// core (sigma=1). The FD hyperviscosity term (sigma<1) is the next increment and
// reuses the WALE velocity-gradient tensor already computed in the kernel.
inline void rr_collide(const float* fl, float r, float u, float v, float w,
                       float omega, float sigma,
                       float Sxx,float Syy,float Szz,float Sxy,float Sxz,float Syz,
                       float* fp){
    const float cs2 = 1.f/3.f;
    float usq = u*u+v*v+w*w;
    float feq[19], fneq[19];
    for(int i=0;i<19;++i){
        float cu = CX[i]*u+CY[i]*v+CZ[i]*w;
        feq[i]  = W19[i]*r*(1.f+3.f*cu+4.5f*cu*cu-1.5f*usq);
        fneq[i] = fl[i]-feq[i];
    }
    // a2_ab = sum_i c_a c_b fneq   (the -cs^2 delta term drops: sum fneq = 0).
    // This PROJECTED a2 = Pi_neq is used for the 2nd-order reconstruction, so the
    // recovered Navier-Stokes viscosity is exact (verified: stress-recovery test).
    float axx=0,ayy=0,azz=0,axy=0,axz=0,ayz=0;
    for(int i=0;i<19;++i){
        float fn=fneq[i]; float cx=CX[i],cy=CY[i],cz=CZ[i];
        axx+=cx*cx*fn; ayy+=cy*cy*fn; azz+=cz*cz*fn;
        axy+=cx*cy*fn; axz+=cx*cz*fn; ayz+=cy*cz*fn;
    }
    // HRR hybrid: the stress fed to the 3rd-order recursion is a sigma-blend of the
    // projected a2 and a FINITE-DIFFERENCE stress a2_FD = -(2 rho/(3 omega)) S
    // (the leading-order Chapman-Enskog stress-strain relation, S = strain rate
    // from central-difference velocity gradients). sigma=1 -> pure RR (no added
    // dissipation); sigma<1 -> the FD/projected mismatch at sharp gradients injects
    // a tunable hyperviscosity through the 3rd-order term only -> stability without
    // touching the 2nd-order viscosity. (Jacob, Malaspinas & Sagaut 2018.)
    float bxx=axx,byy=ayy,bzz=azz,bxy=axy,bxz=axz,byz=ayz;
    if(sigma<1.f){
        float k=-2.f*r/(3.f*omega), s1=1.f-sigma;
        bxx=sigma*axx+s1*k*Sxx; byy=sigma*ayy+s1*k*Syy; bzz=sigma*azz+s1*k*Szz;
        bxy=sigma*axy+s1*k*Sxy; bxz=sigma*axz+s1*k*Sxz; byz=sigma*ayz+s1*k*Syz;
    }
    // a3_iij recursive from the (blended) stress b:  a3_abg = u_a b_bg + u_b b_ag + u_g b_ab
    float axxy=2.f*u*bxy+v*bxx, axxz=2.f*u*bxz+w*bxx;
    float ayyx=2.f*v*bxy+u*byy, ayyz=2.f*v*byz+w*byy;
    float azzx=2.f*w*bxz+u*bzz, azzy=2.f*w*byz+v*bzz;
    for(int i=0;i<19;++i){
        float cx=CX[i],cy=CY[i],cz=CZ[i];
        // 2nd order uses PROJECTED a (viscosity); 3rd order uses BLENDED a3 (hyperviscosity).
        float H2a2 = axx*(cx*cx-cs2)+ayy*(cy*cy-cs2)+azz*(cz*cz-cs2)
                   + 2.f*(axy*cx*cy+axz*cx*cz+ayz*cy*cz);
        float H3a3 = axxy*cy*(cx*cx-cs2) + axxz*cz*(cx*cx-cs2)
                   + ayyx*cx*(cy*cy-cs2) + ayyz*cz*(cy*cy-cs2)
                   + azzx*cx*(cz*cz-cs2) + azzy*cy*(cz*cz-cs2);
        float fneq_reg = W19[i]*(4.5f*H2a2 + 13.5f*H3a3);
        fp[i] = feq[i] + (1.f-omega)*fneq_reg;
    }
}

// ═════════════════════════════════════════════════════════════════════
//  GPU WRAPPER FUNCTIONS — CPU implementation
// ═════════════════════════════════════════════════════════════════════

namespace lbm { namespace gpu {

void alloc(Solver::Impl& I) {
    int N=I.N;
    // ── Memory pre-flight (every run): abort BEFORE allocating if it won't fit.
    // System-adaptive: queries available host RAM (this is the CPU backend).
    {
        size_t need = bytes_per_cell()*(size_t)N
                    + (size_t)3*std::max(I.cfg.nx,I.cfg.ny)*I.cfg.nz*sizeof(float);
        struct sysinfo si; size_t availB=0, totalB=0;
        if (sysinfo(&si)==0){ availB=(size_t)si.freeram*si.mem_unit; totalB=(size_t)si.totalram*si.mem_unit; }
        double GB=1.0/(1024*1024*1024);
        printf("[LBM] Memory pre-flight: need %.2f GB (%zu B/cell × %.1fM cells) | "
               "free %.2f / %.2f GB host RAM\n", need*GB, bytes_per_cell(), N/1e6, availB*GB, totalB*GB);
        if (availB>0 && need > availB) {
            fprintf(stderr,
                "[LBM] FATAL: insufficient host memory — need %.2f GB but only %.2f GB free "
                "(%.2f GB total). Grid %d×%d×%d = %.1fM cells at %zu B/cell.\n"
                "       Reduce the domain (smaller OUTFLOW_H / city_m) or cell count.\n",
                need*GB, availB*GB, totalB*GB, I.cfg.nx, I.cfg.ny, I.cfg.nz, N/1e6, bytes_per_cell());
            std::fflush(stderr); std::exit(3);
        }
    }
    for(int i=0;i<2;++i){
        I.f[i]=new float[19*N]();
        I.g[i]=new float[7*N]();
    }
    I.rho=new float[N](); I.ux=new float[N](); I.uy=new float[N](); I.uz=new float[N]();
    I.ux0=new float[N](); I.uy0=new float[N](); I.uz0=new float[N]();
    I.tp=new uint8_t[N]();
    I.pm=new float[N](); I.inh=new float[N]();
    I.nut=new float[N]();
    I.dvel=new float[N](); I.dep=new float[N](); I.cint=new float[N]();
    I.cmac[0]=new float[N](); I.cmac[1]=new float[N]();
    I.warm_pending=false;
    I.ux_sum=new float[N](); I.uy_sum=new float[N](); I.rho_sum=new float[N]();
    I.uz_sum=new float[N](); I.C_sum=new float[N]();
    I.n_avg=0;
    I.stream_flow=I.stream_scalar=I.evt_macro_done=nullptr; // unused on CPU
}

void free_all(Solver::Impl& I) {
    for(int i=0;i<2;++i){ delete[] I.f[i]; delete[] I.g[i]; }
    delete[] I.rho; delete[] I.ux; delete[] I.uy; delete[] I.uz;
    delete[] I.ux0; delete[] I.uy0; delete[] I.uz0;
    delete[] I.tp; delete[] I.pm; delete[] I.inh;
    delete[] I.nut;
    delete[] I.dvel; delete[] I.dep; delete[] I.cint;
    delete[] I.cmac[0]; delete[] I.cmac[1];
    delete[] I.ux_sum; delete[] I.uy_sum; delete[] I.rho_sum;
    delete[] I.uz_sum; delete[] I.C_sum;
}

// CPU backend: the scalar kernel reads Impl::srcmask (host pointer) directly, so
// there is nothing to upload. Mirrors the CUDA gpu::sync_source_mask signature.
void sync_source_mask(Solver::Impl&) {}

void upload_geometry(Solver::Impl& I,
                     const uint8_t* h_tp, const float* h_pm, const float* h_inh,
                     const float* h_dvel)
{
    int N=I.N;
    std::memcpy(I.tp, h_tp, N*sizeof(uint8_t));
    std::memcpy(I.pm, h_pm, N*sizeof(float));
    std::memcpy(I.inh,h_inh,N*sizeof(float));

    // Convert per-cell deposition velocity (m/s) → per-link sticking
    // probability α∈[0,1]. Leading-order D3Q7 half-way bounce-back closure:
    // the wall-normal incident population ≈ w_axial·C with w_axial=1/8, so the
    // absorbed flux α·g_in ≈ (α/8)·C ⇒ v_dep_lb ≈ α/8 ⇒ α = 8·v_dep_lb.
    // (See test_deposition.cpp / analyze_deposition.py to calibrate the
    // realized v_d against this closure.)
    const float vfac = I.dt_phys / I.cfg.cell_size;   // (m/s) → LU
    const float W7_AXIAL = 0.125f;
    int n_clamped=0;
    for(int i=0;i<N;++i){
        float a = h_dvel[i]*vfac / W7_AXIAL;
        if(a<0.f) a=0.f;
        if(a>1.f){ a=1.f; ++n_clamped; }
        I.dvel[i]=a;
    }
    if(n_clamped>0)
        printf("[LBM] NOTE: %d surface cells clamped to perfect absorption "
               "(deposition velocity exceeds %.3g m/s lattice ceiling)\n",
               n_clamped, W7_AXIAL/vfac);
}


void init_equilibrium(Solver::Impl& I) {
    int N=I.N;
    for(int i=0;i<19;++i)
        for(int j=0;j<N;++j) I.f[0][i*N+j]=I.f[1][i*N+j]=W19[i];
    std::memset(I.g[0],0,7*N*sizeof(float));
    std::memset(I.g[1],0,7*N*sizeof(float));
    std::memset(I.dep,0,N*sizeof(float));   // reset deposited-mass map
    std::memset(I.cint,0,N*sizeof(float));  // reset time-integrated air conc.
    std::memset(I.cmac[0],0,N*sizeof(float));
    std::memset(I.cmac[1],0,N*sizeof(float));
}

void step(Solver::Impl& I, int cur, int nxt, bool with_scalar) {
    int N=I.N, nx=I.cfg.nx, ny=I.cfg.ny, nz=I.cfg.nz;
    float nu0=I.nu_lb, Cw=I.cfg.Cw;
    int colReg=I.cfg.collision_mode;               // 0=MRT 1=projected-reg 2=HRR/RR
    float hrrSigma=I.cfg.hrr_sigma;                // HRR hybrid blend (2 only)
    int wallM =I.cfg.wall_model;                   // 1 = rough-wall log-law floor
    float uwX=I.u_wallX, uwY=I.u_wallY;            // floor slip (lattice units)
    float uIx=I.uInX, uIy=I.uInY, uIz=I.uInZ;
    int inAx=I.inAx, inS=I.inSide, outAx=I.outAx, outS=I.outSide;
    int latBC=I.lateral_bc;           // 0=specular symmetry (default), 1=open zero-gradient (exp #6)
    int perFace=I.per_face;           // 1=per-face inflow/outflow BC (default), 0=legacy single-inlet
    float D0=I.D_lb, Sc_t=I.cfg.Sc_t;
    float Qdt=I.cfg.Q_source*I.dt_phys;
    int srcIdx=I.srcIdx;
    const uint8_t* SRCMASK=I.srcmask;   // per-cell source distribution (null ⇒ single srcIdx)
    float *NUT=I.nut;
    float ws=I.w_settle_lb;          // settling speed in LU (downward, +z is up)
    float sa=ws*8.f; if(sa>1.f) sa=1.f;  // settling sticking on upward-facing surfaces
                                          // (α=8·v_lb closure ⇒ absorbs the w_s·C flux)
    const float *DVEL=I.dvel;        // per-cell sticking probability α
    float *DEP=I.dep;                // per-cell deposited-mass accumulator
    float *CINT=I.cint;              // per-cell ∫C dt accumulator (TIAC)
    int adv=I.cfg.scalar_advection;  // 0=van Leer TVD, 1=upwind, 2=legacy central
    float *cmac0=I.cmac[0];          // scratch: current post-stream C (recomputed each step)

    const float *fs=I.f[cur]; float *fd=I.f[nxt];
    const float *gs=I.g[cur]; float *gd=I.g[nxt];
    const uint8_t *tp=I.tp; const float *pm=I.pm;
    float *RHO=I.rho, *UX=I.ux, *UY=I.uy, *UZ=I.uz;

    // ── K1: macroscopic moments ρ, u from the populations ──
    // ρ = Σ_i f_i (0th moment),  ρu = Σ_i e_i f_i (1st moment).
    // GROUND cells are solid: pin ρ=1, u=0 (no flow inside walls).
    #pragma omp parallel for schedule(static)
    for(int id=0;id<N;++id){
        if(tp[id]==GROUND){RHO[id]=1;UX[id]=UY[id]=UZ[id]=0;continue;}
        float r=0,u=0,v=0,w=0;
        for(int i=0;i<19;++i){float fi=fs[i*N+id]; r+=fi; u+=CX[i]*fi; v+=CY[i]*fi; w+=CZ[i]*fi;}
        float ri=1.f/r;
        RHO[id]=r; UX[id]=u*ri; UY[id]=v*ri; UZ[id]=w*ri;   // u = (Σ e_i f_i)/ρ
    }

    // ── K2: flow update — pull-stream · MRT-WALE collision · write ──
    #pragma omp parallel for schedule(static)
    for(int id=0;id<N;++id){
        uint8_t t=tp[id];
        if(t==GROUND){
            // Full bounce-back: reflect every population (no-slip wall, u=0).
            for(int i=0;i<19;++i) fd[i*N+id]=fs[OPP[i]*N+id];
            NUT[id]=0;
            continue;
        }
        int z=id/(ny*nx),yx=id-z*ny*nx,y=yx/nx,x=yx-y*nx;

        // Pull-streaming: gather f_i from the upstream neighbour (id − e_i).
        // If that neighbour is solid, take this cell's own opposite population
        // instead — half-way bounce-back, which places the no-slip wall midway.
        float fl[19];
        for(int i=0;i<19;++i){
            int sx=x-CX[i],sy=y-CY[i],sz=z-CZ[i];
            sx=std::max(0,std::min(nx-1,sx));
            sy=std::max(0,std::min(ny-1,sy));
            sz=std::max(0,std::min(nz-1,sz));
            int nb=idx3(sx,sy,sz,nx,ny);
            if(tp[nb]==GROUND){
                float bb=fs[OPP[i]*N+id];
                // Moving-wall bounce-back on the DOMAIN FLOOR only (sz==0): impose the
                // rough-wall log-law slip u_w. Halfway moving-wall term (Krüger et al.
                // 2017, eq. 5.26): +2 w_i ρ (e_i·u_w)/c_s² = +6 w_i (e_i·u_w) with ρ≈1.
                // sz==0 ⇒ the reflecting solid is the floor plane, so building side
                // walls (sz==z) and roofs (sz=roof≠0) keep no-slip. Only populations
                // with a horizontal component pick up slip (e_i·u_w, u_w horizontal).
                if(wallM && sz==0) bb += 6.f*W19[i]*(CX[i]*uwX + CY[i]*uwY);
                fl[i]=bb;
            } else fl[i]=fs[i*N+nb];
        }

        // Local macroscopic moments after streaming.
        float r=0,u=0,v=0,w=0;
        for(int i=0;i<19;++i){r+=fl[i];u+=CX[i]*fl[i];v+=CY[i]*fl[i];w+=CZ[i]*fl[i];}
        float ri=1.f/r; u*=ri; v*=ri; w*=ri;

        // WALE subgrid model: turbulent eddy viscosity ν_t from the velocity
        // gradient tensor (central differences). WALE (Nicoud & Ducros 1999) is
        // built from the traceless symmetric square of ∇u so that ν_t → 0 at
        // walls with the correct y³ scaling — preferred over Smagorinsky here.
        float nut=0;
        if(Cw>0 && x>0&&x<nx-1 && y>0&&y<ny-1 && z>0&&z<nz-1){
            int xp=idx3(x+1,y,z,nx,ny),xm=idx3(x-1,y,z,nx,ny);
            int yp=idx3(x,y+1,z,nx,ny),ym=idx3(x,y-1,z,nx,ny);
            int zp=idx3(x,y,z+1,nx,ny),zm=idx3(x,y,z-1,nx,ny);
            nut=wale_nut(
                .5f*(UX[xp]-UX[xm]),.5f*(UX[yp]-UX[ym]),.5f*(UX[zp]-UX[zm]),
                .5f*(UY[xp]-UY[xm]),.5f*(UY[yp]-UY[ym]),.5f*(UY[zp]-UY[zm]),
                .5f*(UZ[xp]-UZ[xm]),.5f*(UZ[yp]-UZ[ym]),.5f*(UZ[zp]-UZ[zm]),
                Cw);
        }
        // Effective viscosity ν_eff = ν_molecular + ν_t, floored for stability.
        // At 4 m, molecular ν_lb ~ 1e-8 is negligible; the floor (ν≥1e-3,
        // τ≥0.503) keeps the relaxation rate safely below the τ→0.5 singularity
        // in quiescent cells. Relaxation rate ω = 1/τ = 1/(3 ν_eff + 1/2)
        // inverts the LBM viscosity law ν = c_s²(τ − 1/2), c_s²=1/3.
        float nu_eff = nu0 + nut;
        if(nu_eff < 1e-3f) nu_eff = 1e-3f;  // tau >= 0.503
        float snu=1.f/(3.f*nu_eff+.5f);
        NUT[id] = nut;  // stored for the scalar kernel's turbulent diffusivity

        // MRT collision in moment space: m = M f, relax m toward equilibrium
        // m^eq at per-moment rates, then f' = M⁻¹ m. Shear moments (indices
        // 9,11,13,14,15) relax at ω(ν_eff); ghost moments at fixed S_GHOST.
        // MRT damps the high-order modes that make plain BGK unstable at low ν.
        float fp[19];
        if(colReg==2){
            // HRR/RR: Hermite-space recursive-regularized collision (see rr_collide).
            // For the hybrid term, strain rate from central-difference velocity
            // gradients (interior only; at boundaries pass sigma=1 => pure RR).
            float Sxx=0,Syy=0,Szz=0,Sxy=0,Sxz=0,Syz=0, sig=hrrSigma;
            if(hrrSigma<1.f && x>0&&x<nx-1 && y>0&&y<ny-1 && z>0&&z<nz-1){
                int xp=idx3(x+1,y,z,nx,ny),xm=idx3(x-1,y,z,nx,ny);
                int yp=idx3(x,y+1,z,nx,ny),ym=idx3(x,y-1,z,nx,ny);
                int zp=idx3(x,y,z+1,nx,ny),zm=idx3(x,y,z-1,nx,ny);
                float dux=.5f*(UX[xp]-UX[xm]),duy=.5f*(UX[yp]-UX[ym]),duz=.5f*(UX[zp]-UX[zm]);
                float dvx=.5f*(UY[xp]-UY[xm]),dvy=.5f*(UY[yp]-UY[ym]),dvz=.5f*(UY[zp]-UY[zm]);
                float dwx=.5f*(UZ[xp]-UZ[xm]),dwy=.5f*(UZ[yp]-UZ[ym]),dwz=.5f*(UZ[zp]-UZ[zm]);
                Sxx=dux; Syy=dvy; Szz=dwz;
                Sxy=.5f*(duy+dvx); Sxz=.5f*(duz+dwx); Syz=.5f*(dvz+dwy);
            } else sig=1.f;
            rr_collide(fl, r,u,v,w, snu, sig, Sxx,Syy,Szz,Sxy,Sxz,Syz, fp);
        }else{
            float m[19],meq[19];
            mrt_fwd(fl,m);
            meq_compute(r,u,v,w,meq);
            float sr[19];
            for(int i=0;i<19;++i) sr[i]=S_GHOST[i];
            sr[9]=sr[11]=sr[13]=sr[14]=sr[15]=snu;
            // Projected regularization (colReg==1): relax every non-hydrodynamic
            // ("ghost") moment fully to its (zero) equilibrium each step. Conserved
            // moments (0,3,5,7) and shear moments (9,11,13,14,15, at snu) are
            // untouched, so the recovered Navier-Stokes viscosity is unchanged; only
            // the aliased ghost content that destabilizes the tau->0.5 limit is
            // filtered. colReg==0 (plain MRT) leaves ghosts at S_GHOST. (Latt &
            // Chopard 2006; moment-space kin of HRR, Jacob/Malaspinas/Sagaut 2018.)
            if(colReg){ sr[1]=sr[2]=sr[4]=sr[6]=sr[8]=sr[10]=sr[12]=sr[16]=sr[17]=sr[18]=1.f; }
            for(int i=0;i<19;++i) m[i]-=sr[i]*(m[i]-meq[i]);
            mrt_inv(m,fp);
        }

        // Write. SHELL = permeable envelope: partial bounce-back — a fraction
        // σ=1−β is reflected, β=perm passes through (models a leaky facade).
        if(t==SHELL){
            float sigma=1.f-pm[id];
            for(int i=0;i<19;++i) fd[i*N+id]=sigma*fp[OPP[i]]+(1.f-sigma)*fp[i];
        }else{
            for(int i=0;i<19;++i) fd[i*N+id]=fp[i];
        }

        // Inlet BC: impose the free-stream by writing the equilibrium at the
        // inlet velocity u_in. f_i^eq = w_i ρ (1 + 3 e·u + 4.5(e·u)² − 1.5 u²).
        const int S_pl=I.inlet_stride;
        if(perFace){
            // ── Per-face inflow/outflow/symmetry (oblique-wind fix) ──────────────
            // Each vertical face is INLET/OUTFLOW/SYM (host-classified by mean normal
            // flux). Oblique wind => TWO inlet faces; their shared vertical edge is
            // written from the RFG-free MEAN (mean_plane) so it is continuous — that
            // removes the inlet-corner discontinuity that blew up. Aligned wind keeps
            // ±y as symmetry, reproducing the legacy result exactly.
            int fx=(x==0)?I.face_bc[0]:(x==nx-1)?I.face_bc[1]:-1;
            int fy=(y==0)?I.face_bc[2]:(y==ny-1)?I.face_bc[3]:-1;
            bool xin=(fx==BC_INLET), yin=(fy==BC_INLET);
            bool xout=(fx==BC_OUTFLOW), yout=(fy==BC_OUTFLOW);
            if(t!=GROUND){
                if(xin||yin){
                    float lux,luy,luz;
                    if(I.cfg.inlet_profile!=1){ lux=uIx; luy=uIy; luz=uIz; }
                    else if(xin&&yin){ lux=I.mean_plane[3*z]; luy=I.mean_plane[3*z+1]; luz=I.mean_plane[3*z+2]; }
                    else if(xin){ int k=z*S_pl+y; lux=I.inlet_plane[3*k]; luy=I.inlet_plane[3*k+1]; luz=I.inlet_plane[3*k+2]; }
                    else        { int k=z*S_pl+x; lux=I.inlet_plane_y[3*k]; luy=I.inlet_plane_y[3*k+1]; luz=I.inlet_plane_y[3*k+2]; }
                    float usq=lux*lux+luy*luy+luz*luz;
                    for(int i=0;i<19;++i){ float cu=CX[i]*lux+CY[i]*luy+CZ[i]*luz;
                        fd[i*N+id]=W19[i]*r*(1.f+3.f*cu+4.5f*cu*cu-1.5f*usq); }
                } else if(xout||yout){        // convective/zero-gradient outflow
                    int ix=x,iy=y; if(xout) ix+=(x==0)?1:-1; else iy+=(y==0)?1:-1;
                    int src=idx3(ix,iy,z,nx,ny);
                    for(int i=0;i<19;++i) fd[i*N+id]=fs[i*N+src];
                } else {                      // symmetry faces + top (+z)
                    if(z==nz-1) for(int i=0;i<19;++i){ if(CZ[i]<0) fd[i*N+id]=fd[ZMIR[i]*N+id]; }
                    if(fy==BC_SYM){
                        if(y==ny-1) for(int i=0;i<19;++i){ if(CY[i]<0) fd[i*N+id]=fd[YMIR[i]*N+id]; }
                        if(y==0)    for(int i=0;i<19;++i){ if(CY[i]>0) fd[i*N+id]=fd[YMIR[i]*N+id]; }
                    }
                    // (x-faces are never SYM for |wind|<=45 deg — no x-mirror needed.)
                }
            }
        } else {  // ===== legacy single-inlet path (WIND_BC=legacy) =====
        bool inlet=(inAx==0)?((inS==0&&x==0)||(inS==1&&x==nx-1))
                  :(inAx==1)?((inS==0&&y==0)||(inS==1&&y==ny-1)):false;
        if(inlet&&t!=GROUND){
            // Sheared turbulent ABL inlet (Richards & Hoxey 1993 mean; RFG
            // turbulence, Kraichnan 1970 / Smirnov et al. 2001): read the
            // per-cell velocity from the host-filled inlet plane. Falls back to
            // the uniform plug velocity when inlet_profile==0.
            float lux=uIx,luy=uIy,luz=uIz;
            if(I.cfg.inlet_profile==1){
                int S=I.inlet_stride;
                int kpl=(inAx==0)?(z*S+y):(z*S+x);
                lux=I.inlet_plane[3*kpl]; luy=I.inlet_plane[3*kpl+1]; luz=I.inlet_plane[3*kpl+2];
            }
            float usq=lux*lux+luy*luy+luz*luz;
            for(int i=0;i<19;++i){
                float cu=CX[i]*lux+CY[i]*luy+CZ[i]*luz;
                fd[i*N+id]=W19[i]*r*(1.f+3.f*cu+4.5f*cu*cu-1.5f*usq);
            }
        }
        // Outlet BC: zero-gradient (convective) — copy populations from the
        // neighbour one cell inside the domain, letting the flow leave cleanly.
        bool outlet=(outAx==0)?((outS==0&&x==0)||(outS==1&&x==nx-1))
                   :(outAx==1)?((outS==0&&y==0)||(outS==1&&y==ny-1)):false;
        if(outlet&&!inlet&&t!=GROUND){
            int ix=x,iy=y;
            if(outAx==0) ix+=(outS==1)?-1:1;
            if(outAx==1) iy+=(outS==1)?-1:1;
            int src=idx3(ix,iy,z,nx,ny);
            for(int i=0;i<19;++i) fd[i*N+id]=fs[i*N+src];
        }
        // ── Free-slip (symmetry) on the top (+z) and lateral (±y) boundaries ──
        // Specular reflection of the OUTWARD-normal populations only; tangential
        // momentum is preserved (zero shear, zero normal flux). On ±y this
        // supersedes the clamped self-pull above (the sy-clamp at the domain edge),
        // which recirculated tangential momentum and spuriously accelerated the
        // near-boundary flow (validation T5: |u|@boundary ≈ 2.3·U). Symmetry
        // lateral/top BCs are the COST-732 / AIJ recommendation for urban CFD
        // (Franke et al. 2007, COST 732 best-practice guideline; Tominaga et al.
        // 2008, JWEIA 96:1749); specular reflection is the standard LBM realisation
        // (Krüger et al. 2017, "The Lattice Boltzmann Method", §5.3.4).
        if(t!=GROUND&&!inlet&&!outlet){
            // Set the UNKNOWN post-stream populations (those whose upstream node is
            // outside the domain) to their axis-MIRROR partner — flips only the
            // wall-normal component, so zero normal flux + zero shear (true symmetry).
            // The previous code used OPP (flips all 3 = bounce-back) and touched the
            // KNOWN half, injecting spurious tangential momentum that accumulated into
            // an instability on large domains (blew up ~1e4 steps). Axis-mirror fixes it.
            if(z==nz-1) for(int i=0;i<19;++i){ if(CZ[i]<0) fd[i*N+id]=fd[ZMIR[i]*N+id]; } // top +z
            if(latBC==1){
                // Experiment #6: OPEN zero-gradient ±y (mirror of the outlet). Copy
                // populations from the interior y-neighbor (source buffer fs, so it
                // is race-free) — lets a mean cross-stream advect OUT of ±y instead
                // of being reflected. If this cures the oblique divergence, the
                // specular-symmetry reflection was the mechanism.
                if(y==ny-1){ int s=idx3(x,y-1,z,nx,ny); for(int i=0;i<19;++i) fd[i*N+id]=fs[i*N+s]; }
                if(y==0)   { int s=idx3(x,y+1,z,nx,ny); for(int i=0;i<19;++i) fd[i*N+id]=fs[i*N+s]; }
            } else {
                if(y==ny-1) for(int i=0;i<19;++i){ if(CY[i]<0) fd[i*N+id]=fd[YMIR[i]*N+id]; } // +y lateral
                if(y==0)    for(int i=0;i<19;++i){ if(CY[i]>0) fd[i*N+id]=fd[YMIR[i]*N+id]; } // −y lateral
            }
        }
        } // end legacy WIND_BC path
    }

    // ── K3: scalar — advection-diffusion + gravitational settling + deposition ──
    if(with_scalar){
    // Pass 1 (TVD/upwind only): compute the CURRENT post-stream macroscopic C
    // for every cell, so the explicit advection in pass 2 uses a field that is
    // consistent with the populations it modifies (no one-step lag → stable).
    if(adv!=2){
        #pragma omp parallel for schedule(static)
        for(int id=0;id<N;++id){
            uint8_t t=tp[id];
            if(t==GROUND){ cmac0[id]=0.f; continue; }
            int z=id/(ny*nx),yx=id-z*ny*nx,y=yx/nx,x=yx-y*nx;
            float C=0.f;
            for(int k=0;k<7;++k){
                int sx=x-EX[k],sy=y-EY[k],sz=z-EZ[k];
                if(sx<0||sx>=nx||sy<0||sy>=ny||sz<0||sz>=nz) continue;  // clean far-field
                int nb=idx3(sx,sy,sz,nx,ny);
                if(tp[nb]==GROUND){
                    float a=DVEL[nb]; if(EZ[k]>0) a+=sa; if(a>1.f) a=1.f;
                    C+=(1.f-a)*gs[GOPP[k]*N+id];        // reflected fraction stays airborne
                }else{
                    C+=gs[k*N+nb];
                }
            }
            cmac0[id]=C;
        }
    }
    #pragma omp parallel for schedule(static)
    for(int id=0;id<N;++id){
        uint8_t t=tp[id];
        if(t==GROUND){
            // Bookkeeping write (ground populations are never pulled; the
            // absorbing bounce-back is applied at the adjacent fluid cell).
            for(int k=0;k<7;++k) gd[k*N+id]=gs[GOPP[k]*N+id];
            continue;
        }
        int z=id/(ny*nx),yx=id-z*ny*nx,y=yx/nx,x=yx-y*nx;

        // Pull. At a GROUND link, reflect only the (1-α) fraction and bank the
        // absorbed αg_in as deposited mass on the ground surface below/beside.
        // Out-of-domain links use a clean far-field (zero inflow): scalar that
        // leaves simply leaves (outflow), and nothing advects in from outside.
        // NOTE: this replaces the previous clamped self-pull, which duplicated
        // boundary populations and made the scalar field non-conservative.
        float gl[7];
        float dep_here=0.f;
        for(int k=0;k<7;++k){
            int sx=x-EX[k],sy=y-EY[k],sz=z-EZ[k];
            if(sx<0||sx>=nx||sy<0||sy>=ny||sz<0||sz>=nz){
                gl[k]=0.f;                    // clean far-field
                continue;
            }
            int nb=idx3(sx,sy,sz,nx,ny);
            if(tp[nb]==GROUND){
                float gin=gs[GOPP[k]*N+id];   // population streaming into the wall
                float a=DVEL[nb];             // ground surface sticking prob
                if(EZ[k]>0) a+=sa;           // wall is below → settling flux deposits here
                if(a>1.f) a=1.f;
                gl[k]=(1.f-a)*gin;            // reflected back into the fluid
                dep_here+=a*gin;              // captured on the ground
            }else{
                gl[k]=gs[k*N+nb];
            }
        }
        float C=0; for(int k=0;k<7;++k) C+=gl[k];
        CINT[id]+=C;   // time-integrate air concentration (Phase B only)
        float u=UX[id],v=UY[id],w=UZ[id];

        // Per-cell effective diffusivity: molecular/floor (D0=I.D_lb, set by D_FLOOR)
        // + turbulent (nut/Sc_t). D_eff ≥ D0 automatically, so no separate hardcoded
        // floor (a hardcoded 1e-3 here would override D_FLOOR — the GPU path already
        // dropped it; matched here so CPU tests honour the same floor).
        float D_eff = D0 + NUT[id] / Sc_t;

        float gp[7];
        if(adv==2){
            // ── Legacy: TRT with central advection in the equilibrium (c_s²=1/3).
            //    Accurate in smooth flow but oscillatory at high cell-Péclet. ──
            float inv_tau_c = 1.f / (3.f * D_eff + 0.5f);
            float wadv = w - ws;
            float geq[7];
            for(int k=0;k<7;++k){
                float cu=EX[k]*u+EY[k]*v+EZ[k]*wadv;
                geq[k]=W7[k]*C*(1.f+3.f*cu);
            }
            float w_minus = inv_tau_c;
            float w_plus  = 1.f / (0.5f + 1.f/(12.f*D_eff));
            for(int k=0;k<7;++k){
                int ko=GOPP[k];
                float g_s = 0.5f*(gl[k]+gl[ko]),  g_a = 0.5f*(gl[k]-gl[ko]);
                float e_s = 0.5f*(geq[k]+geq[ko]), e_a = 0.5f*(geq[k]-geq[ko]);
                gp[k]=gl[k] - w_plus*(g_s-e_s) - w_minus*(g_a-e_a);
            }
        }else{
            // ── Operator split: TRT diffusion (isotropic equilibrium, correct
            //    c_s²=1/4) + explicit positivity-preserving advection (van Leer
            //    TVD or upwind). Removes the high-cell-Péclet oscillation and
            //    fixes the c_s² advection-normalization mismatch.
            //    SEQUENTIAL split (advect the streamed field FIRST, then diffuse
            //    it): D∘A is stable, whereas additive D[C]+A[C]−C is not. ──
            if(t==FLUID || t==INDOOR){
                float dCadv = advect_dC(x,y,z,nx,ny,nz, cmac0, UX,UY,UZ, tp, ws, adv);
                for(int k=0;k<7;++k) gl[k]+=W7[k]*dCadv;
                C += dCadv;
            }
            float inv_tau_c = 1.f / (4.f * D_eff + 0.5f);          // c_s²=1/4 sets D
            float w_minus = inv_tau_c;
            float w_plus  = 1.f / (0.5f + 1.f/(16.f*D_eff));       // Λ=1/4
            for(int k=0;k<7;++k){
                int ko=GOPP[k];
                float g_s = 0.5f*(gl[k]+gl[ko]);                   // isotropic eq:
                float g_a = 0.5f*(gl[k]-gl[ko]);                   //  e_s=W7·C, e_a=0
                float e_s = 0.5f*(W7[k]+W7[ko])*C;
                gp[k]=gl[k] - w_plus*(g_s-e_s) - w_minus*g_a;
            }
        }
        // Source injection: a per-cell MASK (source distribution over Ω) if set,
        // else the single srcIdx cell. Mask present ⇒ mask governs entirely, so
        // the null-mask path is exactly the original `id==srcIdx` behavior.
        if(SRCMASK ? (SRCMASK[id]!=0) : (id==srcIdx))
            for(int k=0;k<7;++k) gp[k]+=Qdt*W7[k];

        if(t==SHELL){
            float sigma=1.f-pm[id];   // reflected (vs transmitted) fraction
            for(int k=0;k<7;++k){
                float a=DVEL[id];     // envelope sticking probability
                if(EZ[k]>0) a+=sa;   // upward-facing envelope (roof) catches settling
                if(a>1.f) a=1.f;
                float refl=sigma*gp[GOPP[k]];          // would bounce back
                gd[k*N+id]=(1.f-a)*refl+(1.f-sigma)*gp[k];
                dep_here+=a*refl;                      // captured on the envelope
            }
        }else{
            for(int k=0;k<7;++k) gd[k*N+id]=gp[k];
        }
        if(dep_here>0.f) DEP[id]+=dep_here;

        bool inlet=(inAx==0)?((inS==0&&x==0)||(inS==1&&x==nx-1))
                  :(inAx==1)?((inS==0&&y==0)||(inS==1&&y==ny-1)):false;
        if(inlet&&t!=GROUND)
            for(int k=0;k<7;++k) gd[k*N+id]=0;
    }
    } // if(with_scalar)
}

void save_checkpoint(Solver::Impl& I) {
    int N=I.N; size_t Nf=N*sizeof(float);
    std::memcpy(I.ux0,I.ux,Nf);
    std::memcpy(I.uy0,I.uy,Nf);
    std::memcpy(I.uz0,I.uz,Nf);
}

float rms_velocity_delta(Solver::Impl& I) {
    int N=I.N;
    double sum_sq=0;
    int count=0;
    #pragma omp parallel for reduction(+:sum_sq,count)
    for(int i=0;i<N;++i){
        if(I.tp[i]==GROUND) continue;
        float dx=I.ux[i]-I.ux0[i];
        float dy=I.uy[i]-I.uy0[i];
        float dz=I.uz[i]-I.uz0[i];
        sum_sq+=dx*dx+dy*dy+dz*dz;
        count++;
    }
    return (count>0)?sqrtf((float)(sum_sq/count)):0.f;
}

void compute_results(Solver::Impl& I, int gfin,
                     float& max_vel, float& max_conc)
{
    int N=I.N;

    // C = Σ g_k
    std::vector<float> C(N,0.f);
    for(int k=0;k<7;++k)
        for(int j=0;j<N;++j) C[j]+=I.g[gfin][k*N+j];

    // max |u|
    float mv=0;
    #pragma omp parallel for reduction(max:mv)
    for(int i=0;i<N;++i){
        float s=std::sqrt(I.ux[i]*I.ux[i]+I.uy[i]*I.uy[i]+I.uz[i]*I.uz[i]);
        mv=std::max(mv,s);
    }
    max_vel=mv;

    // max C
    float mc=0;
    #pragma omp parallel for reduction(max:mc)
    for(int i=0;i<N;++i) mc=std::max(mc,C[i]);
    max_conc=mc;
}

void export_velocity_slice(Solver::Impl& I, int z_slice,
                           float* h_ux, float* h_uy, int nx, int ny)
{
    for(int y=0;y<ny;++y)
        for(int x=0;x<nx;++x){
            int id=idx3(x,y,z_slice,nx,ny);
            h_ux[y*nx+x]=I.ux[id];
            h_uy[y*nx+x]=I.uy[id];
        }
}

void download_mean_flow(Solver::Impl& I, float* h_ux, float* h_uy,
                        float* h_uz, float* h_nut)
{
    int N=I.N;
    if(I.n_avg>0){
        float inv=1.f/I.n_avg;
        for(int i=0;i<N;++i){ h_ux[i]=I.ux_sum[i]*inv; h_uy[i]=I.uy_sum[i]*inv;
                              h_uz[i]=I.uz_sum[i]*inv; }
    } else {
        for(int i=0;i<N;++i){ h_ux[i]=I.ux[i]; h_uy[i]=I.uy[i]; h_uz[i]=I.uz[i]; }
    }
    for(int i=0;i<N;++i) h_nut[i]=I.nut[i];
}

void download_density(Solver::Impl& I, float* h_rho){
    int N=I.N;
    if(I.n_avg>0){ float inv=1.f/I.n_avg; for(int i=0;i<N;++i) h_rho[i]=I.rho_sum[i]*inv; }
    else         { for(int i=0;i<N;++i) h_rho[i]=I.rho[i]; }
}

// Instantaneous (live) velocity — for the divergence-location diagnostic (exp #5).
void download_instant_velocity(Solver::Impl& I, float* h_ux, float* h_uy, float* h_uz){
    const size_t Nf=(size_t)I.N*sizeof(float);
    std::memcpy(h_ux,I.ux,Nf); std::memcpy(h_uy,I.uy,Nf); std::memcpy(h_uz,I.uz,Nf);
}

// ── Particulate deposition + mass budget ────────────────────────────

double total_deposited(Solver::Impl& I) {
    int N=I.N; double s=0;
    #pragma omp parallel for reduction(+:s)
    for(int i=0;i<N;++i) s+=I.dep[i];
    return s;
}

double total_airborne(Solver::Impl& I, int gbuf) {
    int N=I.N; const float* gs=I.g[gbuf]; double s=0;
    #pragma omp parallel for reduction(+:s)
    for(int i=0;i<N;++i){
        double C=0; for(int k=0;k<7;++k) C+=gs[k*N+i];
        s+=C;
    }
    return s;
}

void export_deposition_slice(Solver::Impl& I, int z_slice,
                             float* h_dep, int nx, int ny)
{
    for(int y=0;y<ny;++y)
        for(int x=0;x<nx;++x){
            int id=idx3(x,y,z_slice,nx,ny);
            h_dep[y*nx+x]=I.dep[id];
        }
}

void download_tiac(Solver::Impl& I, float* h_cint){
    std::memcpy(h_cint, I.cint, (size_t)I.N*sizeof(float));   // full ∫C dt field
}
void export_tiac_slice(Solver::Impl& I, int z_slice,
                       float* h_cint, int nx, int ny)
{
    for(int y=0;y<ny;++y)
        for(int x=0;x<nx;++x){
            int id=idx3(x,y,z_slice,nx,ny);
            h_cint[y*nx+x]=I.cint[id];
        }
}

void download_flow(Solver::Impl& I, int buf, float* h_f){
    std::memcpy(h_f, I.f[buf], (size_t)19*I.N*sizeof(float));
}
void upload_flow(Solver::Impl& I, const float* h_f){
    std::memcpy(I.f[0], h_f, (size_t)19*I.N*sizeof(float));
    std::memcpy(I.f[1], h_f, (size_t)19*I.N*sizeof(float));
}

void export_scalar_slice(Solver::Impl& I, int z_slice,
                         float* h_C, int nx, int ny, int gbuf)
{
    int N=I.N; const float* gs=I.g[gbuf];
    for(int y=0;y<ny;++y)
        for(int x=0;x<nx;++x){
            int id=idx3(x,y,z_slice,nx,ny);
            float C=0; for(int k=0;k<7;++k) C+=gs[k*N+id];
            h_C[y*nx+x]=C;
        }
}

void export_avg_scalar_slice(Solver::Impl& I, int z_slice,
                             float* h_C, int nx, int ny)
{
    float inv_n = (I.n_avg>0)?1.f/I.n_avg:0.f;
    for(int y=0;y<ny;++y)
        for(int x=0;x<nx;++x){
            int id=idx3(x,y,z_slice,nx,ny);
            h_C[y*nx+x]=I.C_sum[id]*inv_n;
        }
}

// ── Time-averaging ──────────────────────────────────────────────────

void reset_averages(Solver::Impl& I) {
    int N=I.N;
    std::memset(I.ux_sum,0,N*sizeof(float));
    std::memset(I.uy_sum,0,N*sizeof(float));
    std::memset(I.uz_sum,0,N*sizeof(float));
    std::memset(I.C_sum, 0,N*sizeof(float));
    std::memset(I.rho_sum,0,N*sizeof(float));
    I.n_avg=0;
}

void accumulate(Solver::Impl& I, int g_buf) {
    int N=I.N;
    // Velocity (from K1 macroscopic — one step lagged, fine for averaging)
    #pragma omp parallel for schedule(static)
    for(int i=0;i<N;++i){
        I.ux_sum[i]+=I.ux[i];
        I.uy_sum[i]+=I.uy[i];
        I.uz_sum[i]+=I.uz[i];
        I.rho_sum[i]+=I.rho[i];
    }
    // Concentration: C = Σ g_k
    const float* gs=I.g[g_buf];
    #pragma omp parallel for schedule(static)
    for(int i=0;i<N;++i){
        float C=0;
        for(int k=0;k<7;++k) C+=gs[k*N+i];
        I.C_sum[i]+=C;
    }
    I.n_avg++;
}

void compute_averaged_results(Solver::Impl& I,
                              float& max_vel, float& max_conc)
{
    int N=I.N;
    float inv_n=1.f/I.n_avg;

    // max |u_avg|
    float mv=0;
    #pragma omp parallel for reduction(max:mv)
    for(int i=0;i<N;++i){
        float ux=I.ux_sum[i]*inv_n, uy=I.uy_sum[i]*inv_n, uz=I.uz_sum[i]*inv_n;
        float s=std::sqrt(ux*ux+uy*uy+uz*uz);
        mv=std::max(mv,s);
    }
    max_vel=mv;

    // max C_avg
    float mc=0;
    #pragma omp parallel for reduction(max:mc)
    for(int i=0;i<N;++i)
        mc=std::max(mc,I.C_sum[i]*inv_n);
    max_conc=mc;
}

void export_avg_velocity_slice(Solver::Impl& I, int z_slice,
                               float* h_ux, float* h_uy, int nx, int ny)
{
    float inv_n=1.f/I.n_avg;
    for(int y=0;y<ny;++y)
        for(int x=0;x<nx;++x){
            int id=idx3(x,y,z_slice,nx,ny);
            h_ux[y*nx+x]=I.ux_sum[id]*inv_n;
            h_uy[y*nx+x]=I.uy_sum[id]*inv_n;
        }
}

}} // namespace lbm::gpu
