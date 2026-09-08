// adjoint_transport.h — concentration-level scalar transport for the
// effective-exposure objective and its EXACT discrete adjoint (reverse solver).
//
// Design (per the agreed plan, Option 1 = consistent linear scheme):
//   • First-order UPWIND advection + central diffusion + surface deposition +
//     gravitational settling, on the FROZEN mean flow (u, nut) from the LBM.
//   • The single-step propagator M is LINEAR, so its discrete adjoint Mᵀ is an
//     exact transpose — forward and reverse agree to machine precision (verified
//     by the dot-product test in adjoint_recip_test). No van Leer limiter here,
//     so there is no nonlinearity to reconcile (the production van Leer D3Q7
//     scalar remains untouched for physical-dispersion/validation runs).
//   • Both M and Mᵀ are written as pure GATHER kernels (each cell writes only its
//     own output cell), so the CPU and GPU implementations are identical in logic
//     and the GPU adjoint is race-free (no scatter / no atomics).
//
// Boundary model (wind along +x; matches the LBM scalar BCs):
//   x faces  : OPEN — advect out when flow exits, zero scalar inflow otherwise.
//   y faces  : no-flux walls.   z=top : no-flux.   z=0 : GROUND (deposition).
//   solid (building) faces: no through-flux; surface deposition removes scalar.
//
// Units are lattice units (dx=1). dt is chosen for explicit stability by the
// caller (see stable_dt). Index order matches the solver: id=(z*ny+y)*nx+x.

#pragma once
#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <cstdio>

namespace adj {

// Cell types (match lbm enum): FLUID=0, GROUND=1, SHELL/solid=2, INDOOR=3.
enum { FLUID=0, GROUND=1, SHELL=2, INDOOR=3 };
inline bool is_solid(uint8_t t){ return t!=FLUID && t!=INDOOR; } // GROUND or building

// All fields are flat arrays of length N=nx*ny*nz, row-major id=(z*ny+y)*nx+x.
struct Field {
    int nx, ny, nz, N;
    const uint8_t* tp;      // cell type
    const float* ux;        // frozen mean velocity (lattice units)
    const float* uy;
    const float* uz;
    const float* nut;       // eddy viscosity (lattice units); D_t = nut/Sc_t
    const float* vdep;      // per-cell deposition velocity (lattice units, >=0)
    float D0;               // molecular/floor scalar diffusivity (lattice units)
    float Sc_t;             // turbulent Schmidt number
    float w_s;              // gravitational settling velocity (>=0, downward -z)
    float dt;               // explicit time step (lattice units)

    inline int idx(int x,int y,int z) const { return (z*ny+y)*nx+x; }
};

// Face velocity OUTWARD from cell i along axis `ax` (0,1,2) toward sign `s`(+1/-1).
// Uses the average of the two cell velocities; settling adds a downward bias to w.
inline float face_vel(const Field& F, int i, int j, int ax, int s) {
    float ui, uj;
    if(ax==0){ ui=F.ux[i]; uj=F.ux[j]; }
    else if(ax==1){ ui=F.uy[i]; uj=F.uy[j]; }
    else { ui=F.uz[i]-F.w_s; uj=F.uz[j]-F.w_s; }   // settling: w_eff = w - w_s
    return 0.5f*(ui+uj)*(float)s;                  // component along the OUTWARD normal
}
inline float face_vel_bnd(const Field& F, int i, int ax, int s) {
    float ui = (ax==0)?F.ux[i] : (ax==1)?F.uy[i] : (F.uz[i]-F.w_s);
    return ui*(float)s;
}
inline float face_diff(const Field& F, int i, int j) {     // symmetric face diffusivity
    return F.D0 + 0.5f*(F.nut[i]+F.nut[j])/F.Sc_t;
}

// 6 face offsets and their (axis, sign): +x,-x,+y,-y,+z,-z
static const int DOFF[6][3] = {{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1}};
static const int DAXIS[6]   = {0,0,1,1,2,2};
static const int DSIGN[6]   = {+1,-1,+1,-1,+1,-1};

// Fluid neighbour index in direction d, or -1 if off-domain or solid. Used by QUICK
// to fetch the far-upstream cell (falls back to first-order upwind when it returns -1).
inline int fluid_nb(const Field& F,int x,int y,int z,int d){
    int xn=x+DOFF[d][0], yn=y+DOFF[d][1], zn=z+DOFF[d][2];
    if(xn<0||xn>=F.nx||yn<0||yn>=F.ny||zn<0||zn>=F.nz) return -1;
    int id=F.idx(xn,yn,zn);
    return (F.tp[id]==FLUID||F.tp[id]==INDOOR)? id : -1;
}

// ── Forward propagator  Cn = M · C  (gather: writes only Cn[i]) ──────────────
// M = I + dt·(−∇·(uC) + ∇·(D∇C) − deposition).  No source term (added by driver).
inline void fwd_step(const Field& F, const float* C, float* Cn) {
    const int nx=F.nx, ny=F.ny, nz=F.nz;
    const float ka = F.dt;          // dt/dx, dx=1
    #ifdef _OPENMP
    #pragma omp parallel for collapse(2) schedule(static)
    #endif
    for(int z=0; z<nz; ++z)
    for(int y=0; y<ny; ++y)
    for(int x=0; x<nx; ++x){
        int i = F.idx(x,y,z);
        if(F.tp[i]!=FLUID && F.tp[i]!=INDOOR){ Cn[i]=0.f; continue; } // no scalar in solids
        float acc = C[i];                                            // identity
        for(int d=0; d<6; ++d){
            int xn=x+DOFF[d][0], yn=y+DOFF[d][1], zn=z+DOFF[d][2];
            int ax=DAXIS[d], s=DSIGN[d];
            bool out = (xn<0||xn>=nx||yn<0||yn>=ny||zn<0||zn>=nz);
            if(!out){
                int j=F.idx(xn,yn,zn);
                if(F.tp[j]==FLUID || F.tp[j]==INDOOR){
                    float vfn = face_vel(F,i,j,ax,s);                // outward normal vel
                    // QUICK (Leonard 1979) face value — 3rd-order upwind-biased, LINEAR
                    // (so the discrete adjoint is still an exact transpose). Uses the
                    // near-upstream, downstream, and far-upstream cells; falls back to
                    // first-order upwind when the far-upstream cell is off-domain/solid.
                    float Cface;
                    if(vfn>=0.f){ int iu=fluid_nb(F,x,y,z,d^1);
                                  Cface=(iu>=0)? 0.125f*(6.f*C[i]+3.f*C[j]-C[iu]) : C[i]; }
                    else        { int ju=fluid_nb(F,xn,yn,zn,d);
                                  Cface=(ju>=0)? 0.125f*(6.f*C[j]+3.f*C[i]-C[ju]) : C[j]; }
                    acc += -ka*vfn*Cface;                            // advective out/in
                    float kd = ka*face_diff(F,i,j);
                    acc += kd*(C[j]-C[i]);                           // diffusion
                } else {
                    // solid neighbour: deposition onto that surface (no through-flux).
                    // Use the SURFACE cell's deposition velocity vdep[j]; the downward
                    // ground face (d==5) also collects the gravitational settling flux.
                    float vd = F.vdep[j] + ((d==5)? F.w_s : 0.f);
                    acc += -ka*vd*C[i];
                }
            } else {
                // domain boundary
                if(ax==0){                                           // x: OPEN
                    float vfn = face_vel_bnd(F,i,ax,s);
                    if(vfn>=0.f) acc += -ka*vfn*C[i];                // advect out; zero inflow
                }
                // y, z-top: no-flux walls → nothing
            }
        }
        Cn[i]=acc;
    }
}

// ── Adjoint propagator  Pn = Mᵀ · P  (gather: writes only Pn[i]) ─────────────
// Exact transpose of fwd_step, derived term-by-term (see REVERSE_SOLVER_NOTE.md).
inline void adj_step(const Field& F, const float* P, float* Pn) {
    const int nx=F.nx, ny=F.ny, nz=F.nz;
    const float ka = F.dt;
    #ifdef _OPENMP
    #pragma omp parallel for collapse(2) schedule(static)
    #endif
    for(int z=0; z<nz; ++z)
    for(int y=0; y<ny; ++y)
    for(int x=0; x<nx; ++x){
        int i = F.idx(x,y,z);
        if(F.tp[i]!=FLUID && F.tp[i]!=INDOOR){ Pn[i]=0.f; continue; }
        float acc = P[i];                                            // transpose of identity
        for(int d=0; d<6; ++d){
            int xn=x+DOFF[d][0], yn=y+DOFF[d][1], zn=z+DOFF[d][2];
            int ax=DAXIS[d], s=DSIGN[d];
            bool out = (xn<0||xn>=nx||yn<0||yn>=ny||zn<0||zn>=nz);
            if(!out){
                int j=F.idx(xn,yn,zn);
                if(F.tp[j]==FLUID || F.tp[j]==INDOOR){
                    float vfn = face_vel(F,i,j,ax,s);                // SAME face velocity
                    // QUICK advection transpose (exact transpose of fwd_step's flux).
                    int iu = fluid_nb(F,x,y,z,d^1);                  // i's far-upstream (U when vfn>=0)
                    int ju = fluid_nb(F,xn,yn,zn,d);                 // j's far-upstream (U when vfn<0)
                    // (A) coeff of C[i] in Cn[i]  (i's own line):
                    if(vfn>=0.f){ float cS=(iu>=0)?0.75f:1.f;   acc += -ka*vfn*cS*P[i]; }
                    else        { float cS=(ju>=0)?0.375f:0.f;  acc += -ka*vfn*cS*P[i]; }
                    // (B) coeff of C[i] in Cn[j]  (neighbour j's line):
                    if(vfn>0.f)      { float cN=(iu>=0)?0.75f:1.f;   acc += ka*vfn*cN*P[j]; }
                    else if(vfn<0.f) { float cN=(ju>=0)?0.375f:0.f;  acc += ka*vfn*cN*P[j]; }
                    // diffusion transpose (self-adjoint, symmetric): kd(P[j]-P[i])
                    float kd = ka*face_diff(F,i,j);
                    acc += kd*(P[j]-P[i]);
                } else {
                    float vd = F.vdep[j] + ((d==5)? F.w_s : 0.f);   // surface cell's vdep
                    acc += -ka*vd*P[i];                              // deposition self-loss^T
                }
            } else {
                if(ax==0){
                    float vfn = face_vel_bnd(F,i,ax,s);
                    if(vfn>=0.f) acc += -ka*vfn*P[i];                // open-boundary self-loss^T
                }
            }
        }
        // (C) far-upstream gather: transpose of QUICK's +ka·v·⅛·C[U] term. C[i] was the
        // far-upstream (U) cell of faces one step further out; gather those back.
        //   Role 3: i is the U (vfn≥0) of the face (k=nb(i,d), m=nb(k,d)).
        //   Role 4: i is the U (vfn<0) of the face (k, m=nb(i,d^1)), k=nb(m,d^1).
        // Only fires for FLUID–FLUID faces (where the forward actually used QUICK).
        for(int d=0; d<6; ++d){
            int ax=DAXIS[d], s=DSIGN[d];
            // Role 3
            int k = fluid_nb(F,x,y,z,d);
            if(k>=0){
                int xk=x+DOFF[d][0], yk=y+DOFF[d][1], zk=z+DOFF[d][2];
                int m = fluid_nb(F,xk,yk,zk,d);
                if(m>=0){ float vkm=face_vel(F,k,m,ax,s);
                          if(vkm>=0.f) acc += ka*vkm*0.125f*P[k]; }
            }
            // Role 4
            int e=d^1;
            int m4 = fluid_nb(F,x,y,z,e);
            if(m4>=0){
                int xm=x+DOFF[e][0], ym=y+DOFF[e][1], zm=z+DOFF[e][2];
                int k4 = fluid_nb(F,xm,ym,zm,e);
                if(k4>=0){ float vkm=face_vel(F,k4,m4,ax,s);   // k4→m4 is direction d
                           if(vkm<0.f) acc += ka*vkm*0.125f*P[k4]; }
            }
        }
        Pn[i]=acc;
    }
}

// A conservative explicit time step satisfying advection CFL + diffusion limits.
inline float stable_dt(const Field& F, float cfl=0.4f) {
    float umax=0.f, Dmax=0.f;
    for(int i=0;i<F.N;++i){
        if(F.tp[i]!=FLUID && F.tp[i]!=INDOOR) continue;
        float sp = std::fabs(F.ux[i])+std::fabs(F.uy[i])+std::fabs(F.uz[i])+F.w_s;
        umax = std::max(umax, sp);
        float D = F.D0 + F.nut[i]/F.Sc_t;
        Dmax = std::max(Dmax, D);
    }
    float dt_adv = (umax>0)? cfl/umax : 1e30f;
    float dt_dif = (Dmax>0)? cfl/(6.f*Dmax) : 1e30f;   // 6 faces in 3D
    return std::min({dt_adv, dt_dif, 1.0f});
}

// ── Forward objective driver ─────────────────────────────────────────────────
// Continuous source s injected each step over n_steps; accumulate TIAC=∫C dt;
// return J = Σ_x w(x)·TIAC(x).  (Used for validation / the reciprocity check.)
inline double forward_objective(const Field& F, const std::vector<float>& s,
                                const std::vector<float>& w, int n_steps,
                                std::vector<float>* TIAC_out=nullptr) {
    int N=F.N; std::vector<float> C(N,0.f), Cn(N,0.f), TIAC(N,0.f);
    for(int t=0;t<n_steps;++t){
        fwd_step(F, C.data(), Cn.data());
        for(int i=0;i<N;++i) C[i] = Cn[i] + F.dt*s[i];   // inject continuous source
        for(int i=0;i<N;++i) TIAC[i] += F.dt*C[i];       // time integration
    }
    double J=0.0; for(int i=0;i<N;++i) J += (double)w[i]*TIAC[i];
    if(TIAC_out) *TIAC_out = TIAC;
    return J;
}

// ── Reverse (footprint) driver — the production objective path ───────────────
// Exact reverse-mode adjoint of forward_objective: force with w, accumulate the
// adjoint TIAC into the footprint F_out(x) = effective exposure per unit release
// at x.  Then J = Σ_x F_out(x)·s(x) reproduces forward_objective EXACTLY.
inline void reverse_footprint(const Field& F, const std::vector<float>& w,
                              int n_steps, std::vector<float>& F_out) {
    int N=F.N; std::vector<float> Cb(N,0.f), Cbn(N,0.f), sb(N,0.f);
    for(int t=0;t<n_steps;++t){
        for(int i=0;i<N;++i) Cb[i] += F.dt*w[i];         // adjoint of TIAC += dt·C
        for(int i=0;i<N;++i) sb[i] += F.dt*Cb[i];        // adjoint of C += dt·s
        adj_step(F, Cb.data(), Cbn.data());              // adjoint of C = M·C
        std::swap(Cb, Cbn);
    }
    F_out = sb;
}

// ── Steady-state production drivers ──────────────────────────────────────────
// A CONTINUOUS source on the frozen (steady) flow drives the concentration to a
// stationary field C_ss; the production objective is the duration-INDEPENDENT
// exposure RATE  Jdot = <w, C_ss>  (exposure per unit time; lower is better).
// A steady state exists because the domain is OPEN — advective outflow at the
// streamwise faces plus surface deposition drain the source, so the transport
// operator M is a contraction (spectral radius < 1) and the fixed-point iteration
//   C <- M·C + dt·s
// converges geometrically. Convergence is declared on the objective itself
// (relative change of Jdot per `check` steps < tol), i.e. "the steady exposure
// rate has stabilised" — not a fixed wall-clock horizon.
inline double forward_steady(const Field& F, const std::vector<float>& s,
                             const std::vector<float>& w,
                             double tol=1e-4, int max_iter=200000,
                             std::vector<float>* C_out=nullptr, int* iters_out=nullptr,
                             int check=200) {
    int N=F.N; std::vector<float> C(N,0.f), Cn(N,0.f);
    double Jprev=0.0; int t=0; bool conv=false;
    for(; t<max_iter; ++t){
        fwd_step(F, C.data(), Cn.data());                // C = M·C  (QUICK advection + diffusion + deposition)
        for(int i=0;i<N;++i) C[i] = Cn[i] + F.dt*s[i];   // continuous source injection
        if((t+1)%check==0){
            double J=0.0; for(int i=0;i<N;++i) J += (double)w[i]*C[i];
            double rel = (J!=0.0)? std::fabs(J-Jprev)/std::fabs(J) : std::fabs(J-Jprev);
            Jprev=J;
            if(t+1>check && rel<tol){ conv=true; break; }
        }
    }
    double J=0.0; for(int i=0;i<N;++i) J += (double)w[i]*C[i];
    if(C_out) *C_out=C;
    if(iters_out) *iters_out=t+1;
    if(!conv) std::fprintf(stderr,"[adjoint] WARNING: steady field not converged in %d iters (rel tol %.1e)\n",t+1,tol);
    return J;                                            // Jdot = <w, C_ss>  (exposure rate)
}

// Steady adjoint: iterate  phi <- M^T·phi + dt·w  to its fixed point. The
// converged phi_ss(x) = d(Jdot)/ds(x) = the steady exposure-RATE sensitivity to a
// unit continuous source at x, and <phi_ss, s> = <w, C_ss> = Jdot exactly (steady
// reciprocity). This is the production source-attribution / gradient field.
inline void reverse_steady(const Field& F, const std::vector<float>& w,
                           std::vector<float>& phi_out,
                           double tol=1e-4, int max_iter=200000,
                           int* iters_out=nullptr, int check=200) {
    int N=F.N; std::vector<float> P(N,0.f), Pn(N,0.f);
    double nprev=0.0; int t=0; bool conv=false;
    for(; t<max_iter; ++t){
        adj_step(F, P.data(), Pn.data());                // P = M^T·P
        for(int i=0;i<N;++i) P[i] = Pn[i] + F.dt*w[i];   // constant receptor forcing
        if((t+1)%check==0){
            double n=0.0; for(int i=0;i<N;++i) n += (double)P[i]*P[i]; n=std::sqrt(n);
            double rel = (n!=0.0)? std::fabs(n-nprev)/std::fabs(n) : std::fabs(n-nprev);
            nprev=n;
            if(t+1>check && rel<tol){ conv=true; break; }
        }
    }
    phi_out=P;
    if(iters_out) *iters_out=t+1;
    if(!conv) std::fprintf(stderr,"[adjoint] WARNING: steady adjoint not converged in %d iters (rel tol %.1e)\n",t+1,tol);
}


inline double dot(const std::vector<float>& a, const std::vector<float>& b){
    double s=0.0; for(size_t i=0;i<a.size();++i) s+=(double)a[i]*b[i]; return s;
}

} // namespace adj
