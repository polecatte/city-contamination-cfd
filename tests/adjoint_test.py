"""adjoint_test.py — dual-consistency (adjoint) verification for our scalar
transport scheme, on a city-like geometry with a prescribed flow field.

Validates that a HAND-WRITTEN discrete adjoint of the transport operator is the
exact transpose of the forward operator, via:
  (A) the dot-product (adjoint) test:  <M u, v> == <u, Mᵀ v>  for random u,v
      — the gold-standard adjoint check; passes to machine precision iff the
        hand-coded adjoint is the true transpose;
  (B) source-receptor reciprocity:  C(r; s) == C*(s; r)
      — the physical statement: forward concentration at receptor r from a unit
        release at s equals the adjoint field at s from a unit adjoint-source at r.
      This is the identity the exposure-footprint approach relies on (Marchuk duality;
      Pudykiewicz 1998; Keats et al. 2007).

The forward step replicates our production transport in its LINEAR form: D3Q7-style
first-order upwind advection + central diffusion (D_eff), surface deposition at
solid faces, gravitational settling onto downward (ground) faces, zero-inflow
inlet, open outlet, no-flux lateral/top. (Production uses a van Leer limiter,
which is nonlinear — see note printed at the end.)

Geometry mimics the city-builder architecture: a ground plane (z=0) plus solid
building blocks (SHELL) carrying per-face deposition; everything else fluid.
"""
import numpy as np

# ── grid + geometry (small, city-like) ──────────────────────────────────────
nx, ny, nz = 28, 20, 12
N = nx*ny*nz
def idx(x,y,z): return (z*ny + y)*nx + x
FLUID, GROUND, SOLID = 0, 1, 2

tp = np.full(N, FLUID, np.int8)
for x in range(nx):
    for y in range(ny):
        tp[idx(x,y,0)] = GROUND                       # ground plane
blocks = [(6,6,5,5,5),(16,5,4,6,7),(11,12,5,4,4),(20,12,4,5,6)]  # x0,y0,w,d,h
for (x0,y0,w,d,h) in blocks:
    for x in range(x0,x0+w):
        for y in range(y0,y0+d):
            for z in range(1,1+h):
                if x<nx and y<ny and z<nz: tp[idx(x,y,z)] = SOLID
fluid = (tp==FLUID)

# ── prescribed steady flow (sheared +x) + settling ──────────────────────────
U, Dco, w_s = 0.06, 0.02, 0.006
ux = np.zeros(N); uy = np.zeros(N); uz = np.zeros(N)
for z in range(nz):
    for y in range(ny):
        for x in range(nx):
            if tp[idx(x,y,z)]==FLUID:
                ux[idx(x,y,z)] = U*(z/(nz-1))          # log-ish shear proxy
dep = 0.02                                              # sticking fraction per solid face
dt, dx = 1.0, 1.0; ka = dt/dx; kd = dt/(dx*dx)*Dco

NB = [(+1,0,0),(-1,0,0),(0,+1,0),(0,-1,0),(0,0,+1),(0,0,-1)]
def vface(i, j, ax):                                   # face velocity, axis ax (0,1,2)
    if ax==0: vf = 0.5*(ux[i]+ux[j])
    elif ax==1: vf = 0.5*(uy[i]+uy[j])
    else: vf = 0.5*(uz[i]+uz[j]) - w_s                 # settling biases z downward
    return vf

def fwd_step(C):
    Cn = C.copy()                                       # identity: Cn[i] += C[i]
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                i = idx(x,y,z)
                if tp[i]!=FLUID: continue
                for d,(dxv,dyv,dzv) in enumerate(NB):
                    ax = d//2
                    xn,yn,zn = x+dxv, y+dyv, z+dzv
                    out = not(0<=xn<nx and 0<=yn<ny and 0<=zn<nz)
                    # internal fluid-fluid faces: process only +dirs (each face once)
                    if not out and tp[idx(xn,yn,zn)]==FLUID:
                        if d%2==1: continue
                        j = idx(xn,yn,zn); vf = vface(i,j,ax)
                        if vf>=0: Cn[i]-=ka*vf*C[i]; Cn[j]+=ka*vf*C[i]
                        else:     Cn[i]-=ka*vf*C[j]; Cn[j]+=ka*vf*C[j]
                        Cn[i]+=kd*(C[j]-C[i]); Cn[j]+=kd*(C[i]-C[j])
                    elif not out and tp[idx(xn,yn,zn)]!=FLUID:   # fluid-solid: deposit
                        dcoef = dep + (w_s if d==5 else 0.0)     # +settling onto ground below
                        Cn[i]-=dcoef*C[i]
                    else:                                # domain boundary face
                        if ax==0 and xn<0:               # inlet (x=0): zero inflow
                            vf=ux[i]
                            if vf<0: Cn[i]-=ka*vf*C[i]    # only outward advection acts
                        elif ax==0 and xn>=nx:           # outlet: open (zero-gradient)
                            vf=ux[i]
                            if vf>=0: Cn[i]-=ka*vf*C[i]
                        # lateral/top: no-flux (nothing)
    return Cn

def adj_step(P):
    Pn = P.copy()                                       # transpose of identity
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                i = idx(x,y,z)
                if tp[i]!=FLUID: continue
                for d,(dxv,dyv,dzv) in enumerate(NB):
                    ax = d//2
                    xn,yn,zn = x+dxv, y+dyv, z+dzv
                    out = not(0<=xn<nx and 0<=yn<ny and 0<=zn<nz)
                    if not out and tp[idx(xn,yn,zn)]==FLUID:
                        if d%2==1: continue
                        j = idx(xn,yn,zn); vf = vface(i,j,ax)
                        # transpose of advection (each fwd "Cn[a]+=c*C[b]" -> "Pn[b]+=c*P[a]")
                        if vf>=0:
                            Pn[i]-=ka*vf*P[i]            # from Cn[i]-=ka*vf*C[i]
                            Pn[i]+=ka*vf*P[j]            # from Cn[j]+=ka*vf*C[i]
                        else:
                            Pn[j]-=ka*vf*P[i]            # from Cn[i]-=ka*vf*C[j]
                            Pn[j]+=ka*vf*P[j]            # from Cn[j]+=ka*vf*C[j]
                        # transpose of diffusion
                        Pn[j]+=kd*P[i]; Pn[i]-=kd*P[i]   # from Cn[i]+=kd*(C[j]-C[i])
                        Pn[i]+=kd*P[j]; Pn[j]-=kd*P[j]   # from Cn[j]+=kd*(C[i]-C[j])
                    elif not out and tp[idx(xn,yn,zn)]!=FLUID:
                        dcoef = dep + (w_s if d==5 else 0.0)
                        Pn[i]-=dcoef*P[i]                # transpose of self-loss = self-loss
                    else:
                        if ax==0 and xn<0:
                            vf=ux[i]
                            if vf<0: Pn[i]-=ka*vf*P[i]
                        elif ax==0 and xn>=nx:
                            vf=ux[i]
                            if vf>=0: Pn[i]-=ka*vf*P[i]
    return Pn

def fwd_N(C,n):
    for _ in range(n): C = fwd_step(C)
    return C
def adj_N(P,n):
    for _ in range(n): P = adj_step(P)      # adjoint of MⁿN is (Mᵀ)ⁿ applied n times
    return P

rng = np.random.default_rng(0)
mask = fluid.astype(float)

# ── (A) dot-product adjoint test (single step and N steps) ──────────────────
print("=== (A) dot-product test  <M u,v> vs <u, Mᵀ v>  (fluid cells only) ===")
for n in (1, 10, 40):
    u = rng.standard_normal(N)*mask
    v = rng.standard_normal(N)*mask
    Mu = fwd_N(u, n)
    MTv = adj_N(v, n)
    lhs = float(np.dot(Mu, v))
    rhs = float(np.dot(u, MTv))
    rel = abs(lhs-rhs)/max(abs(lhs),abs(rhs),1e-300)
    print(f"  n={n:>2} steps:  <Mu,v>={lhs:+.8e}  <u,Mᵀv>={rhs:+.8e}  rel.err={rel:.2e}")

# ── (B) source-receptor reciprocity for a few (s,r) pairs ───────────────────
print("\n=== (B) reciprocity  C(r;s) vs C*(s;r)  over n=40 steps ===")
def cell_ok(x,y,z): return tp[idx(x,y,z)]==FLUID
pairs = [((2,10,3),(22,10,3)), ((2,6,2),(18,14,5)), ((2,14,6),(24,8,2))]
n = 40
for (s,r) in pairs:
    si, ri = idx(*s), idx(*r)
    es = np.zeros(N); es[si]=1.0                  # unit release at s
    Cf = fwd_N(es, n); fwd_val = Cf[ri]           # forward conc at r
    er = np.zeros(N); er[ri]=1.0                  # unit adjoint-source at r
    Ca = adj_N(er, n); adj_val = Ca[si]           # adjoint field at s
    rel = abs(fwd_val-adj_val)/max(abs(fwd_val),abs(adj_val),1e-300)
    print(f"  s={s} r={r}:  C(r;s)={fwd_val:.8e}  C*(s;r)={adj_val:.8e}  rel.err={rel:.2e}")

print("\nNote: this validates the LINEAR (upwind) transport adjoint, incl. the")
print("deposition, settling, inlet (zero-inflow) and outlet (open) boundary terms.")
print("Production advection uses a van Leer limiter (nonlinear); for that, the")
print("adjoint must be taken of the operator linearized about the forward solution")
print("(tangent-linear/adjoint pair), or the linear scheme used for the adjoint pass.")
