"""sensitivity_density.py — screen the 12 density-field parameters against a
CHEAP exposure proxy, because the real exposure comes from expensive LBM runs.

Strategy (multi-fidelity):
  1. Proxy: source-anonymous OR localized steady advection-diffusion-removal
     (the project's anon_exposure closures) on a coarse grid — a sparse solve,
     ~ms, not the LBM. J = <I, C>, C solves A[K(morph), lam(morph)] C = q over a
     narrow wind rose about wind_angle. I and q are sum-normalized.
  2. Morris elementary-effects screening over all 12 params: mu* (importance),
     sigma (nonlinearity/interaction). O(k) model calls per trajectory.
  3. Linear-w check: J is linear in the receptor w, so for a FIXED morphology the
     receptor-allocation params can be swept against a FROZEN C for free. We
     measure how well frozen-C tracks a full re-solve.
"""
import os, subprocess, numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from param_space_density import PARAM_SPACE_DENSITY, NAMES_DENSITY, to_physical_density, cli_args

BIN="./gen_density_city"; TMP="/tmp/sens"; os.makedirs(TMP,exist_ok=True)
N=48                      # proxy grid
SX=680.0; H=SX/N          # domain m / cell m
U=3.0                     # wind speed m/s
K_BG,AK,BK,AL = 2.0,0.15,2.0,0.12    # anon_exposure closures
HREF=60.0
SOURCE="point"           # 'point' = localized upwind release | 'uniform' = source-anonymous

def rasterize(fn, block_w_m):
    """City file -> height field (m) and inhabitance field on the N×N proxy grid (x,y)."""
    Hm=np.zeros((N,N)); Iw=np.zeros((N,N)); cell=4.0
    for ln in open(fn):
        if ln.startswith("CELL"): cell=float(ln.split()[1])
        if not ln.startswith("B "): continue
        q=ln.split()
        x0,y0,x1,y1=[int(q[t]) for t in (5,6,7,8)]; hc=int(q[9]); inh=float(q[12])
        gx0=int(x0*cell/H); gx1=max(gx0+1,int(x1*cell/H))
        gy0=int(y0*cell/H); gy1=max(gy0+1,int(y1*cell/H))
        gx0,gx1=max(0,gx0),min(N,gx1); gy0,gy1=max(0,gy0),min(N,gy1)
        if gx1<=gx0 or gy1<=gy0: continue
        Hm[gy0:gy1,gx0:gx1]=np.maximum(Hm[gy0:gy1,gx0:gx1], hc*cell)
        ncell=(gx1-gx0)*(gy1-gy0)
        Iw[gy0:gy1,gx0:gx1]+=inh/ncell
    return Hm.T, Iw.T, block_w_m   # transpose to [x,y] so wind (x) is applied correctly

def source_field(theta, mode):
    if mode=="uniform":
        return np.ones((N,N))/(N*N)
    xx,yy=np.mgrid[0:N,0:N]
    sx=N*0.5 - 0.42*N*np.cos(theta); sy=N*0.5 - 0.42*N*np.sin(theta)
    g=np.exp(-((xx-sx)**2+(yy-sy)**2)/(2*(N*0.06)**2))
    return g/g.sum()

def solve_adr(d, s_m, theta, q):
    """Vectorized steady advection-diffusion-removal; returns C (N×N). d,q in [x,y]."""
    Kf = K_BG + AK*U*(s_m*(1.0+BK*d)); lam= AL*(d/max(1.0,s_m))*U
    Ux,Uy = U*np.cos(theta), U*np.sin(theta)
    idx=lambda i,j:(i*N+j); rows=[]; cols=[]; vals=[]
    def add(r,c,v): rows.append(r); cols.append(c); vals.append(v)
    Kfl=Kf; laml=lam
    for i in range(N):
        for j in range(N):
            k=idx(i,j); diag=laml[i,j]
            if Ux>=0:
                if i>0: add(k,idx(i-1,j),-Ux/H)
                diag+=Ux/H
            else:
                if i<N-1: add(k,idx(i+1,j),Ux/H)
                diag+=-Ux/H
            if Uy>=0:
                if j>0: add(k,idx(i,j-1),-Uy/H)
                diag+=Uy/H
            else:
                if j<N-1: add(k,idx(i,j+1),Uy/H)
                diag+=-Uy/H
            for ii,jj in ((i+1,j),(i-1,j),(i,j+1),(i,j-1)):
                if 0<=ii<N and 0<=jj<N:
                    Kface=0.5*(Kfl[i,j]+Kfl[ii,jj])/(H*H); add(k,idx(ii,jj),-Kface); diag+=Kface
            add(k,k,diag)
    A=sp.csr_matrix((vals,(rows,cols)),shape=(N*N,N*N))
    return spsolve(A,q.reshape(N*N)).reshape(N,N)

def wind_rose(wind_angle_deg):
    th0=np.radians(wind_angle_deg)
    return [(th0,0.5),(th0+np.radians(25),0.25),(th0-np.radians(25),0.25)]

def proxy_from_fields(Hm, Iw, s_m, wind_angle_deg, C_cache=None):
    d=np.clip(0.08+0.5*(Hm/HREF),0.05,0.75); In=Iw/max(1e-12,Iw.sum())
    if C_cache is not None: C=C_cache
    else:
        C=np.zeros((N,N))
        for th,wt in wind_rose(wind_angle_deg): C+=wt*solve_adr(d,s_m,th,source_field(th,SOURCE))
    return float(np.sum(In*C)), C

def build_city(x_unit, tag):
    phys=to_physical_density(x_unit); out=os.path.join(TMP,f"{tag}.txt")
    subprocess.run([BIN,*cli_args(phys,out)],capture_output=True,check=True)
    return out, phys

def proxy_J(x_unit, tag):
    out,phys=build_city(x_unit,tag); Hm,Iw,s=rasterize(out,phys['block_w'])
    J,_=proxy_from_fields(Hm,Iw,s,phys['wind_angle_deg']); return J

def morris(k, r, levels=4, seed=0):
    rng=np.random.default_rng(seed); delta=levels/(2*(levels-1)); grid=np.linspace(0,1-delta,levels)
    EE={i:[] for i in range(k)}; ncall=[0]
    def J(x): ncall[0]+=1; return proxy_J(x,f"m{ncall[0]}")
    for t in range(r):
        base=rng.choice(grid,size=k); order=rng.permutation(k); x=base.copy(); jprev=J(x)
        for p in order:
            xn=x.copy()
            if xn[p]+delta<=1.0: xn[p]+=delta; sgn=1
            else: xn[p]-=delta; sgn=-1
            jn=J(xn); EE[p].append((jn-jprev)/(sgn*delta)); x=xn; jprev=jn
        print(f"  trajectory {t+1}/{r} done ({ncall[0]} proxy calls)")
    mu=np.array([np.mean(np.abs(EE[i])) for i in range(k)])
    sig=np.array([np.std(EE[i]) for i in range(k)]); return mu,sig,ncall[0]

def frozen_c_check(m=16, seed=1):
    rng=np.random.default_rng(seed); base=np.full(12,0.5); pop_dims=[6,7,8,9,10,11]
    out,phys=build_city(base,"fc_base"); Hm,Iw,s=rasterize(out,phys['block_w'])
    _,C_base=proxy_from_fields(Hm,Iw,s,phys['wind_angle_deg']); Jex=[]; Jfr=[]
    for i in range(m):
        x=base.copy()
        for d in pop_dims: x[d]=rng.random()
        out,phys=build_city(x,f"fc{i}"); Hm,Iw,s=rasterize(out,phys['block_w'])
        je,_=proxy_from_fields(Hm,Iw,s,phys['wind_angle_deg'])
        jf,_=proxy_from_fields(Hm,Iw,s,phys['wind_angle_deg'],C_cache=C_base)
        Jex.append(je); Jfr.append(jf)
    Jex=np.array(Jex); Jfr=np.array(Jfr)
    return np.corrcoef(Jex,Jfr)[0,1], np.mean(np.abs(Jfr-Jex)/np.abs(Jex)), Jex, Jfr

if __name__=="__main__":
    k=len(PARAM_SPACE_DENSITY); runs={}
    for mode in ("point","uniform"):
        globals()["SOURCE"]=mode
        print(f"\n=== Morris screening — source = {mode} ===")
        mu,sig,nc=morris(k, r=8, seed=3); runs[mode]=(mu,sig)
        for i in np.argsort(mu)[::-1]:
            print(f"  {NAMES_DENSITY[i]:20}{mu[i]:>14.3e}{sig[i]:>12.2e}")
    globals()["SOURCE"]="point"
    corr,relerr,Jex,Jfr=frozen_c_check(m=16)
    print(f"\nFrozen-C: corr={corr:.3f}  relerr={relerr:.1%}")
    np.savez("/tmp/sens/results.npz", point=runs["point"], uniform=runs["uniform"],
             Jex=Jex, Jfr=Jfr, corr=corr, relerr=relerr, names=NAMES_DENSITY)
    print("saved /tmp/sens/results.npz")
