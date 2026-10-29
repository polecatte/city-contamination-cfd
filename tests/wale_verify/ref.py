# Reference check for OpenLB WaleEffectiveOmega: run via run.sh (needs ./wale_check built from wale_check.cpp)
import numpy as np, subprocess, sys
rng=np.random.default_rng(1)
Cw=0.325
def ref(g):  # Nicoud & Ducros 1999, eq. for nu_t, lattice units (Delta=1), returns tau_t = 3 nu_t
    S=0.5*(g+g.T); g2=g@g; Sd=0.5*(g2+g2.T)-np.trace(g2)/3*np.eye(3)
    a=(Sd*Sd).sum(); b=(S*S).sum(); den=b**2.5+a**1.25
    return 0.0 if den==0 else 3*Cw**2*a**1.5/den
def ourfix(g):  # urban_les.h formula, transcribed
    g2=g@g; tr=np.trace(g2)/3; sdsd=ss=0
    for i in range(3):
        for j in range(3):
            sd=0.5*(g2[i,j]+g2[j,i])-(tr if i==j else 0); s=0.5*(g[i,j]+g[j,i]); sdsd+=sd*sd; ss+=s*s
    den=ss**2.5+sdsd**1.25; return 0.0 if den==0 else 3*Cw**2*sdsd**1.5/den
def stockpred(g):  # what the stock loops reduce to: only the (2,2) entries
    g2=g@g; G22=g2[2,2]-(g[0,0]**2+g[1,1]**2+g[2,2]**2)/3; s22=g[2,2]
    a=G22**2; b=s22**2; den=b**2.5+a**1.25; return 0.0 if den==0 else 3*Cw**2*a**1.5/den
cases={}
gam=0.01
cases['pure shear du/dz']=np.array([[0,0,gam],[0,0,0],[0,0,0]])
cases['solid rotation']=np.array([[0,-gam,0],[gam,0,0],[0,0,0]])
cases['axisymmetric strain']=np.diag([gam,gam,-2*gam])
cases['plane strain in x-y (w=0)']=np.diag([gam,-gam,0])
cases['shear + rotation in x-y']=np.array([[0,gam,0],[0.3*gam,0,0],[0,0,0]])
for k in range(6):
    A=rng.normal(size=(3,3))*gam; A-=np.trace(A)/3*np.eye(3); cases[f'random traceless #{k}']=A
inp="\n".join(" ".join(f"{x:.17g}" for x in g.ravel()) for g in cases.values())
out=subprocess.run(['./wale_check'],input=inp,capture_output=True,text=True).stdout.split()
print(f"{'case':30s} {'reference':>12s} {'urban_les':>12s} {'OpenLB 1.8.1':>12s} {'stock=(2,2) only?':>18s}")
for (n,g),o in zip(cases.items(),out):
    print(f"{n:30s} {ref(g):12.4e} {ourfix(g):12.4e} {float(o):12.4e} {stockpred(g):18.4e}")
