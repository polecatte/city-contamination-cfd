"""First-order (variance-based) sensitivity from the Monte Carlo sample.
S_i = Var(E[Y|X_i]) / Var(Y), estimated by quantile-binning X_i (given-data
Sobol). One table per metric; then a synthesis ranking parameters by the most
variation they introduce anywhere. Low across the board => fold/remove candidate."""
import numpy as np, csv

rows=list(csv.reader(open("sensitivity.csv")))
hdr=rows[0]; data=np.array([[float(v) for v in r] for r in rows[1:]])
col={n:i for i,n in enumerate(hdr)}
INPUTS=["block_w","block_d","cbd_peak","cbd_decay","patchiness",
        "park_centrality","park_fraction","roughness","street_width"]
# metrics grouped by what they drive in J = sum w*C
AERO=["lam_f","HW","HW_per","maxH","meanH_per","sigmaH","lam_p"]   # drive concentration C (flow)
WEIGHT=["bizfloor","inh_core"]                    # drive exposure weight w
BOTH=["parkfrac"]
METRICS=AERO+BOTH+WEIGHT

def sobol1(x,y,K=12):
    """first-order index via quantile bins"""
    if y.std()<1e-12: return 0.0
    qs=np.quantile(x,np.linspace(0,1,K+1)); qs[0]-=1e-9; qs[-1]+=1e-9
    idx=np.digitize(x,qs)-1
    vy=y.var(); num=0.0; N=len(y)
    for b in range(K):
        m=idx==b
        if m.sum()<3: continue
        num+=(m.sum()/N)*(y[m].mean()-y.mean())**2
    return num/vy

S=np.zeros((len(INPUTS),len(METRICS)))
for j,met in enumerate(METRICS):
    y=data[:,col[met]]
    for i,inp in enumerate(INPUTS):
        S[i,j]=sobol1(data[:,col[inp]], y)

# ---- per-metric variability (is the metric even moving?) ----
print("metric            mean       CoV(std/mean)   (range)")
for met in METRICS:
    y=data[:,col[met]]
    cov=y.std()/abs(y.mean()) if abs(y.mean())>1e-12 else float('nan')
    print(f"  {met:9s} {y.mean():10.4g}   {cov:6.2f}        [{y.min():.4g}, {y.max():.4g}]")

print("\nFirst-order Sobol indices  S_i  (fraction of each metric's variance, %)")
print("                 | "+" ".join(f"{m:>6s}" for m in METRICS)+" |  AERO  WEIGHT   MAX")
print("-"*110)
order=np.argsort(-S.max(1))
for i in order:
    aero=S[i,[METRICS.index(m) for m in AERO]].sum()
    wt  =S[i,[METRICS.index(m) for m in WEIGHT]].sum()
    cells=" ".join(f"{100*S[i,j]:6.1f}" for j in range(len(METRICS)))
    print(f"{INPUTS[i]:16s} | {cells} | {100*aero:5.0f} {100*wt:6.0f} {100*S[i].max():5.0f}")

# ---- variance closure: how much is captured by first-order alone? ----
print("\nSum of first-order S_i per metric (≈1 ⇒ additive; ≪1 ⇒ interactions matter):")
for j,met in enumerate(METRICS):
    print(f"  {met:9s} {100*S[:,j].sum():5.0f}%")

# ---- candidates: low MAX S across ALL metrics ----
print("\nLOW-VARIATION candidates (max first-order S_i < 5% across every metric):")
for i in order[::-1]:
    if S[i].max()<0.05:
        print(f"  {INPUTS[i]:16s}  max S = {100*S[i].max():.1f}%  "
              f"(aero {100*S[i,[METRICS.index(m) for m in AERO]].sum():.0f}%, "
              f"weight {100*S[i,[METRICS.index(m) for m in WEIGHT]].sum():.0f}%)")
