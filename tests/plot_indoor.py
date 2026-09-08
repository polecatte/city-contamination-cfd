import numpy as np, pandas as pd, matplotlib.pyplot as plt
d=pd.read_csv("indoor_demo.csv")
# infiltration model for the io-vs-dp panel
def P(dpum): z=(np.log(np.maximum(dpum,1e-6))-np.log(0.3))/1.1; return np.clip(0.95*np.exp(-0.5*z*z),0,1)
def k(dpum): return 0.10+0.05/np.maximum(1e-3,dpum)+0.20*dpum
def io(dpum,a,lam): return P(dpum)*a/(a+k(dpum)+lam)

fig,ax=plt.subplots(1,2,figsize=(13,5.2))
# (a) protective-action curve from the real run
ax[0].plot(d.lambda_filt,d.total_indoor_exposure/d.total_indoor_exposure.iloc[0],'-o',
           color="#2a5a9a",lw=2,ms=7)
for x,red in zip(d.lambda_filt,d.reduction_pct):
    ax[0].annotate(f"-{red:.0f}%",(x,d.total_indoor_exposure[d.lambda_filt==x].iloc[0]/d.total_indoor_exposure.iloc[0]),
                   textcoords="offset points",xytext=(6,8),fontsize=9,color="#2a5a9a")
ax[0].set_xlabel("indoor filtration loss rate $\\lambda_{filt}$ (1/h)  [CADR/V]")
ax[0].set_ylabel("total indoor exposure (normalized)")
ax[0].set_title("(a) Indoor filtration as protective action\n(end-to-end run: solid buildings + ABL inlet)")
ax[0].annotate("portable\ncleaner",(2,0.30),fontsize=8,color="#666")
ax[0].annotate("HEPA",(5,0.16),fontsize=8,color="#666")
ax[0].grid(alpha=.3); ax[0].set_ylim(0,1.05)

# (b) I/O ratio vs particle size at several filtration levels (model)
dp=np.geomspace(0.02,10,200)
for lam,c,lab in [(0,"#d45050","no filter"),(2,"#c49030","cleaner $\\lambda$=2/h"),
                  (5,"#2a5a9a","HEPA $\\lambda$=5/h"),(10,"#4a8a3a","strong $\\lambda$=10/h")]:
    ax[1].plot(dp,io(dp,0.5,lam),lw=2,color=c,label=lab)
ax[1].set_xscale("log"); ax[1].set_xlabel("particle diameter ($\\mu$m)")
ax[1].set_ylabel("indoor/outdoor ratio  C$_{in}$/C$_{out}$")
ax[1].set_title("(b) I/O = P·a/(a+k+$\\lambda_{filt}$)\n(Nazaroff 2004; a_inf=0.5/h)")
ax[1].axhspan(0.3,0.82,color="#4a8a3a",alpha=.10)
ax[1].legend(fontsize=8); ax[1].grid(alpha=.3,which="both"); ax[1].set_ylim(0,1)

fig.suptitle("Indoor filtration in the exposure chain  "
             "(solid buildings + infiltration model; Liu & Nazaroff 2001, Chen & Zhao 2011, Nazaroff 2004)",
             fontsize=12,fontweight="bold")
plt.tight_layout(rect=[0,0,1,0.95])
plt.savefig("/mnt/user-data/outputs/indoor_filtration.png",dpi=150,bbox_inches="tight")
print("saved indoor_filtration.png")
