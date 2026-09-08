import numpy as np, pandas as pd, matplotlib.pyplot as plt
d=pd.read_csv("infil.csv")

# Per-size model
def P(dpum):
    z=(np.log(np.maximum(dpum,1e-6))-np.log(0.3))/1.1
    return np.clip(0.95*np.exp(-0.5*z*z),0,1)
def k(dpum): return 0.10+0.05/np.maximum(1e-3,dpum)+0.20*dpum
def Finf(dpum,a): return P(dpum)*a/(a+k(dpum))

# Class-integrated F_inf over a lognormal MASS distribution (MMD, GSD)
def Finf_class(MMD,GSD,a):
    dp=np.geomspace(0.01,15,400)
    lnsig=np.log(GSD)
    w=np.exp(-0.5*((np.log(dp)-np.log(MMD))/lnsig)**2)/(dp*lnsig*np.sqrt(2*np.pi))
    w/=np.trapezoid(w,dp)
    return np.trapezoid(w*Finf(dp,a),dp)

fig,ax=plt.subplots(1,2,figsize=(13,5.2))
# (a) penetration + F_inf vs size
ax[0].plot(d.dp_um,d.P,'k-',lw=2,label="penetration P(d$_p$)")
for col,a in [("Finf_a0.2",0.2),("Finf_a0.5",0.5),("Finf_a1.0",1.0)]:
    ax[0].plot(d.dp_um,d[col],lw=1.8,label=f"F$_{{inf}}$, a={a}/h")
ax[0].set_xscale("log"); ax[0].set_xlabel("particle diameter d$_p$ ($\\mu$m)")
ax[0].set_ylabel("factor (0–1)")
ax[0].set_title("(a) Per-size penetration & infiltration\nP peaks ~0.3 $\\mu$m (Liu & Nazaroff 2003)")
ax[0].axvspan(2.0,2.5,color="#d45050",alpha=.12)
ax[0].axvline(2.5,color="#d45050",ls=":",lw=1); ax[0].axvline(10,color="#d45050",ls=":",lw=1)
ax[0].annotate("PM2.5\ncutoff",(2.5,0.85),color="#d45050",fontsize=8,ha="center")
ax[0].annotate("PM10\ncutoff",(10,0.85),color="#d45050",fontsize=8,ha="center")
ax[0].legend(fontsize=8); ax[0].grid(alpha=.3,which="both")

# (b) class-integrated F_inf vs AER, with literature band
a=np.linspace(0.1,1.5,30)
for MMD,GSD,lab,c in [(0.4,2.0,"PM2.5 (MMD 0.4 µm, GSD 2.0)","#2a5a9a"),
                      (3.0,2.2,"PM10 (MMD 3 µm, GSD 2.2)","#c49030")]:
    ax[1].plot(a,[Finf_class(MMD,GSD,ai) for ai in a],lw=2,color=c,label=lab)
ax[1].axhspan(0.3,0.82,color="#4a8a3a",alpha=.15,label="PM2.5 measured F$_{inf}$\n0.3–0.82 (Chen & Zhao 2011)")
ax[1].set_xlabel("air-exchange rate a (1/h)"); ax[1].set_ylabel("class-integrated F$_{inf}$")
ax[1].set_title("(b) Mass-distribution-integrated F$_{inf}$\nlands in the measured band")
ax[1].set_ylim(0,1); ax[1].legend(fontsize=8,loc="upper left"); ax[1].grid(alpha=.3)

fig.suptitle("Building infiltration model  F$_{inf}$ = P·a/(a+k)  "
             "(Liu & Nazaroff 2001; Chen & Zhao 2011)",fontsize=13,fontweight="bold")
plt.tight_layout(rect=[0,0,1,0.95])
plt.savefig("/mnt/user-data/outputs/infiltration_model.png",dpi=150,bbox_inches="tight")
print("PM2.5 class F_inf @a=0.5:", round(Finf_class(0.4,2.0,0.5),2),
      "| PM10 class F_inf @a=0.5:", round(Finf_class(3.0,2.2,0.5),2))
print("saved infiltration_model.png")
