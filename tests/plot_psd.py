import numpy as np, pandas as pd, matplotlib.pyplot as plt
b=pd.read_csv("psd_bins.csv"); v=pd.read_csv("vd_curve.csv")
fig,ax=plt.subplots(1,3,figsize=(16,5))

# (a) lognormal activity PSD + bin fractions
MMAD,GSD=1.0,2.0
dp=np.geomspace(0.05,20,300); lnsig=np.log(GSD)
pdf=np.exp(-0.5*((np.log(dp)-np.log(MMAD))/lnsig)**2)/(dp*lnsig*np.sqrt(2*np.pi))
ax[0].plot(dp,pdf/pdf.max(),'k-',lw=2,label="lognormal activity pdf\n(MMAD=1 µm, GSD=2)")
ax[0].bar(b.d_ae_um,b.frac/b.frac.max(),width=b.d_ae_um*0.35,color="#2a5a9a",alpha=.6,
          label="6 size bins")
ax[0].set_xscale("log"); ax[0].set_xlabel("aerodynamic diameter (µm)")
ax[0].set_ylabel("normalized activity"); ax[0].legend(fontsize=8)
ax[0].set_title("(a) Aerosol size distribution\n(lognormal; Hinds 1999)"); ax[0].grid(alpha=.3,which="both")

# (b) deposition velocity V-curve + old constants
ax[1].plot(v.dp_um,v.vd_park,color="#4a8a3a",lw=2,label="Park (vegetation)")
ax[1].plot(v.dp_um,v.vd_res_low,color="#c49030",lw=2,label="Lo-dens res")
ax[1].plot(v.dp_um,v.vd_biz,color="#2a5a9a",lw=2,label="Business (urban)")
# old hand-set constants (m/s -> cm/s)
ax[1].axhline(3.0,ls="--",color="#4a8a3a",alpha=.6); ax[1].annotate("old DEP_PARK=3 cm/s (10-30x too high)",(0.02,3.2),fontsize=7,color="#4a8a3a")
ax[1].axhline(0.1,ls="--",color="#2a5a9a",alpha=.6); ax[1].annotate("old DEP_BIZ=0.1 cm/s",(0.02,0.11),fontsize=7,color="#2a5a9a")
ax[1].set_xscale("log"); ax[1].set_yscale("log")
ax[1].set_xlabel("particle diameter (µm)"); ax[1].set_ylabel("dry deposition velocity v$_d$ (cm/s)")
ax[1].set_title("(b) Size-dependent v$_d$ — Zhang et al. 2001\n(V-shape, min in accumulation mode)")
ax[1].legend(fontsize=8); ax[1].grid(alpha=.3,which="both")

# (c) per-bin settling velocity (log) + airborne vs depositing tendency
ax2=ax[2]
ax2.bar(np.arange(len(b)),b.w_s_mm_s,color="#d45050",alpha=.7)
ax2.set_yscale("log"); ax2.set_xlabel("bin (fine -> coarse)")
ax2.set_ylabel("settling velocity w$_s$ (mm/s)",color="#d45050")
ax2.set_xticks(np.arange(len(b))); ax2.set_xticklabels([f"{d:.2f}" for d in b.d_ae_um],fontsize=7)
ax2.tick_params(axis='y',labelcolor="#d45050")
ax2b=ax2.twinx(); ax2b.plot(np.arange(len(b)),b.frac,'-o',color="#2a5a9a")
ax2b.set_ylabel("mass fraction",color="#2a5a9a"); ax2b.tick_params(axis='y',labelcolor="#2a5a9a")
ax2.set_title("(c) Per-bin settling (×500 across PSD)\nfine→airborne (airborne), coarse→deposits (surface)")
ax2.set_xlabel("bin representative d$_{ae}$ (µm)")

fig.suptitle("Polydisperse contaminant aerosol: mass-weighted size bins + size-resolved settling & deposition",
             fontsize=13,fontweight="bold")
plt.tight_layout(rect=[0,0,1,0.95])
plt.savefig("/mnt/user-data/outputs/polydisperse_psd_deposition.png",dpi=150,bbox_inches="tight")
print("saved polydisperse_psd_deposition.png")
