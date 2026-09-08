import numpy as np, pandas as pd, matplotlib.pyplot as plt
prof=pd.read_csv("abl_profile.csv"); ts=pd.read_csv("abl_tseries.csv")
ustar=0.5046; kappa=0.41; z0=0.7
fig,ax=plt.subplots(2,2,figsize=(13,9))

# (a) mean profile vs analytic log-law
z=prof.z.to_numpy()
ax[0,0].plot(prof.Umean,z,'o',ms=3,color="#2a5a9a",label="generator (sampled)")
ax[0,0].plot(ustar/kappa*np.log((z+z0)/z0),z,'-',color="#d45050",lw=2,
             label=r"log-law $\frac{u_*}{\kappa}\ln\frac{z+z_0}{z_0}$ (Richards & Hoxey 1993)")
ax[0,0].set_xlabel("mean speed U (m/s)"); ax[0,0].set_ylabel("height z (m)")
ax[0,0].set_title("(a) Sheared mean inlet profile"); ax[0,0].legend(fontsize=8); ax[0,0].grid(alpha=.3)

# (b) sigma profiles vs surface-layer targets
ax[0,1].plot(prof.sig_u,z,color="#2a5a9a",label=r"$\sigma_u$ realized")
ax[0,1].plot(prof.sig_v,z,color="#4a8a3a",label=r"$\sigma_v$ realized")
ax[0,1].plot(prof.sig_w,z,color="#c49030",label=r"$\sigma_w$ realized")
for r,c in [(2.5,"#2a5a9a"),(1.9,"#4a8a3a"),(1.25,"#c49030")]:
    ax[0,1].axvline(r*ustar,ls="--",color=c,alpha=.6)
ax[0,1].set_xlabel(r"turbulence std (m/s); dashed = target $\{2.5,1.9,1.25\}u_*$")
ax[0,1].set_ylabel("height z (m)")
ax[0,1].set_title("(b) Fluctuation intensities vs surface-layer\nsimilarity (Panofsky & Dutton 1984)")
ax[0,1].legend(fontsize=8); ax[0,1].grid(alpha=.3)

# (c) time series
ax[1,0].plot(ts.t[:1500],ts.up[:1500],lw=.7,color="#333")
ax[1,0].set_xlabel("time (s)"); ax[1,0].set_ylabel("u' at z=40 m (m/s)")
ax[1,0].set_title("(c) Synthetic turbulence time series (RFG:\nKraichnan 1970; Smirnov et al. 2001)"); ax[1,0].grid(alpha=.3)

# (d) autocorrelation + spectrum (inset-style: two panels share)
up=ts.up.to_numpy(); up=up-up.mean(); dt=ts.t[1]-ts.t[0]
ac=np.correlate(up,up,'full'); ac=ac[ac.size//2:]; ac/=ac[0]
lags=np.arange(len(ac))*dt
axd=ax[1,1]; axd.plot(lags[:600],ac[:600],color="#2a5a9a")
axd.axhline(0,color="#999",lw=.6); axd.set_xlabel("lag (s)")
axd.set_ylabel("temporal autocorrelation"); axd.set_title("(d) Temporal coherence + PSD (inset)")
axd.grid(alpha=.3)
# spectrum inset
ai=axd.inset_axes([0.45,0.45,0.5,0.5])
f=np.fft.rfftfreq(len(up),dt); P=np.abs(np.fft.rfft(up))**2
m=(f>2e-2)&(f<8)
ai.loglog(f[m],P[m],lw=.8,color="#333")
ai.loglog(f[m],P[m][0]*(f[m]/f[m][0])**(-5/3),'--',color="#d45050",lw=1.2,label=r"$-5/3$")
ai.set_xlabel("freq (Hz)",fontsize=7); ai.set_ylabel("PSD",fontsize=7)
ai.tick_params(labelsize=6); ai.legend(fontsize=7)

fig.suptitle("Sheared turbulent ABL inlet — validation  "
             "(u*=%.2f m/s, z0=0.7 m, divergence residual 0.7%%)"%ustar,
             fontsize=13,fontweight="bold")
plt.tight_layout(rect=[0,0,1,0.97])
plt.savefig("/mnt/user-data/outputs/abl_inlet_validation.png",dpi=150,bbox_inches="tight")
print("saved abl_inlet_validation.png")
