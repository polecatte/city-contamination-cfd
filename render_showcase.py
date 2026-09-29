# render_showcase.py GEOM_DIR [PNG] — 3D view, domain plan and receptor map of a gen_openlb_geom city (needs scipy).
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import sys
G=sys.argv[1] if len(sys.argv)>1 else "geom_showcase"
def read5(fn,dt):
    b=open(fn,'rb').read(); h=np.frombuffer(b[:20],np.int32); return h,np.frombuffer(b[20:],dt)
h,m=read5(f"{G}/material_map.dat",np.int32); nx,ny,nz=h[:3]; dx=h[3]/1000
m=m[:nx*ny*nz].reshape(nz,ny,nx)
_,src=read5(f"{G}/source_mask.u8",np.uint8); src=src[:nx*ny*nz].reshape(nz,ny,nx)
_,w=read5(f"{G}/receptor_w.f32",np.float32); w=w[:nx*ny*nz].reshape(nz,ny,nx)
wall=(m==2); por=(m==6)
Hb=np.where(wall.any(0), nz-1-np.argmax(wall[::-1],axis=0), 0)*dx   # roof height per column (m)
park=por.any(0)
# building footprints -> labelled blocks
from scipy import ndimage
lab,nb=ndimage.label(Hb>0)
hs=[Hb[lab==i].max() for i in range(1,nb+1)]
xs=np.where((Hb>0)|park)[1]; x0,x1=xs.min(),xs.max()
print(f"grid {nx}x{ny}x{nz} dx={dx} m; buildings {nb}, heights {min(hs):.0f}-{max(hs):.0f} m (median {np.median(hs):.0f}); "
      f"city x {x0*dx:.0f}-{(x1+1)*dx:.0f} m of {nx*dx:.0f} m; parks {park.sum()*dx*dx/1e4:.2f} ha; "
      f"source cells {int(src.sum())}; receptor weight {w.sum():.0f}")

fig=plt.figure(figsize=(15,9.5))
# (a) 3D city
ax=fig.add_subplot(2,2,(1,3),projection='3d')
cm=plt.cm.plasma
for i in range(1,nb+1):
    ys,xx=np.where(lab==i); H=Hb[lab==i].max()
    ax.bar3d(xx.min()*dx,ys.min()*dx,0,(xx.max()-xx.min()+1)*dx,(ys.max()-ys.min()+1)*dx,H,
             color=cm(H/max(hs)),edgecolor='k',linewidth=0.3,shade=True,alpha=0.95)
pl,npk=ndimage.label(park)
for i in range(1,npk+1):
    ys,xx=np.where(pl==i)
    ax.bar3d(xx.min()*dx,ys.min()*dx,0,(xx.max()-xx.min()+1)*dx,(ys.max()-ys.min()+1)*dx,
             (np.where(por[:, ys[0], xx[0]])[0].max()+1)*dx,color='#6cc24a',alpha=0.55,edgecolor='none')
sy,sx=np.where(src[1]>0)
ax.scatter(sx*dx+2,sy*dx+2,np.full(sx.size,1.0),s=0.4,c='#e8590c',alpha=0.5)
ax.set_xlim(0,(x1+15)*dx); ax.set_ylim(0,ny*dx); ax.set_zlim(0,max(hs)*1.1)
ax.set_box_aspect(((x1+15)*dx,ny*dx,max(hs)*1.1*1.6))
ax.view_init(elev=28,azim=-60)
ax.set_xlabel('x (m, downwind)'); ax.set_ylabel('y (m)'); ax.set_zlabel('z (m)',labelpad=2)
ax.set_title(f"Showcase city: {nb} buildings ({min(hs):.0f}–{max(hs):.0f} m), 4 parks (green),\nrelease area Ω = streets + parks at ground level (orange)",fontsize=11)
sm=plt.cm.ScalarMappable(cmap=cm,norm=plt.Normalize(0,max(hs))); fig.colorbar(sm,ax=ax,shrink=0.5,pad=0.08,label='building height (m)')
# (b) whole domain plan
ax2=fig.add_subplot(2,2,2)
img=np.ma.masked_where(Hb==0,Hb)
ax2.imshow(np.where(park,1,np.nan),origin='lower',extent=[0,nx*dx,0,ny*dx],cmap=LinearSegmentedColormap.from_list('g',['#6cc24a','#6cc24a']),aspect='equal')
ax2.imshow(img,origin='lower',extent=[0,nx*dx,0,ny*dx],cmap=cm,vmin=0,vmax=max(hs),aspect='equal')
ax2.set_facecolor('#eef3f8')
ax2.axvspan((x1+1)*dx,nx*dx,color='#4263eb',alpha=0.07)
ax2.text(((x1+1)*dx+nx*dx)/2,ny*dx*0.5,f'wake buffer {(nx-x1-1)*dx:.0f} m (15 H)\n→ outlet',ha='center',va='center',fontsize=10,color='#364fc7')
ax2.set_xlabel('x (m)'); ax2.set_ylabel('y (m)'); ax2.set_title(f'Whole domain (plan, wind → +x): {nx*dx:.0f} m × {ny*dx:.0f} m × {nz*dx:.0f} m high, {nx*ny*nz/1e6:.1f} M cells at dx = {dx:.0f} m',fontsize=10)
# (c) receptor weight (where people are)
ax3=fig.add_subplot(2,2,4)
wc=w.sum(0); wz=np.ma.masked_where(wc<=0,wc)
im=ax3.imshow(wz,origin='lower',extent=[0,nx*dx,0,ny*dx],cmap='magma_r')
ax3.contour(np.arange(nx)*dx+2,np.arange(ny)*dx+2,(Hb>0).astype(float),levels=[0.5],colors='#495057',linewidths=0.6)
ax3.set_xlim(0,(x1+15)*dx); ax3.set_xlabel('x (m)'); ax3.set_ylabel('y (m)')
ax3.set_title('Where people are: receptor weight w per column (indoor rings inside\nbuilding outlines, outdoor pedestrians on streets) — J integrates exposure over this',fontsize=10)
fig.colorbar(im,ax=ax3,shrink=0.8,label='w per column')
fig.tight_layout(); fig.savefig(sys.argv[2] if len(sys.argv)>2 else "showcase_city.png",dpi=110)
