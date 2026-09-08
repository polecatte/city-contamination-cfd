"""Standalone city renderer. Reads exported city text files."""
import sys, numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.colors as mc

COLORS = {0:'#4a8a3a', 1:'#2a5a9a', 2:'#c49030', 3:'#e8d070'}
LABELS = {0:'Park', 1:'Business', 2:'Hi-Dens Res', 3:'Lo-Dens Res'}

def load_city(fn):
    meta = {}; buildings = []
    with open(fn) as f: lines = f.readlines()
    i = 0
    while i < len(lines):
        tok = lines[i].strip().split()
        if not tok: i += 1; continue
        if tok[0] == 'CELL':
            meta['cell'] = float(tok[1])
        elif tok[0] == 'SIZE':
            meta['Sx'] = float(tok[1]); meta['Sy'] = float(tok[2])
        elif tok[0] == 'GRID':
            v = [int(x) for x in tok[1:]]
            meta['bw'],meta['bd'],meta['rmaj'],meta['rmin'],meta['sb'] = v[0:5]
            meta['ox'],meta['oy'],meta['nbx'],meta['nby'] = v[5:9]
            if len(v) > 9: meta['maj_interval'] = v[9]
        elif tok[0] == 'STATS':
            v = tok[1:]
            meta['alpha'] = float(v[0]); meta['maxH'] = float(v[1])
            meta['pop'] = float(v[2]); meta['N'] = int(v[3])
            meta['np'] = int(v[4]); meta['nb'] = int(v[5])
            meta['nr'] = int(v[6]); meta['nm'] = int(v[7])
        elif tok[0] == 'BUILDINGS':
            n = int(tok[1]); i += 1
            for _ in range(n):
                q = lines[i].strip().split(); i += 1
                buildings.append({
                    'x0':int(q[1]),'y0':int(q[2]),'x1':int(q[3]),'y1':int(q[4]),
                    'bx0':int(q[5]),'by0':int(q[6]),'bx1':int(q[7]),'by1':int(q[8]),
                    'hc':int(q[9]),'usage':int(q[10])})
            continue
        i += 1
    return buildings, meta

def draw_2d(ax, buildings, meta, cell=2.0):
    Sx = meta['Sx']; Sy = meta['Sy']; mxH = max(meta['maxH'], 1)

    ax.set_xlim(0, Sx); ax.set_ylim(0, Sy)
    ax.set_aspect('equal')

    # Fill entire domain as road (dark)
    ax.set_facecolor('#555')

    # Draw block setback zones first (lighter gray)
    for b in buildings:
        ax.add_patch(Rectangle((b['bx0']*cell, b['by0']*cell),
                     (b['bx1']-b['bx0'])*cell, (b['by1']-b['by0'])*cell,
                     facecolor='#b5b0a5', zorder=2, linewidth=0))

    # Draw buildings
    for b in buildings:
        x0 = b['x0']*cell; y0 = b['y0']*cell
        w = (b['x1']-b['x0'])*cell; h = (b['y1']-b['y0'])*cell
        if w <= 0 or h <= 0: continue
        u = b['usage']; hc = b['hc']
        if u == 0:
            ax.add_patch(Rectangle((x0,y0),w,h,facecolor='#4a8a3a',
                         edgecolor='#3a7a2a',linewidth=0.15,zorder=3))
        else:
            rgb = mc.to_rgb(COLORS[u])
            bright = 0.55 + 0.45 * (hc * cell / mxH)
            fc = tuple(min(1.0, c * bright) for c in rgb)
            ax.add_patch(Rectangle((x0,y0),w,h,facecolor=fc,
                         edgecolor='#444',linewidth=0.1,zorder=3))

def draw_3d(ax, buildings, meta, cell=2.0, z_exag=1.5):
    Sx = meta['Sx']; Sy = meta['Sy']; mxH = max(meta['maxH'], 1)

    gx = np.array([[0, Sx], [0, Sx]])
    gy = np.array([[0, 0], [Sy, Sy]])
    gz = np.array([[0, 0], [0, 0]])
    ax.plot_surface(gx, gy, gz, color='#c8c4bc', alpha=0.3)

    for b in buildings:
        x0 = b['x0']*cell; y0 = b['y0']*cell
        x1 = b['x1']*cell; y1 = b['y1']*cell
        bh = b['hc']*cell; u = b['usage']
        w = x1 - x0; d = y1 - y0
        if w <= 0 or d <= 0: continue
        if u == 0:
            # Park with canopy height
            if bh < 1:
                ax.add_collection3d(Poly3DCollection(
                    [[[x0,y0,0.1],[x1,y0,0.1],[x1,y1,0.1],[x0,y1,0.1]]],
                    facecolors=['#4a8a3a'], linewidths=0, alpha=1.0))
            else:
                # Draw canopy as semi-transparent green volume
                sides = [
                    [[x0,y0,0],[x1,y0,0],[x1,y0,bh],[x0,y0,bh]],
                    [[x1,y0,0],[x1,y1,0],[x1,y1,bh],[x1,y0,bh]],
                    [[x1,y1,0],[x0,y1,0],[x0,y1,bh],[x1,y1,bh]],
                    [[x0,y1,0],[x0,y0,0],[x0,y0,bh],[x0,y1,bh]]]
                ax.add_collection3d(Poly3DCollection(sides,
                    facecolors=['#3a7a2a','#2d6a1d','#2d6a1d','#3a7a2a'],
                    linewidths=0, alpha=0.6))
                ax.add_collection3d(Poly3DCollection(
                    [[[x0,y0,bh],[x1,y0,bh],[x1,y1,bh],[x0,y1,bh]]],
                    facecolors=['#4a8a3a'], linewidths=0, alpha=0.7))
            continue
        if bh < 1: continue
        rgb = mc.to_rgb(COLORS[u])
        bright = 0.55 + 0.45 * (bh / mxH)
        sc_l = tuple(min(1.0, c*0.80*bright) for c in rgb)
        sc_d = tuple(min(1.0, c*0.55*bright) for c in rgb)
        roof = tuple(min(1.0, c*bright*1.05) for c in rgb)
        sides = [
            [[x0,y0,0],[x1,y0,0],[x1,y0,bh],[x0,y0,bh]],
            [[x1,y0,0],[x1,y1,0],[x1,y1,bh],[x1,y0,bh]],
            [[x1,y1,0],[x0,y1,0],[x0,y1,bh],[x1,y1,bh]],
            [[x0,y1,0],[x0,y0,0],[x0,y0,bh],[x0,y1,bh]]]
        ax.add_collection3d(Poly3DCollection(sides,
            facecolors=[sc_l, sc_d, sc_d, sc_l], linewidths=0, alpha=1.0))
        ax.add_collection3d(Poly3DCollection(
            [[[x0,y0,bh],[x1,y0,bh],[x1,y1,bh],[x0,y1,bh]]],
            facecolors=[roof], edgecolors=['#555'], linewidths=0.08, alpha=1.0))

    ax.set_xlim(0, Sx); ax.set_ylim(0, Sy)
    ax.set_zlim(0, mxH * 1.1)
    ax.set_box_aspect([Sx, Sy, z_exag * mxH * 1.1])
    ax.view_init(elev=28, azim=240)
    ax.tick_params(labelsize=7)
    ax.set_xlabel('x (m)', fontsize=8)
    ax.set_ylabel('y (m)', fontsize=8)
    ax.set_zlabel('h (m)', fontsize=8)

def render(city_files, labels, output_fn):
    n = len(city_files)
    fig_w = max(14, 7 * n)
    fig, axes = plt.subplots(2, n, figsize=(fig_w, 12),
                              gridspec_kw={'height_ratios': [1, 1.2]})
    if n == 1: axes = axes.reshape(2, 1)
    for col in range(n):
        bl, m = load_city(city_files[col])
        mxH = m['maxH']
        ax2d = axes[0, col]
        draw_2d(ax2d, bl, m)
        ax2d.set_title(f"{labels[col]}\n{m['N']} blk, max {mxH:.0f}m, {m['pop']:.0f} pop",
                       fontsize=11, fontweight='bold')
        ax2d.set_xticks([]); ax2d.set_yticks([])
        axes[1, col].remove()
        ax3d = fig.add_subplot(2, n, n + col + 1, projection='3d')
        draw_3d(ax3d, bl, m)
        ax3d.set_title(f"max {mxH:.0f}m ({int(mxH/3)} floors)", fontsize=11)
    legend_patches = [Patch(facecolor=COLORS[i], edgecolor='#444', label=LABELS[i]) for i in range(4)]
    fig.legend(handles=legend_patches, loc='lower center', ncol=4, fontsize=11, frameon=True)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(output_fn, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_fn}")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        render([sys.argv[1]], ['City'], '/mnt/user-data/outputs/city_render.png')

# ═══════════════════════════════════════════════════════════════════
# Velocity field overlay
# ═══════════════════════════════════════════════════════════════════

def synthetic_velocity(buildings, meta, cell=4.0, U=1.0, angle=0.0):
    """Generate a simple synthetic velocity field for visualization testing.
    Uniform wind with building blockage — NOT physically accurate."""
    Sx, Sy = meta['Sx'], meta['Sy']
    nx, ny = int(Sx/cell), int(Sy/cell)
    ux = np.full((ny,nx), U*np.cos(angle), dtype=np.float32)
    uy = np.full((ny,nx), U*np.sin(angle), dtype=np.float32)

    # Zero velocity inside building footprints
    for b in buildings:
        x0,y0,x1,y1 = b['x0'],b['y0'],b['x1'],b['y1']
        x0=max(0,min(nx,x0)); x1=max(0,min(nx,x1))
        y0=max(0,min(ny,y0)); y1=max(0,min(ny,y1))
        ux[y0:y1, x0:x1] = 0
        uy[y0:y1, x0:x1] = 0

    # Simple wake: reduce velocity behind buildings (in wind direction)
    wake_len = int(20/cell)  # ~20m wake
    for b in buildings:
        if b['usage']==0: continue  # skip parks (porous)
        x0,y0,x1,y1 = b['x0'],b['y0'],b['x1'],b['y1']
        hc = b['hc']
        if hc < 1: continue
        # Wake behind building (downwind)
        if abs(np.cos(angle)) > abs(np.sin(angle)):
            # Wind mainly in x
            if np.cos(angle) > 0:  # wind +x, wake after x1
                for dx in range(1, wake_len+1):
                    xw = min(nx-1, x1+dx)
                    decay = 1.0 - 0.7*np.exp(-dx/(wake_len/3))
                    ux[y0:y1, xw] *= decay
            else:
                for dx in range(1, wake_len+1):
                    xw = max(0, x0-dx)
                    decay = 1.0 - 0.7*np.exp(-dx/(wake_len/3))
                    ux[y0:y1, xw] *= decay
        else:
            if np.sin(angle) > 0:
                for dy in range(1, wake_len+1):
                    yw = min(ny-1, y1+dy)
                    decay = 1.0 - 0.7*np.exp(-dy/(wake_len/3))
                    uy[yw, x0:x1] *= decay
            else:
                for dy in range(1, wake_len+1):
                    yw = max(0, y0-dy)
                    decay = 1.0 - 0.7*np.exp(-dy/(wake_len/3))
                    uy[yw, x0:x1] *= decay

    # Slight acceleration in street canyons (mass conservation)
    speed = np.sqrt(ux**2 + uy**2)
    mean_speed = np.mean(speed[speed > 0.01])
    # Boost streets where speed is below average but nonzero
    mask = (speed > 0.01) & (speed < mean_speed*0.8)
    ux[mask] *= 1.15
    uy[mask] *= 1.15

    return ux, uy


def load_velocity_binary(fn, nx, ny, nz, z_slice=1):
    """Load velocity from solver binary output.
    Format: nx,ny,nz (int32), then ux[N],uy[N],uz[N] (float32), N=nx*ny*nz."""
    import struct
    with open(fn, 'rb') as f:
        hdr = struct.unpack('3i', f.read(12))
        assert hdr == (nx,ny,nz), f"Grid mismatch: file {hdr} vs expected ({nx},{ny},{nz})"
        N = nx*ny*nz
        ux_flat = np.frombuffer(f.read(N*4), dtype=np.float32)
        uy_flat = np.frombuffer(f.read(N*4), dtype=np.float32)
    # Reshape and extract z-slice
    ux_3d = ux_flat.reshape(nz, ny, nx)
    uy_3d = uy_flat.reshape(nz, ny, nx)
    return ux_3d[z_slice], uy_3d[z_slice]


def draw_velocity(ax, ux, uy, meta, cell=4.0, stride=6, scale=None,
                  cmap='RdYlBu_r', alpha=0.85, zorder=5,
                  arrow_width=None, U_phys=None):
    """Overlay velocity vectors on a 2D city map.

    Args:
        ax:      matplotlib axes (should already have draw_2d applied)
        ux, uy:  2D arrays (ny x nx) of velocity components (any units)
        meta:    city metadata dict
        cell:    meters per cell
        stride:  subsample every N cells for readability
        scale:   quiver scale (auto-computed if None)
        cmap:    colormap for speed magnitude
        alpha:   arrow transparency
        zorder:  draw order (above buildings)
        arrow_width: shaft width in axes fraction (auto if None)
        U_phys:  physical wind speed for axis label (None = show raw units)
    """
    ny, nx = ux.shape
    Sx, Sy = meta['Sx'], meta['Sy']

    # Subsample
    xs = np.arange(0, nx, stride) * cell + cell*stride/2
    ys = np.arange(0, ny, stride) * cell + cell*stride/2
    X, Y = np.meshgrid(xs, ys)
    Ux = ux[::stride, ::stride]
    Uy = uy[::stride, ::stride]

    # Speed for coloring
    speed = np.sqrt(Ux**2 + Uy**2)
    max_speed = np.max(speed)
    if max_speed < 1e-10:
        print("WARNING: velocity field is near-zero everywhere")
        return None

    # Normalize arrows so they fill roughly one stride-gap in length
    # at max speed. This makes arrows visible regardless of unit system.
    if scale is None:
        # scale = max_speed / (stride * cell * 0.8) in quiver's convention:
        # quiver scale = U / arrow_length_in_data_coords
        scale = max_speed / (stride * cell * 0.7)

    if arrow_width is None:
        arrow_width = 0.0025

    # If physical speed given, normalize colors to that
    if U_phys is not None:
        color_scale = speed * (U_phys / max_speed)
        clabel = 'Wind speed (m/s)'
        clim = [0, U_phys]
    else:
        color_scale = speed
        clabel = 'Wind speed'
        clim = [0, max_speed]

    q = ax.quiver(X, Y, Ux, Uy, color_scale,
                  cmap=cmap, alpha=alpha, scale=scale,
                  width=arrow_width, headwidth=3.5, headlength=4.5,
                  headaxislength=3.5, minshaft=1.2, minlength=0.5,
                  zorder=zorder, clim=clim)

    cb = plt.colorbar(q, ax=ax, shrink=0.6, pad=0.02, aspect=30)
    cb.set_label(clabel, fontsize=9)
    cb.ax.tick_params(labelsize=8)

    return q
