"""Plot the ground-level deposition map (and concentration) over the city.

Reads deposition_z1.bin / concentration_z1.bin written by main_cpu.cpp
(format: [nx,ny int32] then a single float32 field, ny x nx).
"""
from render_city import load_city, draw_2d
import matplotlib.pyplot as plt
import numpy as np
import struct, os


def load_field(fn):
    with open(fn, 'rb') as f:
        nx, ny = struct.unpack('2i', f.read(8))
        a = np.frombuffer(f.read(nx * ny * 4), dtype=np.float32).reshape(ny, nx).copy()
    return a, nx, ny


bl, m = load_city('city_export.txt')
cell = m.get('cell', 4.0)
Sx, Sy = m['Sx'], m['Sy']

panels = []
if os.path.exists('deposition_z1.bin'):
    panels.append(('deposition_z1.bin', 'Deposited mass (ground, cumulative)', 'inferno'))
if os.path.exists('concentration_z1.bin'):
    panels.append(('concentration_z1.bin', 'Concentration (ground, time-avg)', 'viridis'))

if not panels:
    raise SystemExit('No deposition_z1.bin / concentration_z1.bin found — run ./solver first.')

fig, axes = plt.subplots(1, len(panels), figsize=(11 * len(panels), 10))
if len(panels) == 1:
    axes = [axes]

for ax, (fn, title, cmap) in zip(axes, panels):
    field, nx, ny = load_field(fn)
    # Mask zeros so the city underneath shows through where nothing deposited.
    pos = field[field > 0]
    vmax = np.percentile(pos, 99) if pos.size else 1.0
    masked = np.ma.masked_less_equal(field, 0.0)

    draw_2d(ax, bl, m, cell=cell)
    xs = np.linspace(0, Sx, nx)
    ys = np.linspace(0, Sy, ny)
    im = ax.pcolormesh(xs, ys, masked, cmap=cmap, alpha=0.85,
                       shading='auto', vmin=0, vmax=vmax, zorder=6)
    plt.colorbar(im, ax=ax, shrink=0.7, label=title.split('(')[0].strip())
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xticks([]); ax.set_yticks([])

plt.tight_layout()
plt.savefig('deposition.png', dpi=150, bbox_inches='tight')
print('Saved deposition.png')
