#!/usr/bin/env python3
"""gate5_expect.py — expected per-material voxel counts for Gate 5.

Reflects the one-face-per-cell rule (OPENLB_MIGRATION_PLAN.md 6.4): box edges and
corners are MAT_DONOTHING, so INLET/OUTLET/SLIP are smaller than the pre-fix figures
and MAT 0 is no longer zero.

Reads Stage A's material_map.dat, replays urban_flow.cpp's carve_sponge() on the host
array exactly as the solver does, and prints the histogram OpenLB's
superGeometry.getStatistics().getNvoxel(m) must reproduce.

  python3 tests/gate5_expect.py geom_out [SPONGE_CELLS] [WIND_DEG]
"""
import sys, struct, collections

MAT_FLUID, MAT_SPONGE = 1, 8

def main():
    d      = sys.argv[1] if len(sys.argv) > 1 else "geom_out"
    nsp    = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    wind   = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
    with open(f"{d}/material_map.dat", "rb") as f:
        nx, ny, nz, dxm, ncomp = struct.unpack("<5i", f.read(20))
        mat = list(struct.unpack(f"<{nx*ny*nz}i", f.read(4*nx*ny*nz)))
    idx = lambda x, y, z: z*ny*nx + y*nx + x
    pre = collections.Counter(mat)

    # carve_sponge(): wind-aligned band of FLUID cells before the outlet face
    wd = wind % 360.0
    if wd < 45 or wd >= 315:  rng = [(x, y, z) for z in range(nz) for y in range(ny) for x in range(nx-nsp, nx)]
    elif wd < 135:            rng = [(x, y, z) for z in range(nz) for x in range(nx) for y in range(ny-nsp, ny)]
    elif wd < 225:            rng = [(x, y, z) for z in range(nz) for y in range(ny) for x in range(nsp)]
    else:                     rng = [(x, y, z) for z in range(nz) for x in range(nx) for y in range(nsp)]
    carved = 0
    for x, y, z in rng:
        i = idx(x, y, z)
        if mat[i] == MAT_FLUID:
            mat[i] = MAT_SPONGE; carved += 1

    post = collections.Counter(mat)
    names = {0:"VOID",1:"FLUID",2:"WALL",3:"INLET",4:"OUTLET",5:"SLIP",6:"POROUS",7:"GROUND",8:"SPONGE"}
    print(f"grid {nx}x{ny}x{nz}  dx={dxm/1000.0:.3f}  total={nx*ny*nz}")
    print(f"sponge: {nsp} cells at wind {wind:g} deg -> {carved} FLUID cells converted\n")
    print(f"{'MAT':>3} {'name':<7} {'stage A':>10} {'after carve':>12}   <- GATE5 must equal this")
    for m in range(9):
        print(f"{m:>3} {names[m]:<7} {pre.get(m,0):>10} {post.get(m,0):>12}")
    print(f"\ntotal preserved: {sum(post.values())} == {nx*ny*nz}: {sum(post.values())==nx*ny*nz}")

if __name__ == "__main__":
    main()
