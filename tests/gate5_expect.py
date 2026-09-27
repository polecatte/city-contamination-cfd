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

    # carve_sponge(): erosion, matching urban_flow.cpp. A cell joins the sponge only if IT
    # AND ALL SIX NEIGHBOURS are MAT_FLUID, keeping MAT_SPONGE one cell clear of every domain
    # boundary plane -- OpenLB needs a boundary cell's inward neighbour to be MAT_FLUID
    # specifically. Eligibility is judged against the ORIGINAL map, so an earlier mark in the
    # same band cannot disqualify its neighbour.
    dn = ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1))
    def eligible(x, y, z):
        if mat[idx(x,y,z)] != MAT_FLUID:
            return False
        for ddx, ddy, ddz in dn:
            xx, yy, zz = x+ddx, y+ddy, z+ddz
            if not (0 <= xx < nx and 0 <= yy < ny and 0 <= zz < nz):
                return False
            if mat[idx(xx,yy,zz)] != MAT_FLUID:
                return False
        return True

    wd = wind % 360.0
    if wd < 45 or wd >= 315: rng = [(x,y,z) for z in range(nz) for y in range(ny) for x in range(nx-nsp, nx)]
    elif wd < 135:           rng = [(x,y,z) for z in range(nz) for x in range(nx) for y in range(ny-nsp, ny)]
    elif wd < 225:           rng = [(x,y,z) for z in range(nz) for y in range(ny) for x in range(nsp)]
    else:                    rng = [(x,y,z) for z in range(nz) for x in range(nx) for y in range(nsp)]
    pick = [idx(x,y,z) for (x,y,z) in rng if eligible(x,y,z)]
    for i in pick:
        mat[i] = MAT_SPONGE
    carved = len(pick)

    post = collections.Counter(mat)
    names = {0:"VOID",1:"FLUID",2:"WALL",3:"INLET",4:"OUTLET",5:"SLIP",6:"POROUS",
             7:"GROUND",8:"SPONGE",9:"FRAME"}
    print(f"grid {nx}x{ny}x{nz}  dx={dxm/1000.0:.3f}  total={nx*ny*nz}")
    print(f"sponge: {nsp} cells at wind {wind:g} deg, eroded -> {carved} FLUID cells converted\n")
    print(f"{'MAT':>3} {'name':<7} {'stage A':>10} {'after carve':>12}   <- GATE5 must equal this")
    for m in range(10):
        print(f"{m:>3} {names[m]:<7} {pre.get(m,0):>10} {post.get(m,0):>12}")
    print(f"\ntotal preserved: {sum(post.values())} == {nx*ny*nz}: {sum(post.values())==nx*ny*nz}")

if __name__ == "__main__":
    main()
