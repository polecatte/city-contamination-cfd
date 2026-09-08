#pragma once
// geometry_loader.h — Stage B: load the geometry-bridge material map into an OpenLB
// SuperGeometry, and load the Ω source mask for the advection–diffusion lattice.
//
// Consumes the artifacts written by gen_openlb_geom.cpp (Stage A):
//   material_map.dat : 5-int32 header [nx,ny,nz,dx*1000,1] + nx*ny*nz int32 materials
//   source_mask.u8   : 5-int32 header + nx*ny*nz uint8   (Ω = 1)
//
// The material numbers are exactly the OpenLB convention Stage B assigns dynamics to
// (see openlb_geometry.h): 1 fluid, 2 wall(buildings), 3 inlet, 4 outlet, 5 slip,
// 6 porous(parks), 7 ground(rough-wall floor).
//
// TWO PARTS:
//   (A) A plain, dependency-free reader (`GridField`) — parses the 5-int format. This is
//       compilable and unit-testable WITHOUT OpenLB (see GEOMLOADER_SELFTEST below), so
//       the file handshake is verified independently of the solver build.
//   (B) An OpenLB stamping helper `stampSuperGeometry(...)` that copies the material
//       array onto a SuperGeometry cell-for-cell. This half needs the OpenLB headers and
//       compiles inside your 1.8 build tree. Every version-sensitive call is flagged
//       "CONFIRM 1.8" — verify against the OpenLB 1.8 User Guide before relying on it.

#include <cstdint>
#include <cstdio>
#include <vector>
#include <string>

namespace bridge {

// ── (A) dependency-free reader ──────────────────────────────
template <class T>
struct GridField {
    int nx=0, ny=0, nz=0; double dx=0.0; int ncomp=1;
    std::vector<T> data;
    inline size_t idx(int x,int y,int z) const { return (size_t)z*ny*nx + (size_t)y*nx + x; }
    inline size_t size() const { return (size_t)nx*ny*nz; }
};

template <class T>
inline bool read_grid(const std::string& fn, GridField<T>& g) {
    FILE* f = fopen(fn.c_str(), "rb");
    if (!f) { fprintf(stderr,"[geometry_loader] cannot open %s\n", fn.c_str()); return false; }
    int hdr[5];
    if (fread(hdr, sizeof(int), 5, f) != 5) { fclose(f); return false; }
    g.nx=hdr[0]; g.ny=hdr[1]; g.nz=hdr[2]; g.dx=hdr[3]/1000.0; g.ncomp=hdr[4];
    size_t n = g.size() * (size_t)g.ncomp;
    g.data.resize(n);
    size_t got = fread(g.data.data(), sizeof(T), n, f);
    fclose(f);
    if (got != n) { fprintf(stderr,"[geometry_loader] short read %s (%zu/%zu)\n",fn.c_str(),got,n); return false; }
    return true;
}

// Convenience typed loaders.
inline bool load_material_map(const std::string& fn, GridField<int32_t>& m){ return read_grid(fn, m); }
inline bool load_source_mask (const std::string& fn, GridField<uint8_t>& s){ return read_grid(fn, s); }

// ── (B) OpenLB stamping (compiles inside the OpenLB 1.8 build tree) ──
// Guarded so this header also compiles standalone for the self-test.
#ifdef URBAN_FLOW_WITH_OPENLB
// Copy the imported material numbers onto `superGeometry`. Assumes the SuperGeometry was
// created over a cuboid of exactly nx×ny×nz cells with physical spacing dx (see
// urban_flow.cpp main()), so the OpenLB lattice coordinate (iX,iY,iZ) maps 1:1 to our
// (x,y,z) array index.
//
// Two defects fixed here versus the first draft, both of which would have silently
// scrambled the geometry rather than failing loudly:
//
//  (S1) `block.getOrigin()` returns the block origin "in SI units (meter)", NOT lattice
//       indices — at dx=4 m every global index was 4× too small. The global offset must
//       come from the cuboid decomposition's mother cuboid. A single-cuboid serial run
//       has origin (0,0,0), so this looked correct right up until it went MPI/multi-block.
//
//  (B7) `block.get(x,y,z) = mat` cannot compile: BlockGeometry::get() returns `int` BY
//       VALUE and is const. The 1.8 write accessor is `block.set({x,y,z}, mat)`.
//
// The caller supplies the global lattice origin of each local block (urban_flow.cpp's
// blockOriginOf()), so this header stays free of decomposition-API details.
template <class SGEOM>
inline void stampSuperGeometry(SGEOM& superGeometry, const GridField<int32_t>& m) {
    auto& load = superGeometry.getLoadBalancer();
    auto& cd   = superGeometry.getCuboidDecomposition();          // CONFIRM 1.8 accessor name
    for (int iC = 0; iC < load.size(); ++iC) {                    // local cuboids
        auto& block = superGeometry.getBlockGeometry(iC);
        // (S1) global LATTICE origin of this block
        auto originR = cd.getMotherCuboid().getLatticeR(cd.get(load.glob(iC)).getOrigin());
        const int gx0 = (int)originR[0], gy0 = (int)originR[1], gz0 = (int)originR[2];
        const int bnx = block.getNx(), bny = block.getNy(), bnz = block.getNz();
        for (int x=0; x<bnx; ++x) for (int y=0; y<bny; ++y) for (int z=0; z<bnz; ++z) {
            int X=gx0+x, Y=gy0+y, Z=gz0+z;
            if (X<0||X>=m.nx||Y<0||Y>=m.ny||Z<0||Z>=m.nz) continue;
            block.set({x,y,z}, m.data[m.idx(X,Y,Z)]);             // (B7) 1.8 write accessor
        }
    }
    superGeometry.updateStatistics();
}
#endif // URBAN_FLOW_WITH_OPENLB

} // namespace bridge

// ── standalone self-test of the parser (no OpenLB needed) ───
//   g++ -O2 -std=c++17 -DGEOMLOADER_SELFTEST geometry_loader.h -x c++ -o geomloader_test
#ifdef GEOMLOADER_SELFTEST
#include <map>
int main(int argc, char** argv){
    std::string dir = argc>1 ? argv[1] : "geom_out";
    bridge::GridField<int32_t> m; bridge::GridField<uint8_t> s;
    if(!bridge::load_material_map(dir+"/material_map.dat", m)) return 2;
    if(!bridge::load_source_mask (dir+"/source_mask.u8",   s)) return 2;
    printf("material_map: %dx%dx%d dx=%.3f  (%zu cells)\n", m.nx,m.ny,m.nz,m.dx,m.size());
    std::map<int,long> hist; for(auto v: m.data) hist[v]++;
    const char* nm[8]={"donothing","fluid","wall","inlet","outlet","slip","porous","ground"};
    for(auto& kv: hist) printf("  MAT %d %-10s %ld\n", kv.first, (kv.first>=0&&kv.first<=7)?nm[kv.first]:"?", kv.second);
    long omega=0; for(auto v: s.data) if(v) omega++;
    printf("source_mask: Omega = %ld cells\n", omega);
    printf("SELFTEST OK\n");
    return 0;
}
#endif
