// must_geom.h — parametric container-array geometry for the MUST (Mock Urban
// Setting Test) dispersion benchmark. 120 shipping containers (12.2 x 2.42 x
// 2.54 m) in a regular array; a neutral tracer is released from a point source
// and sampled at receptor towers. Refs: Yee & Biltoft (2004); Biltoft (2001);
// wind-tunnel companion Bezpalcova (2007), CEDVAL-LES / Hamburg.
//
// NOTE ON PRECEDENT: container DIMENSIONS below are the published MUST values.
// The exact array PITCH, array WIND ANGLE (theta, varies by trial), point-SOURCE
// location, and RECEPTOR coordinates are trial-specific and must be set from the
// chosen trial's spec sheet. Defaults here are the nominal aligned (theta=0)
// array; set MustSpec fields to match a specific trial before scoring.
#pragma once
#include <vector>
#include <cstdint>
#include <cmath>
#include <cstdio>

namespace must {

enum { FLUID=0, GROUND=1, SOLID=2 };   // SOLID = impermeable container (SHELL, perm=0)

struct MustSpec {
    double dx      = 0.5;    // m/cell (resolution; 0.5 -> H~5 cells, fits 16 GB)
    double cL=12.2, cW=2.42, cH=2.54;   // container L,W,H (m) — MUST published
    int    rows=12, cols=10;            // 12x10 = 120 containers
    double pitchX=12.9, pitchY=6.5;     // centre-to-centre street pitch (m) [trial-set]
    double up_m=25, down_m=220, lat_m=40, top_m=25;  // margins (m): downwind reaches receptors
    // source + a couple of receptor towers (m, domain coords) — TRIAL-SET placeholders
    double src_x=30, src_y=0, src_z=1.3;             // point source (rel. array origin)
};

// Selected NEAR-NEUTRAL MUST trials for a spread across wind angle, release
// height, and flow regime. Stable/very-stable trials are excluded on purpose:
// the solver is neutral (no buoyancy), so those would fail for unmodelled
// stratification, confounding a numerics check. theta = approach-wind angle to
// the array x-axis (deg); the LBM keeps the array axis-aligned and imposes the
// oblique inflow via the per-face inlet. Values marked [DATA] must be filled
// from the Yee & Biltoft (2004) trial files (exact source x/y, release rate Q,
// receptor coords, u_ref for the aligned trial) before scoring.
struct MustTrial { const char* name; double theta_deg, u_ref_ms, release_h_m; const char* note; };
static const MustTrial MUST_TRIALS[] = {
  {"2681829", -41.0, 7.93, 1.80,
   "reference: strongly oblique wind, elevated release (~71% of H); plume deflects across the array and is swept over containers. Near-neutral (Lo>2500 m)."},
  {"2681849", -41.0, 7.93, 0.15,
   "release-height contrast: same horizontal source as 2681829, near-ground release; low plume channels along the streets instead of over the tops. Near-neutral."},
  {"2640246",  20.0, 0.0 , 1.30,
   "wind-angle contrast: near-aligned approach (<30 deg to array) -> street-channeling regime, plume carried farther downwind. [u_ref, exact theta, source from DATA]"},
};
inline MustSpec spec_for(const MustTrial& t){ MustSpec s; s.src_z=t.release_h_m; return s; }

struct MustGeom {
    int nx, ny, nz; size_t N;
    std::vector<uint8_t> tp;
    double dx; int ax0, ay0;             // array origin cell (x,y of container block start)
    int n_containers; double blockage_frontal;
};

inline int gidx(int x,int y,int z,int nx,int ny){ return (z*ny+y)*nx+x; }

inline MustGeom build(const MustSpec& s){
    auto m2c=[&](double m){ return (int)std::llround(m/s.dx); };
    int cLx=std::max(1,m2c(s.cL)), cWy=std::max(1,m2c(s.cW)), cHz=std::max(1,m2c(s.cH));
    int pX=std::max(cLx+1,m2c(s.pitchX)), pY=std::max(cWy+1,m2c(s.pitchY));
    int arrX=(s.cols-1)*pX+cLx, arrY=(s.rows-1)*pY+cWy;
    int upc=m2c(s.up_m), dnc=m2c(s.down_m), latc=m2c(s.lat_m), topc=m2c(s.top_m);
    MustGeom g; g.dx=s.dx;
    g.nx=upc+arrX+dnc; g.ny=arrY+2*latc; g.nz=cHz+topc;
    g.N=(size_t)g.nx*g.ny*g.nz; g.ax0=upc; g.ay0=latc;
    g.tp.assign(g.N,(uint8_t)FLUID);
    // ground plane z=0
    for(int y=0;y<g.ny;++y)for(int x=0;x<g.nx;++x) g.tp[gidx(x,y,0,g.nx,g.ny)]=GROUND;
    // container array (rows x cols), impermeable SOLID, resting on the ground (z=1..cHz)
    int nc=0;
    for(int r=0;r<s.rows;++r)for(int c=0;c<s.cols;++c){
        int x0=g.ax0+c*pX, y0=g.ay0+r*pY;
        for(int z=1;z<=cHz && z<g.nz;++z)
          for(int y=y0;y<y0+cWy && y<g.ny;++y)
            for(int x=x0;x<x0+cLx && x<g.nx;++x)
                g.tp[gidx(x,y,z,g.nx,g.ny)]=SOLID;
        ++nc;
    }
    g.n_containers=nc;
    // frontal blockage = summed frontal area of one column of containers / (ny*nz)
    double frontal=(double)s.rows*cWy*cHz;  // one streamwise column presents rows*W*H
    g.blockage_frontal=frontal/((double)g.ny*g.nz);
    return g;
}

inline void report(const MustGeom& g){
    printf("[MUST geom] %d x %d x %d = %.1fM cells @ %.2f m/cell\n",
           g.nx,g.ny,g.nz,g.N/1e6,g.dx);
    printf("[MUST geom] containers=%d  array origin cell=(%d,%d)  frontal blockage=%.1f%%\n",
           g.n_containers,g.ax0,g.ay0,100.0*g.blockage_frontal);
    double GB=g.N*289.0/(1024.0*1024*1024);
    printf("[MUST geom] est. LBM memory @289 B/cell = %.2f GB %s\n",
           GB, GB<15.0?"(fits A4000 16 GB)":"(EXCEEDS 16 GB — coarsen dx)");
}

} // namespace must
