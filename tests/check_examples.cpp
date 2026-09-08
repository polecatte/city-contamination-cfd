// Acceptance checks on the example cities:
//   1) total effective inhabitance (buildings+parks+streets) == total population
//   2) city centred in the simulation space
//   3) business employment capacity matches employment need (P*LABOR_RATE)
//   4) domain borders meet COST 732 / Tominaga et al. (2008) (power-of-2 NOT required)
//
// COST 732 (Franke et al. 2006) / AIJ (Tominaga et al. 2008, JWEIA 96:1749) minima,
// H = tallest building: inflow >=5H upstream, outflow >=15H downstream, lateral
// >=5H each side, top >=5H above tallest (=> domain height >=6H), blockage <3%.
// To meet ALL minima AND keep the city centred, buffers must be uniform 15H (the
// max requirement); we size to that, foregoing the power-of-2 round-up.
#include "city_builder7.h"
#include "voxelize.h"
#include "occupancy.h"
#include <cstdio>
#include <cmath>
#include <string>
using namespace city;

struct Spec { std::string name; double sx,sy, base_h, cbd_peak, cbd_decay, park_frac, block_w, block_d; };

// build params for `cityW` footprint with given knobs and uniform buffer `buf` (m),
// vertical domain `Sz` (m); no power-of-2 (set Sx/Sy/Scz directly).
static Params mkp(const Spec& s, double buf){
    Params p{};
    p.city_w=480; p.city_h=480;
    p.block_w=s.block_w; p.block_d=s.block_d; p.base_height=s.base_h;
    p.cbd_peak=s.cbd_peak; p.cbd_decay=s.cbd_decay; p.cbd_aspect=1; p.cbd_angle=0;
    p.biz_inner_frac=0.05; p.biz_aspect=1;
    p.park_centrality=0.5; p.park_fraction=s.park_frac;
 p.roughness=0.15;
    p.road_w_x=s.sx; p.road_w_y=s.sy;
    p.population_total=20000; p.wind_direction=0;
    p.buf_xn=p.buf_xp=p.buf_yn=p.buf_yp=buf;
    p.Sx=p.buf_xn+p.city_w+p.buf_xp;
    p.Sy=p.buf_yn+p.city_h+p.buf_yp;
    double cx=p.buf_xn+p.city_w*0.5, cy=p.buf_yn+p.city_h*0.5;
    p.cbd_x=cx; p.cbd_y=cy; p.biz_center_x=cx; p.biz_center_y=cy;
    p.source_x=p.buf_xn-20; p.source_y=cy;
    return p;
}

int main(){
    const double P=20000.0, LR=LABOR_RATE, AREA_PW=BIZ_AREA_PW;
    double street_frac = street_fraction(nhaps_average());   // 0.093
    std::vector<Spec> S = {
        {"A baseline",      20,20, 12, 40, 8e-6, 0.10, 48,48},
        {"B narrow 8m",      8, 8, 12, 40, 8e-6, 0.10, 48,48},
        {"C wide 40m",      40,40, 12, 40, 8e-6, 0.10, 48,48},
        {"D anisotropic",   40, 8, 12, 40, 8e-6, 0.10, 48,48},
        {"E high parks",    20,20, 12, 40, 8e-6, 0.30, 48,48},
        {"F tall CBD",      20,20, 10,120, 2.5e-5,0.08, 48,48},
    };
    int total_fail=0;
    for(auto& s : S){
        // First a cheap generate to learn the tallest building H, then size the
        // COST-732 domain (uniform 15H buffer, vertical 6H), foregoing power-of-2.
        Params p0=mkp(s, 200.0);
        Result r0=generate(p0);
        double H=r0.max_height;
        double buf = 15.0*H;                      // uniform => meets 5H & 15H, stays centred
        double Sz  = 6.0*H;                        // 5H headroom above tallest
        int nz = std::max(2,(int)std::ceil(Sz/CELL));   // raw, NOT rounded to pow2
        Params p=mkp(s, buf);
        Result r=generate(p);

        // ---- Check 1: total effective inhabitance == population ----
        double inh_bld=0; for(auto&b:r.blocks) inh_bld+=b.eff_inh;   // buildings+parks (0.908P)
        // voxelize a MODEST instance (street total is domain-independent) to place streets
        Params pv=mkp(s, 80.0); Result rv=generate(pv);
        VoxelGrid g=voxelize(pv,rv,true,0.0,1800.0,0.5);
        StreetField sf=build_street_occupancy(pv,rv,g, street_frac*P);
        double inh_total=inh_bld + sf.total;
        bool c1 = std::fabs(inh_total - P)/P < 2e-3;

        // ---- Check 2: city centred in the simulation space ----
        int minx=1e9,miny=1e9,maxx=-1e9,maxy=-1e9;
        for(auto&b:r.blocks){ minx=std::min(minx,b.bx0); miny=std::min(miny,b.by0);
                              maxx=std::max(maxx,b.bx1); maxy=std::max(maxy,b.by1); }
        double cx_city=(minx+maxx)*0.5*CELL, cy_city=(miny+maxy)*0.5*CELL;
        double off_x=cx_city-p.Sx*0.5, off_y=cy_city-p.Sy*0.5;
        bool c2 = std::fabs(off_x)<=CELL && std::fabs(off_y)<=CELL;

        // ---- Check 3: employment capacity matches need ----
        double capacity = r.biz_floor_area / AREA_PW;   // workers the offices can hold
        double need     = r.population * LR;             // workers the population supplies
        double ratio = capacity/std::max(1.0,need);
        bool c3 = ratio>=0.9 && ratio<=1.3;              // builder's land-use swap band

        // ---- Check 4: COST-732 borders (no pow2) ----
        double up=p.buf_xn, down=p.buf_xp, latn=p.buf_yn, latp=p.buf_yp;
        double top = nz*CELL - H;
        bool up_ok=up>=5*H, down_ok=down>=15*H, lat_ok=(latn>=5*H&&latp>=5*H), top_ok=top>=5*H;
        // Blockage = PROJECTED frontal silhouette onto the y–z plane (flow along x):
        // per y-cell the tallest building covering it (buildings behind it are
        // shadowed, so this is the true projected area, not a sum of facades).
        int nyc=(int)std::round(p.Sy/CELL);
        std::vector<double> ymax(nyc,0.0);
        for(auto&b:r.blocks){ if(b.usage==PARK)continue; double h=b.height_cells*CELL;
            for(int y=std::max(0,b.y0); y<std::min(nyc,b.y1); ++y) ymax[y]=std::max(ymax[y],h); }
        double frontal=0; for(double h:ymax) frontal+=h*CELL;
        double blockage = frontal/(p.Sy*(nz*CELL));
        bool block_ok = blockage<0.03;
        bool c4 = up_ok&&down_ok&&lat_ok&&top_ok&&block_ok;

        int fails=(!c1)+(!c2)+(!c3)+(!c4); total_fail+=fails;
        printf("\n=== %s  (H=%.0fm) ===\n", s.name.c_str(), H);
        printf(" 1 inhabitance: buildings+parks=%.1f + street=%.1f = %.1f  vs P=%.0f  -> %s\n",
               inh_bld, sf.total, inh_total, P, c1?"PASS":"FAIL");
        printf(" 2 centring   : city centre=(%.0f,%.0f)  domain centre=(%.0f,%.0f)  off=(%.1f,%.1f)m -> %s\n",
               cx_city,cy_city, p.Sx*0.5,p.Sy*0.5, off_x,off_y, c2?"PASS":"FAIL");
        printf(" 3 employment : capacity=%.0f workers  need=%.0f (P*%.2f)  ratio=%.2f -> %s\n",
               capacity, need, LR, ratio, c3?"PASS":"FAIL");
        printf(" 4 COST-732   : up=%.0f(>=%.0f)%s down=%.0f(>=%.0f)%s lat=%.0f(>=%.0f)%s top=%.0f(>=%.0f)%s blockage=%.2f%%(<3%%)%s\n",
               up,5*H, up_ok?"ok":"NO", down,15*H, down_ok?"ok":"NO",
               latn,5*H, lat_ok?"ok":"NO", top,5*H, top_ok?"ok":"NO",
               100*blockage, block_ok?"ok":"NO");
        printf("   domain (no pow2): %.0f x %.0f x %.0f m  = %d x %d x %d cells  -> %s\n",
               p.Sx,p.Sy,nz*CELL, (int)(p.Sx/CELL),(int)(p.Sy/CELL),nz, c4?"PASS":"FAIL");
    }
    printf("\n%s (%d total failures across 6 cities x 4 checks)\n",
           total_fail==0?"ALL CHECKS PASS":"FAILURES PRESENT", total_fail);
    return total_fail?1:0;
}
