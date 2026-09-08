// Monte Carlo over the 12-D design space -> morphological metrics that drive the
// flow and the exposure weights. Output CSV is analysed by sensitivity.py for
// first-order (variance-based) sensitivity indices. Builder is deterministic;
// the SAMPLER uses a seeded RNG (analysis only — not part of the city build).
//
// Metrics chosen for dispersion/exposure relevance:
//   maxH, meanH, sigmaH  : trap depth, canopy depth, height heterogeneity (mixing)
//   lam_p                : plan-area density (built fraction)
//   lam_f                : frontal-area density  -> drag / roughness z0 / ventilation
//                          (Macdonald 1998; Grimmond & Oke 1999) — the key aero knob
//   HW                   : canyon aspect ratio meanH/street_width -> flow regime (Oke 1988)
//   parkfrac             : realized open-space fraction (ventilation, source-free area)
//   bizfloor             : business floor area (employment capacity / worker weight)
//   inh_core             : fraction of effective inhabitance within inner R/3
//                          (how centralized the exposure WEIGHT w is)
//   nblk                 : number of built blocks
#include "city_builder7.h"
#include <cstdio>
#include <cmath>
#include <random>
using namespace city;

int main(int argc,char**argv){
    int N = (argc>1)?atoi(argv[1]):2000;
    std::mt19937 rng(12345); std::uniform_real_distribution<double> U(0,1);
    const double cm=512.0, carea=cm*cm;
    FILE* f=fopen("sensitivity.csv","w");
    fprintf(f,"block_w,block_d,cbd_peak,cbd_decay,patchiness,"
              "park_centrality,park_fraction,roughness,street_width,"
              "maxH,meanH,sigmaH,lam_p,lam_f,HW,parkfrac,bizfloor,inh_core,nblk,"
              "meanH_per,HW_per\n");
    for(int s=0;s<N;++s){
        double u[12]; for(double&v:u) v=U(rng);
        double block_w=24+u[0]*(72-24), block_d=16+u[1]*(40-16);
        double cbd_peak=20+u[2]*(120-20);
        double cbd_decay=1e-6*std::pow(3e-5/1e-6,u[3]);              // log scale
        double patch=u[4]*1.0, park_cen=u[5];
        double park_frac=u[6]*0.40, rough=u[7]*0.80;                 // roughness now [0,0.8]
        double street=8+u[8]*(40-8);
        Params p{};
        p.city_w=cm; p.city_h=cm; p.block_w=block_w; p.block_d=block_d; // base_height fixed (BASE_HEIGHT_M)
        p.cbd_peak=cbd_peak; p.cbd_decay=cbd_decay; p.cbd_aspect=1; p.cbd_angle=0;
        p.patchiness=patch; p.biz_inner_frac=0; p.biz_aspect=1;
        p.park_centrality=park_cen; p.park_fraction=park_frac; p.roughness=rough;
        p.road_w_x=street; p.road_w_y=street; p.population_total=20000; p.wind_direction=0;
        p.buf_xn=p.buf_xp=p.buf_yn=p.buf_yp=200;
        p.Sx=p.buf_xn+cm+p.buf_xp; p.Sy=p.buf_yn+cm+p.buf_yp;
        double cx=p.buf_xn+cm*0.5, cy=p.buf_yn+cm*0.5;
        p.cbd_x=cx; p.cbd_y=cy; p.biz_center_x=cx; p.biz_center_y=cy;
        p.source_x=p.buf_xn-20; p.source_y=cy;
        Result r=generate(p);
        // ---- metrics ----
        double sumH=0,sumH2=0,foot_built=0,foot_park=0,frontal=0,inh_tot=0,inh_core=0;
        double sumH_per=0; int nper=0;
        int nblk=0; double R=cm*0.5, R3=R/3.0, Rper=2.0*R/3.0;
        for(auto&b:r.blocks){
            double fw=(b.x1-b.x0)*CELL, fd=(b.y1-b.y0)*CELL, foot=fw*fd;
            double h=b.height_cells*CELL;
            double bx=(b.bx0+b.bx1)*0.5*CELL, by=(b.by0+b.by1)*0.5*CELL;
            double rr=std::hypot(bx-cx,by-cy);
            inh_tot+=b.eff_inh; if(rr<R3) inh_core+=b.eff_inh;
            if(b.usage==PARK){ foot_park+=foot; continue; }
            nblk++; sumH+=h; sumH2+=h*h; foot_built+=foot; frontal+=fd*h;  // frontal to +x
            if(rr>Rper){ sumH_per+=h; nper++; }                            // outer-ring canopy
        }
        double meanH = nblk? sumH/nblk : 0;
        double sigmaH= nblk? std::sqrt(std::max(0.0,sumH2/nblk-meanH*meanH)) : 0;
        double meanH_per = nper? sumH_per/nper : 0;
        double lam_p = foot_built/carea, lam_f = frontal/carea;
        double HW = meanH/street, HW_per = meanH_per/street;
        double parkfrac = foot_park/carea;
        double inhc = inh_tot>0? inh_core/inh_tot : 0;
        fprintf(f,"%.4f,%.4f,%.4f,%.3e,%.4f,%.4f,%.4f,%.4f,%.4f,"
                  "%.3f,%.3f,%.3f,%.5f,%.5f,%.4f,%.5f,%.1f,%.5f,%d,%.3f,%.4f\n",
            block_w,block_d,cbd_peak,cbd_decay,patch,park_cen,park_frac,
            rough,street, r.max_height,meanH,sigmaH,lam_p,lam_f,HW,parkfrac,
            r.biz_floor_area,inhc,nblk, meanH_per,HW_per);
    }
    fclose(f); printf("wrote sensitivity.csv (%d samples)\n",N);
    return 0;
}
