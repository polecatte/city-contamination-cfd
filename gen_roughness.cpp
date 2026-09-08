// Generate a ROUGHNESS series: identical city, only `roughness` varies, so the
// renders isolate the (now exaggerated, log-normal) height-heterogeneity axis.
// roughness ≈ height coefficient of variation σ_H/H̄. Pure builder (no LBM).
#include "city_builder7.h"
#include <cstdio>
using namespace city;

int main(){
    double rns[]={0.0, 0.3, 0.6, 0.8};
    const char* files[]={"/tmp/rough_0.txt","/tmp/rough_3.txt","/tmp/rough_6.txt","/tmp/rough_8.txt"};
    for(int i=0;i<4;++i){
        Params p{};
        p.city_w=480; p.city_h=480; p.block_w=48; p.block_d=48; p.base_height=15;
        p.cbd_peak=40; p.cbd_decay=8e-6; p.cbd_aspect=1; p.cbd_angle=0;
        p.biz_inner_frac=0.05; p.biz_aspect=1;
        p.park_centrality=0.5; p.park_fraction=0.10; p.roughness=rns[i];
        p.road_w_x=20; p.road_w_y=20; p.population_total=20000; p.wind_direction=0;
        p.buf_xn=p.buf_xp=p.buf_yn=p.buf_yp=40;
        p.Sx=p.buf_xn+p.city_w+p.buf_xp; p.Sy=p.buf_yn+p.city_h+p.buf_yp;
        double cx=p.buf_xn+p.city_w*0.5, cy=p.buf_yn+p.city_h*0.5;
        p.cbd_x=cx; p.cbd_y=cy; p.biz_center_x=cx; p.biz_center_y=cy;
        p.source_x=p.buf_xn-20; p.source_y=cy;
        Result r=generate(p);
        export_city(files[i],p,r);
        // report the realized height CoV
        double s=0,s2=0,mx=0; int n=0;
        for(auto&b:r.blocks){ if(b.usage==PARK)continue; double h=b.height_cells*CELL;
            s+=h;s2+=h*h;n++; if(h>mx)mx=h; }
        double m=s/n, sd=std::sqrt(std::max(0.0,s2/n-m*m));
        printf("roughness=%.1f -> %s  meanH=%.0fm  CoV=%.2f  maxH=%.0fm  pop=%.0f\n",
               rns[i],files[i],m,sd/m,mx,r.population);
    }
    return 0;
}
