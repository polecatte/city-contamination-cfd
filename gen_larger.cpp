// Larger (720 m) example cities across the new `patchiness` (business-spread) knob.
// A modest, sharpish CBD + a big footprint leaves the periphery below the RES_LOW
// threshold, so the low-density residential ring (25% lot coverage) is visible too.
#include "city_builder7.h"
#include <cstdio>
using namespace city;
int main(){
    struct C{double patch;const char*f;};
    C cases[]={{0.0,"/tmp/big_p0.txt"},{0.5,"/tmp/big_p5.txt"},{1.0,"/tmp/big_p10.txt"}};
    for(auto&c:cases){
        Params p{};
        p.city_w=720; p.city_h=720; p.block_w=52; p.block_d=52; p.base_height=99; // base ignored (fixed 9m)
        p.cbd_peak=28; p.cbd_decay=2.2e-5; p.cbd_aspect=1; p.cbd_angle=0;
        p.biz_inner_frac=0; p.biz_aspect=1;
        p.park_centrality=0.5; p.park_fraction=0.08; p.roughness=0.35; p.patchiness=c.patch;
        p.road_w_x=22; p.road_w_y=22; p.population_total=40000; p.wind_direction=0;
        p.buf_xn=p.buf_xp=p.buf_yn=p.buf_yp=40;
        p.Sx=p.buf_xn+p.city_w+p.buf_xp; p.Sy=p.buf_yn+p.city_h+p.buf_yp;
        double cx=p.buf_xn+p.city_w*0.5, cy=p.buf_yn+p.city_h*0.5;
        p.cbd_x=cx; p.cbd_y=cy; p.biz_center_x=cx; p.biz_center_y=cy;
        p.source_x=p.buf_xn-20; p.source_y=cy;
        Result r=generate(p);
        export_city(c.f,p,r);
        printf("patchiness=%.1f -> %s  park=%d biz=%d resHi=%d resLo=%d  pop=%.0f\n",
               c.patch,c.f,r.counts[0],r.counts[1],r.counts[2],r.counts[3],r.population);
    }
    return 0;
}
