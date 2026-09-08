// Export representative cities in their COST-732-compliant domains (uniform 15H
// buffer, foregoing power-of-2) so the actual domain borders can be drawn around
// the real city. H is learned from a first generate, then the domain is rebuilt.
#include "city_builder7.h"
#include <cstdio>
#include <string>
using namespace city;

static Params mkp(double sx,double sy,double base_h,double cbd_peak,double cbd_decay,
                  double park_frac,double buf){
    Params p{};
    p.city_w=480; p.city_h=480;
    p.block_w=48; p.block_d=48; p.base_height=base_h;
    p.cbd_peak=cbd_peak; p.cbd_decay=cbd_decay; p.cbd_aspect=1; p.cbd_angle=0;
    p.biz_inner_frac=0.05; p.biz_aspect=1;
    p.park_centrality=0.5; p.park_fraction=park_frac;
 p.roughness=0.15;
    p.road_w_x=sx; p.road_w_y=sy;
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
    struct C{const char*file;double sx,sy,bh,cp,cd,pf;};
    std::vector<C> cs = {
        {"/tmp/cost_A.txt", 20,20, 12, 40, 8e-6, 0.10},   // baseline
        {"/tmp/cost_F.txt", 20,20, 10,120, 2.5e-5,0.08},  // tall CBD
    };
    for(auto&c:cs){
        Params p0=mkp(c.sx,c.sy,c.bh,c.cp,c.cd,c.pf,200.0);
        Result r0=generate(p0);
        double H=r0.max_height, buf=15.0*H;
        Params p=mkp(c.sx,c.sy,c.bh,c.cp,c.cd,c.pf,buf);
        Result r=generate(p);
        export_city(c.file,p,r);
        printf("%s  H=%.0fm  buf=15H=%.0fm  domain=%.0fx%.0fm  Sz(6H)=%.0fm\n",
               c.file,H,buf,p.Sx,p.Sy,6*H);
    }
    return 0;
}
