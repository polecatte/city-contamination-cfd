#include "city_builder7.h"
#include <cstdio>
using namespace city;
int main(){
  double pcs[]={0.1,0.5,0.9}; const char* fs[]={"/tmp/park_e.txt","/tmp/park_d.txt","/tmp/park_c.txt"};
  for(int i=0;i<3;++i){ Params p{}; p.city_w=560;p.city_h=560;p.block_w=44;p.block_d=44;p.base_height=99;
    p.cbd_peak=45;p.cbd_decay=1.2e-5;p.cbd_aspect=1;p.biz_inner_frac=0;p.biz_aspect=1;
    p.park_centrality=pcs[i];p.park_fraction=0.16;p.roughness=0.25;p.patchiness=0;
    p.road_w_x=20;p.road_w_y=20;p.population_total=30000;p.wind_direction=0;
    p.buf_xn=p.buf_xp=p.buf_yn=p.buf_yp=40;p.Sx=640;p.Sy=640;
    double c=320;p.cbd_x=c;p.cbd_y=c;p.biz_center_x=c;p.biz_center_y=c;p.source_x=20;p.source_y=c;
    Result r=generate(p); export_city(fs[i],p,r);
    printf("park_centrality=%.1f -> %s (parks=%d, pop=%.0f, workers=%.0f)\n",pcs[i],fs[i],
           r.counts[0],r.population,r.biz_floor_area/(r.worker_density>0?r.worker_density:1)); }
  return 0;
}
