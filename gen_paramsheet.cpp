// One city per parameter: a common baseline, then each panel varies ONE knob to a
// notable value so its effect is visible. Prints population/employment verification.
#include "city_builder7.h"
#include <cstdio>
#include <cmath>
#include <vector>
#include <string>
using namespace city;

static Params baseline(){
    Params p{};
    p.city_w=560; p.city_h=560; p.block_w=48; p.block_d=32; p.base_height=99; // base fixed(9m)
    p.cbd_peak=45; p.cbd_decay=1.2e-5; p.cbd_aspect=1; p.cbd_angle=0;
    p.biz_inner_frac=0; p.biz_aspect=1;
    p.park_centrality=0.5; p.park_fraction=0.10; p.roughness=0.25; p.patchiness=0.0;
    p.road_w_x=20; p.road_w_y=20; p.population_total=30000; p.wind_direction=0;
    p.buf_xn=p.buf_xp=p.buf_yn=p.buf_yp=40;
    p.Sx=p.buf_xn+p.city_w+p.buf_xp; p.Sy=p.buf_yn+p.city_h+p.buf_yp;
    double cx=p.buf_xn+p.city_w*0.5, cy=p.buf_yn+p.city_h*0.5;
    p.cbd_x=cx; p.cbd_y=cy; p.biz_center_x=cx; p.biz_center_y=cy;
    p.source_x=p.buf_xn-20; p.source_y=cy;
    return p;
}

struct Case{ const char* name; const char* file; const char* color; };

int main(){
    struct V{ const char* name; const char* file; const char* color; Params p; };
    std::vector<V> cases;
    auto add=[&](const char* n,const char* f,const char* col,Params p){ cases.push_back({n,f,col,p}); };

    add("baseline",             "/tmp/ps_base.txt","usage", baseline());
    { Params p=baseline(); p.block_w=72;        add("block_w = 72 (wide blocks)","/tmp/ps_bw.txt","usage",p); }
    { Params p=baseline(); p.block_d=40;        add("block_d = 40 (deep blocks)","/tmp/ps_bd.txt","usage",p); }
    { Params p=baseline(); p.road_w_x=8; p.road_w_y=8; add("street_width = 8 (narrow)","/tmp/ps_sw.txt","usage",p); }
    { Params p=baseline(); p.cbd_peak=120;      add("cbd_peak = 120 (tall core)","/tmp/ps_cp.txt","height",p); }
    { Params p=baseline(); p.cbd_decay=3e-5;    add("cbd_decay = 3e-5 (sharp core)","/tmp/ps_cd.txt","height",p); }
    { Params p=baseline(); p.roughness=0.8;     add("roughness = 0.8 (heterogeneous)","/tmp/ps_rg.txt","height",p); }
    { Params p=baseline(); p.patchiness=1.0;    add("patchiness = 1.0 (biz spread)","/tmp/ps_pt.txt","usage",p); }
    { Params p=baseline(); p.park_fraction=0.35;add("park_fraction = 0.35","/tmp/ps_pf.txt","usage",p); }
    { Params p=baseline(); p.park_centrality=0.1; add("park_centrality = 0.1 (edge ring)","/tmp/ps_pc.txt","usage",p); }
    { Params p=baseline(); p.park_centrality=0.9; add("park_centrality = 0.9 (central park)","/tmp/ps_pcc.txt","usage",p); }

    printf("%-32s  pop    (=30000?)  workers (=%.0f?)  Σcounts  maxH  park%%\n",
           "case", 30000.0*LABOR_RATE);
    printf("%s\n", std::string(96,'-').c_str());
    // manifest for the renderer
    FILE* mf=fopen("/tmp/ps_manifest.txt","w");
    for(auto& c:cases){
        Result r=generate(c.p);
        export_city(c.file,c.p,r);
        double workers = (r.worker_density>0)? r.biz_floor_area/r.worker_density : 0.0;
        int csum = r.counts[0]+r.counts[1]+r.counts[2]+r.counts[3]+r.counts[4];
        printf("%-32s  %6.0f     %8.0f       %4d/%-4d  %4.0f  p%d/b%d/rH%d/rL%d\n",
               c.name, r.population, workers, csum, r.num_blocks, r.max_height,
               r.counts[0],r.counts[1],r.counts[2],r.counts[3]);
        fprintf(mf,"%s|%s|%s\n", c.file, c.name, c.color);
    }
    fclose(mf);
    printf("\nlabor target = population × LABOR_RATE(%.2f); workers = biz_floor_area / worker_density\n", LABOR_RATE);
    return 0;
}
