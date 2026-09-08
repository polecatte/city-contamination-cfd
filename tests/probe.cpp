// probe.cpp — read "key=val key=val ..." param lines on stdin, emit one CSV
// row of city-builder + voxelizer metrics per line. Used to characterize
// continuity and extreme behaviour of the generator for the BO chain.
#include "voxelize.h"   // pulls in city_builder7.h
#include <cstdio>
#include <sstream>
#include <string>
#include <map>
#include <cmath>
#include <iostream>
using namespace city;

int main(){
    // header
    printf("nblk,npark,nbiz,nhir,nmix,nlor,biz_frac,biz_floor,worker_dens,"
           "maxH,alpha,pop,target_pop,pop_err_pct,inhab,open_ratio,park_frac,"
           "nudges,nx,ny,nz,fluid_frac,indoor_inh_frac,"
           "block_w,block_d,base_height,cbd_peak,cbd_decay,biz_inner_frac,"
           "park_centrality,roughness,road_w_x,population_total\n");
    std::string line;
    while(std::getline(std::cin,line)){
        if(line.empty()) continue;
        std::map<std::string,double> kv;
        std::istringstream is(line); std::string tok;
        while(is>>tok){ auto e=tok.find('='); if(e==std::string::npos) continue;
            kv[tok.substr(0,e)]=std::stod(tok.substr(e+1)); }
        auto g=[&](const char*k,double d){ auto it=kv.find(k); return it==kv.end()?d:it->second; };

        Params p{};
        p.city_w=600; p.city_h=600;
        p.block_w=g("block_w",48); p.block_d=g("block_d",24);
        p.base_height=g("base_height",12);
        p.cbd_peak=g("cbd_peak",80); p.cbd_decay=g("cbd_decay",8e-6);
        p.cbd_aspect=1; p.cbd_angle=0;
        p.biz_inner_frac=g("biz_inner_frac",0.05); p.biz_aspect=1;
        p.park_centrality=g("park_centrality",0.5); p.park_fraction=g("park_fraction",0.10);
        p.roughness=g("roughness",0.15);
        p.road_w_x=g("road_w_x",20); p.road_w_y=g("road_w_y",20);
        p.population_total=g("population_total",20000);
        p.wind_direction=0;
        default_buffers(p);
        p.source_x=50; p.source_y=p.buf_yn+p.city_h*0.5;
        ensure_source_buffer(p); compute_domain(p);
        double cx=p.buf_xn+p.city_w*0.5, cy=p.buf_yn+p.city_h*0.5;
        p.cbd_x=cx; p.cbd_y=cy; p.biz_center_x=cx; p.biz_center_y=cy;

        Result r=generate(p);
        VoxelGrid vg=voxelize(p,r);
        size_t tot=vg.type.size(); size_t nfluid=0; double inh_total=0,inh_indoor=0;
        for(size_t i=0;i<tot;++i){ if(vg.type[i]==CELL_FLUID) nfluid++; inh_total+=vg.inh[i];
            if(vg.type[i]==CELL_INDOOR) inh_indoor+=vg.inh[i]; }
        double cw=p.city_w, ch=p.city_h;
        double target=p.population_total;
        double pop_err=(target>0)?(r.population-target)/target*100:0;

        printf("%d,%d,%d,%d,%d,%d,%.4f,%.0f,%.2f,%.1f,%.4f,%.1f,%.1f,%.3f,%.1f,"
               "%.4f,%.4f,%d,%d,%d,%d,%.4f,%.4f,"
               "%.3f,%.3f,%.3f,%.3f,%.3e,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.1f\n",
            r.num_blocks,r.counts[0],r.counts[1],r.counts[2],r.counts[3],r.counts[4],
            r.biz_frac_actual,r.biz_floor_area,r.worker_density,r.max_height,r.alpha,
            r.population,target,pop_err,r.total_inhabitance,r.open_space_ratio,r.park_frac,
            r.nudges,r.nx_cells,r.ny_cells,vg.nz,(double)nfluid/tot,
            (inh_total>0?inh_indoor/inh_total:0),
            p.block_w,p.block_d,p.base_height,p.cbd_peak,p.cbd_decay,p.biz_inner_frac,
            p.park_centrality,p.roughness,p.road_w_x,p.population_total);
    }
    return 0;
}
