// Generate a spread of example cities that each isolate a v8 parameter, so the
// 2D/3D renders make it obvious the knobs behave. Pure builder (no LBM); domain
// sized tight (no pow2) for visualization clarity only.
#include "city_builder7.h"
#include <cstdio>
#include <string>
using namespace city;

// Build a city with the common baseline, overriding the knobs of interest.
static Params base_params(double street_x, double street_y, double base_h,
                          double cbd_peak, double cbd_decay, double park_frac,
                          double block_w=48, double block_d=48){
    Params p{};
    p.city_w=480; p.city_h=480;
    p.block_w=block_w; p.block_d=block_d; p.base_height=base_h;
    p.cbd_peak=cbd_peak; p.cbd_decay=cbd_decay; p.cbd_aspect=1; p.cbd_angle=0;
    p.biz_inner_frac=0.05; p.biz_aspect=1;
    p.park_centrality=0.5; p.park_fraction=park_frac;
 p.roughness=0.15;
    p.road_w_x=street_x; p.road_w_y=street_y;
    p.population_total=20000; p.wind_direction=0;
    // Tight, non-pow2 domain for rendering (LBM not used here).
    p.buf_xn=p.buf_xp=p.buf_yn=p.buf_yp=40;
    p.Sx=p.buf_xn+p.city_w+p.buf_xp;
    p.Sy=p.buf_yn+p.city_h+p.buf_yp;
    double cx=p.buf_xn+p.city_w*0.5, cy=p.buf_yn+p.city_h*0.5;
    p.cbd_x=cx; p.cbd_y=cy; p.biz_center_x=cx; p.biz_center_y=cy;
    p.source_x=p.buf_xn-20; p.source_y=cy;
    return p;
}

struct Ex { const char* file; const char* label; Params p; };

int main(){
    std::vector<Ex> ex = {
        {"/tmp/ex_A.txt","A: baseline (street 20m, park 0.10)",
            base_params(20,20, 12,  40, 8e-6, 0.10)},
        {"/tmp/ex_B.txt","B: narrow streets 8m",
            base_params( 8, 8, 12,  40, 8e-6, 0.10)},
        {"/tmp/ex_C.txt","C: wide streets 40m",
            base_params(40,40, 12,  40, 8e-6, 0.10)},
        {"/tmp/ex_D.txt","D: anisotropic roads (x=40m, y=8m)",
            base_params(40, 8, 12,  40, 8e-6, 0.10)},
        {"/tmp/ex_E.txt","E: high park fraction 0.30",
            base_params(20,20, 12,  40, 8e-6, 0.30)},
        {"/tmp/ex_F.txt","F: tall concentrated CBD (peak 120m)",
            base_params(20,20, 10, 120, 2.5e-5, 0.08)},
    };
    FILE* idx=fopen("/tmp/ex_index.txt","w");
    for(auto& e: ex){
        Result r=generate(e.p);
        export_city(e.file, e.p, r);
        // quick numeric sanity to stdout + index for the renderer labels
        int npark=0; double sp=0,si=0;
        for(auto&b:r.blocks){ if(b.usage==PARK)npark++; sp+=b.eff_pop; si+=b.eff_inh; }
        printf("%-40s blocks=%3d parks=%2d maxH=%3.0fm pop=%.0f inh=%.0f\n",
               e.label, r.num_blocks, npark, r.max_height, sp, si);
        fprintf(idx,"%s\t%s\n", e.file, e.label);
    }
    fclose(idx);
    return 0;
}
