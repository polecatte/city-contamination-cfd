// gen_paramspace.cpp — build a city straight from the LIVE 9-D parameter space.
//
// Reads the 9 PHYSICAL parameters (already mapped from the [0,1]^9 unit cube by
// param_space.to_physical) on the command line, in the exact order of
// param_space.PARAM_SPACE, followed by an output filename. It wires those names
// into city_builder7::Params using the SAME conventions the production driver
// uses (street_width -> road_w_x == road_w_y; population fixed at 20,000; wind 0)
// and holds the FIXED run conditions from param_space.FIXED. Pure builder, no LBM.
//
//   argv: block_w block_d cbd_peak cbd_decay patchiness park_centrality \
//         park_fraction roughness street_width  out.txt
//
#include "city_builder7.h"
#include <cstdio>
#include <cstdlib>
#include <string>
using namespace city;

int main(int argc, char** argv){
    if(argc != 11){
        fprintf(stderr,
          "usage: %s block_w block_d cbd_peak cbd_decay patchiness "
          "park_centrality park_fraction roughness street_width out.txt\n", argv[0]);
        return 2;
    }
    double block_w        = atof(argv[1]);
    double block_d        = atof(argv[2]);
    double cbd_peak       = atof(argv[3]);
    double cbd_decay      = atof(argv[4]);
    double patchiness     = atof(argv[5]);
    double park_centrality= atof(argv[6]);
    double park_fraction  = atof(argv[7]);
    double roughness      = atof(argv[8]);
    double street_width   = atof(argv[9]);
    const char* out       = argv[10];

    Params p{};
    // ── FIXED run conditions (param_space.FIXED) ──
    p.city_w = 600.0; p.city_h = 600.0;
    p.population_total = 20000.0;
    p.wind_direction   = 0.0;               // wind_angle 0, streets grid-aligned
    p.base_height      = BASE_HEIGHT_M;     // ignored by height_at, set for ABI

    // ── The 9 live search parameters ──
    p.block_w = block_w;  p.block_d = block_d;
    p.cbd_peak = cbd_peak; p.cbd_decay = cbd_decay;
    p.patchiness = patchiness;
    p.park_centrality = park_centrality;
    p.park_fraction   = park_fraction;
    p.roughness       = roughness;
    // Isotropic grid: single street_width knob drives both road directions.
    p.road_w_x = p.road_w_y = street_width;

    // ── ABI-only / ignored circular-model fields ──
    p.cbd_aspect = 1; p.cbd_angle = 0; p.biz_inner_frac = 0.0; p.biz_aspect = 1;

    // ── Tight, non-pow2 buffers: visualization clarity only (LBM not run) ──
    p.buf_xn = p.buf_xp = p.buf_yn = p.buf_yp = 40.0;
    p.Sx = p.buf_xn + p.city_w + p.buf_xp;
    p.Sy = p.buf_yn + p.city_h + p.buf_yp;
    double cx = p.buf_xn + p.city_w*0.5, cy = p.buf_yn + p.city_h*0.5;
    p.cbd_x = cx; p.cbd_y = cy;
    p.biz_center_x = cx; p.biz_center_y = cy;
    p.source_x = p.buf_xn - 20; p.source_y = cy;

    Result r = generate(p);
    export_city(out, p, r);

    // Numeric sanity to stdout for the orchestrator / verification.
    int npark=0; for(auto&b:r.blocks) if(b.usage==PARK) npark++;
    printf("%s\tblocks=%d\tparks=%d\tmaxH=%.0f\tpop=%.0f\tbizfrac=%.2f\tparkfrac=%.2f\n",
           out, r.num_blocks, npark, r.max_height, r.population,
           r.biz_frac_actual, r.park_frac);
    return 0;
}
