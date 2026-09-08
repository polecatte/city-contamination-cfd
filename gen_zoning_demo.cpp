// gen_zoning_demo.cpp — command-line driver for city_zoning.h.
//
// Calls generate() to lay out the block grid + morphology, then calls
// rezone() with the mixed potential field. The old builder is untouched.
//
// Usage:
//   ./gen_zoning_demo \
//       block_w block_d cbd_peak cbd_decay patchiness park_fraction roughness street_width \
//       w_radial w_grad_x w_grad_y w_bipeak_x w_bipeak_y w_corridor corridor_angle_deg \
//       biz_scatter \
//       park_centrality park_w_grad_x park_w_grad_y park_w_bipeak_x park_w_bipeak_y \
//       park_w_corridor park_corridor_angle_deg park_scatter \
//       out.txt
//
// Parks now use the SAME designed potential-field basis as business (radial via
// park_centrality, gradient, twin-peak, corridor) plus park_scatter (0=one
// contiguous green mass, 1=maximin-dispersed pocket parks). The old periodic
// stripe parameters are gone.
//
#include "city_zoning.h"
#include <cstdio>
#include <cstdlib>
using namespace city;

int main(int argc, char** argv) {
    if (argc != 26) {
        fprintf(stderr,
            "usage: %s "
            "block_w block_d cbd_peak cbd_decay patchiness park_fraction roughness street_width "
            "w_radial w_grad_x w_grad_y w_bipeak_x w_bipeak_y w_corridor corridor_angle_deg "
            "biz_scatter "
            "park_centrality park_w_grad_x park_w_grad_y park_w_bipeak_x park_w_bipeak_y "
            "park_w_corridor park_corridor_angle_deg park_scatter "
            "out.txt\n", argv[0]);
        return 2;
    }

    // ── City morphology params ────────────────────────────────────────────
    Params p{};
    p.city_w = 600.0; p.city_h = 600.0;
    p.population_total = 20000.0;
    p.wind_direction = 0.0;
    p.base_height    = BASE_HEIGHT_M;
    p.block_w   = atof(argv[1]);
    p.block_d   = atof(argv[2]);
    p.cbd_peak  = atof(argv[3]);
    p.cbd_decay = atof(argv[4]);
    p.patchiness     = atof(argv[5]);
    p.park_fraction  = atof(argv[6]);
    p.roughness      = atof(argv[7]);
    p.road_w_x = p.road_w_y = atof(argv[8]);
    p.cbd_aspect=1; p.cbd_angle=0; p.biz_inner_frac=0; p.biz_aspect=1;
    p.park_centrality = 0.5;  // overridden by MixedZoningParams below
    p.buf_xn = p.buf_xp = p.buf_yn = p.buf_yp = 40.0;
    p.Sx = p.buf_xn + p.city_w + p.buf_xp;
    p.Sy = p.buf_yn + p.city_h + p.buf_yp;
    double dom_cx = p.buf_xn + p.city_w*0.5;
    double dom_cy = p.buf_yn + p.city_h*0.5;
    p.cbd_x=dom_cx; p.cbd_y=dom_cy;
    p.biz_center_x=dom_cx; p.biz_center_y=dom_cy;
    p.source_x=p.buf_xn-20; p.source_y=dom_cy;

    // ── Mixed zoning params ───────────────────────────────────────────────
    MixedZoningParams mzp{};
    mzp.w_radial    = atof(argv[9]);
    mzp.w_grad_x    = atof(argv[10]);
    mzp.w_grad_y    = atof(argv[11]);
    mzp.w_bipeak_x  = atof(argv[12]);
    mzp.w_bipeak_y  = atof(argv[13]);
    mzp.w_corridor  = atof(argv[14]);
    mzp.corridor_angle       = atof(argv[15]) * M_PI / 180.0; // deg → rad
    mzp.biz_scatter          = atof(argv[16]);
    mzp.park_centrality      = atof(argv[17]);
    mzp.park_w_grad_x        = atof(argv[18]);
    mzp.park_w_grad_y        = atof(argv[19]);
    mzp.park_w_bipeak_x      = atof(argv[20]);
    mzp.park_w_bipeak_y      = atof(argv[21]);
    mzp.park_w_corridor      = atof(argv[22]);
    mzp.park_corridor_angle  = atof(argv[23]) * M_PI / 180.0; // deg → rad
    mzp.park_scatter         = atof(argv[24]);
    const char* outfile       = argv[25];

    // ── Step 1: Original generate() — block grid + morphology ────────────
    p.patchiness = 0.0;
    Result r0 = generate(p);

    // ── Step 2: rezone() — re-assign business and parks with mixed Φ ─────
    Result r = rezone(p, r0, mzp);

    // ── Export ────────────────────────────────────────────────────────────
    export_city(outfile, p, r);

    int npark=0; for(auto&b:r.blocks) if(b.usage==PARK) npark++;
    printf("%s\tblocks=%d\tparks=%d\tbiz=%d\tmaxH=%.0f\tpop=%.0f\n",
           outfile, r.num_blocks, npark, r.counts[1], r.max_height, r.population);
    return 0;
}
