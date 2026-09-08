// gen_rd_city.cpp — build a city whose zoning comes from a reaction–diffusion field.
//
// generate() lays out the block grid + morphology as usual; rezone() then assigns
// business to the Gray–Scott activator peaks and parks to its valleys, so the RD
// morphology is realised as actual buildings (heights, employment balance, parks,
// population all handled by the normal pipeline).
//
// Usage:
//   ./gen_rd_city block_w block_d cbd_peak cbd_decay patchiness park_fraction \
//                 roughness street_width  rd_F rd_k rd_aniso rd_scale rd_steps  out.txt
#include "city_zoning.h"
#include <cstdio>
#include <cstdlib>
using namespace city;

int main(int argc, char** argv) {
    if (argc != 15) {
        fprintf(stderr,
            "usage: %s block_w block_d cbd_peak cbd_decay patchiness park_fraction "
            "roughness street_width rd_F rd_k rd_aniso rd_scale rd_steps out.txt\n", argv[0]);
        return 2;
    }
    Params p{};
    p.city_w = 600.0; p.city_h = 600.0;
    p.population_total = 20000.0;
    p.wind_direction = 0.0;
    p.base_height    = BASE_HEIGHT_M;
    p.block_w   = atof(argv[1]);
    p.block_d   = atof(argv[2]);
    p.cbd_peak  = atof(argv[3]);
    p.cbd_decay = atof(argv[4]);
    p.patchiness    = atof(argv[5]);
    p.park_fraction = atof(argv[6]);
    p.roughness     = atof(argv[7]);
    p.road_w_x = p.road_w_y = atof(argv[8]);
    p.cbd_aspect=1; p.cbd_angle=0; p.biz_inner_frac=0; p.biz_aspect=1;
    p.park_centrality = 0.5;
    p.buf_xn = p.buf_xp = p.buf_yn = p.buf_yp = 40.0;
    p.Sx = p.buf_xn + p.city_w + p.buf_xp;
    p.Sy = p.buf_yn + p.city_h + p.buf_yp;
    double dom_cx = p.buf_xn + p.city_w*0.5, dom_cy = p.buf_yn + p.city_h*0.5;
    p.cbd_x=dom_cx; p.cbd_y=dom_cy; p.biz_center_x=dom_cx; p.biz_center_y=dom_cy;
    p.source_x=p.buf_xn-20; p.source_y=dom_cy;

    MixedZoningParams mzp{};
    mzp.rd_enable = true;
    mzp.rd_F      = atof(argv[9]);
    mzp.rd_k      = atof(argv[10]);
    mzp.rd_aniso  = atof(argv[11]);
    mzp.rd_scale  = atoi(argv[12]);
    mzp.rd_steps  = atoi(argv[13]);
    mzp.rd_seed   = 1u;
    mzp.biz_scatter = 0.0;   // pure ranking → business follows the activator peaks
    mzp.park_scatter = 0.0;  // pure ranking → parks fill the valleys contiguously
    const char* outfile = argv[14];

    p.patchiness = 0.0;
    Result r0 = generate(p);
    Result r  = rezone(p, r0, mzp);
    export_city(outfile, p, r);

    int npark=0; for (auto&b:r.blocks) if (b.usage==PARK) npark++;
    printf("%s\tblocks=%d\tparks=%d\tbiz=%d\tmaxH=%.0f\tpop=%.0f\n",
           outfile, r.num_blocks, npark, r.counts[1], r.max_height, r.population);
    return 0;
}
