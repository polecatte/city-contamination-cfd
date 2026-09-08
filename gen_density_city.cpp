// gen_density_city.cpp — build a city whose zoning comes from a density-field
// population model.
//
// generate() lays out the block grid; the density rezone then sets land use,
// heights, and the population/receptor distribution from an employment density
// field + a residential density field. Business locates and rises at the job
// peaks, residential density follows a monocentric gradient modulated by the
// jobs-housing mix, and parks fill the low-intensity valleys — so where the
// people are (the exposure weighting) and the built form move together.
//
// Usage:
//   ./gen_density_city block_w block_d roughness street_width park_fraction \
//        peak_height emp_centers emp_centrality emp_concentration \
//        res_gradient jh_mix  out.txt
#include "city_zoning.h"
#include <cstdio>
#include <cstdlib>
using namespace city;

int main(int argc, char** argv) {
    if (argc != 13) {
        fprintf(stderr,
            "usage: %s block_w block_d roughness street_width park_fraction "
            "peak_height emp_centers emp_centrality emp_concentration "
            "res_gradient jh_mix out.txt\n", argv[0]);
        return 2;
    }
    Params p{};
    p.city_w = 600.0; p.city_h = 600.0;
    p.population_total = 20000.0;
    p.wind_direction = 0.0;
    p.base_height    = BASE_HEIGHT_M;
    p.block_w   = atof(argv[1]);
    p.block_d   = atof(argv[2]);
    p.roughness = atof(argv[3]);
    p.road_w_x = p.road_w_y = atof(argv[4]);
    p.park_fraction = atof(argv[5]);
    // CBD height params unused by the density model (heights come from the fields),
    // but generate() needs sane values for its initial layout pass.
    p.cbd_peak = 40.0; p.cbd_decay = 8e-6; p.patchiness = 0.0;
    p.cbd_aspect=1; p.cbd_angle=0; p.biz_inner_frac=0; p.biz_aspect=1;
    p.park_centrality = 0.5;
    p.buf_xn = p.buf_xp = p.buf_yn = p.buf_yp = 40.0;
    p.Sx = p.buf_xn + p.city_w + p.buf_xp;
    p.Sy = p.buf_yn + p.city_h + p.buf_yp;
    double dom_cx = p.buf_xn + p.city_w*0.5, dom_cy = p.buf_yn + p.city_h*0.5;
    p.cbd_x=dom_cx; p.cbd_y=dom_cy; p.biz_center_x=dom_cx; p.biz_center_y=dom_cy;
    p.source_x=p.buf_xn-20; p.source_y=dom_cy;

    MixedZoningParams mzp{};
    mzp.dens_enable       = true;
    mzp.peak_height       = atof(argv[6]);
    mzp.emp_centers       = atoi(argv[7]);
    mzp.emp_centrality    = atof(argv[8]);
    mzp.emp_concentration = atof(argv[9]);
    mzp.res_gradient      = atof(argv[10]);
    mzp.jh_mix            = atof(argv[11]);
    mzp.biz_scatter  = 0.0;   // business follows the employment field contiguously
    mzp.park_scatter = 0.0;   // parks fill the valleys contiguously
    const char* outfile = argv[12];

    Result r0 = generate(p);
    Result r  = rezone(p, r0, mzp);
    export_city(outfile, p, r);

    int npark=0; for (auto&b:r.blocks) if (b.usage==PARK) npark++;
    printf("%s\tblocks=%d\tparks=%d\tbiz=%d\tmaxH=%.0f\tpop=%.0f\n",
           outfile, r.num_blocks, npark, r.counts[1], r.max_height, r.population);
    return 0;
}
