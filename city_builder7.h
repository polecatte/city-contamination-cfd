#pragma once
// city_builder.h v7 — Per-building population/employment model
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cstdint>

namespace city {

// ── Physical constants ───────────────────────────────────────
// Residential: 35 m²/person floor area (AHS 2023 CBSA 35620, 50+ unit bldgs)
constexpr double RES_AREA_PP  = 35.0;   // m² floor area per resident
constexpr double RES_FLOOR_H  = 3.0;    // IBC §1208.2: 2.3m ceiling + 0.7m structure
// Baseline (peripheral) building height — FIXED (v8.2). It was a searched knob but
// largely redundant: its 6–24 m span is dwarfed by the CBD Gaussian tail, so it
// carried ~4% of morphology variance. Held at 3 residential floors (v8.3): this
// sits just below the RES_LOW 4-floor threshold, so the residential periphery —
// where the CBD tail is weak — resolves to LOW-DENSITY housing (25% lot coverage),
// giving a dense-core / low-density-ring gradient. (At 4 floors the CBD tail pushed
// everything to ≥5 floors and RES_LOW vanished.) The CBD peak rises above this.
constexpr double BASE_HEIGHT_M = 3 * RES_FLOOR_H;   // 9 m ≈ 3 floors
constexpr double DENS_RES     = 1.0/RES_AREA_PP;  // 0.0286 residents per m² floor area

// Low-density residential: 80 m²/person (Census 2024: median SF home 2,146 sqft / 2.5 persons)
constexpr double RES_LOW_AREA_PP = 80.0;
constexpr double DENS_RES_LOW    = 1.0/RES_LOW_AREA_PP;  // 0.0125 residents per m² floor
constexpr int    RES_LOW_MAX_FLOORS = 4;                   // max 4 floors (12m) for low-density
constexpr double RES_LOW_COVERAGE  = 0.25;                  // 25% lot coverage (Census 2024: 2,146 sqft home / 8,506 sqft lot)

// Business: 14 m²/worker floor area (GSA 150 USF/person, P100 2024)
constexpr double BIZ_AREA_PW  = 14.0;   // m² floor area per worker
constexpr double BIZ_FLOOR_H  = 4.0;    // IBC §1208.2: 2.3m ceiling + 1.7m plenum
constexpr double DENS_BIZ     = 1.0/BIZ_AREA_PW;  // 0.0714 workers per m² floor area

// Labor force: 47% of total population works (BLS CPS: 62% LFPR × 78% aged 16+)
constexpr double LABOR_RATE   = 0.47;

// Business blocks assigned iteratively in generate(): blocks converted from
// residential to business (center outward) until total business floor area
// meets the employment constraint: workers × BIZ_AREA_PW m² per worker.

// ── Microenvironment time budget — sourced from measured activity data ──────
// Source: Klepeis et al. (2001), "The National Human Activity Pattern Survey
// (NHAPS): a resource for assessing exposure to environmental pollutants,"
// J. Expo. Anal. Environ. Epidemiol. 11(3):231-252 (n=9386, EPA-funded, built
// to feed exposure models). 24-h population averages: residence-indoors 69%,
// other-indoor (workplace/commercial) ~18%, in-vehicle 5.5%, outdoors ~7.6%.
// (Corroborated by Canadian CHAPS-2: 88.9% indoor / 5.8% outdoor / 5.3%
// vehicle.) Time-of-day (diurnal) breakdown: Tsang & Klepeis (1996), EPA Final
// Report EPA/600/R-96/148 — see occupancy.h for the diurnal profiles.
//
// Mapping to this model's exposure pathways:
//   TIME_HOME   residence indoor  -> residential buildings (infiltration)   0.69
//   TIME_WORK   other indoor      -> business buildings    (infiltration)0.18
//   TIME_PARK   recreational share of outdoors -> porous parks              0.038
//   TIME_STREET in-vehicle + pedestrian outdoors -> ROAD CELLS (occupancy.h)0.092
// The outdoors 7.6% is split park/street by OUTDOOR_PARK_SHARE (=0.5); NHAPS
// does not subdivide "outdoors" into park vs sidewalk, so that split is an
// explicit assumption, not measured. Buildings+parks receive TIME_HOME+TIME_WORK
// +TIME_PARK (=0.908); the remaining TIME_STREET (=0.092) is distributed over
// the road network by occupancy.h (street occupants get full outdoor exposure).
constexpr double OUTDOOR_PARK_SHARE = 0.5;        // park vs sidewalk split of outdoors
constexpr double TIME_HOME    = 0.69;             // NHAPS residence-indoors
constexpr double TIME_WORK    = 0.18;             // NHAPS other-indoor (work/commercial)
constexpr double TIME_PARK    = 0.076 * OUTDOOR_PARK_SHARE;        // 0.038
constexpr double TIME_STREET  = 0.055 + 0.076 * (1.0 - OUTDOOR_PARK_SHARE); // 0.093 (occupancy.h)

// Geometry
constexpr double SLENDERNESS  = 7.0;
#ifndef CELL_SIZE_M
#define CELL_SIZE_M 4.0          // override at build time, e.g. -DCELL_SIZE_M=2.0
#endif
constexpr double CELL         = CELL_SIZE_M;
// Physical dimensions in METERS — snapped to cells inside generate()
constexpr double SETBACK_M=4.0;          // parcel margin (building inset from lot edge)
constexpr double ROAD_DEFAULT_M=20.0;    // fallback street width if a Params field is 0
constexpr double POP_TOL      = 0.01;
constexpr double MAX_PARK_FRAC= 0.60;  // sanity clamp on the park_fraction knob
constexpr double SQM_PER_ACRE = 4046.86;

// Permeability
constexpr float PERM_BIZ=0.005f, PERM_RES=0.010f;
constexpr float PERM_RES_LOW=0.020f; // houses: wood frame, more permeable than apartments
constexpr float PERM_PARK=0.800f;  // trees: very high porosity
constexpr double PARK_CANOPY_H=10.0; // meters, typical urban tree canopy

// Height-heterogeneity gain. Building heights are well modelled as LOG-NORMAL, so
// we perturb each block's height multiplicatively: h *= exp(ROUGH_GAIN·roughness·ξ),
// ξ = block_noise ∈ [−0.5,0.5] (std = 1/√12). ROUGH_GAIN = √12 makes the exponent's
// std equal `roughness`, i.e. roughness ≈ σ(ln h) ≈ the height coefficient of
// variation σ_H/H̄ — a physically interpretable, first-class morphology axis.
// Height heterogeneity is a primary control on urban ventilation: a uniform canopy
// gives skimming flow with weak vertical exchange, while staggered heights break
// the shear layer and enhance turbulent transport (Xie, Coceal & Castro 2008,
// Boundary-Layer Meteorol. 129:1; Nakayama, Takemi & Nagai 2011, J. Appl. Meteorol.
// Climatol. 50:1692). exp() keeps heights strictly positive (no clamp pile-up).
constexpr double ROUGH_GAIN = 3.4641016;   // = sqrt(12)

// ── Types ────────────────────────────────────────────────────
enum Usage { PARK=0, BUSINESS=1, RES_HIGH=2, RES_LOW=3 };  // MIXED removed (v8.1)

struct Params {
    double Sx, Sy;
    double block_w, block_d; // meters (snapped to cells internally)
    // Street widths (v8). Replaces the major/minor road system AND coverage: the
    // road grid is always aligned to the simulation grid, with all vertical roads
    // (running along y, width measured along x) of width road_w_x and all horizontal
    // roads (running along x, width measured along y) of width road_w_y. With block
    // size these set the canyon aspect H/W (Oke 1988) and the plan-area density λp =
    // block_area/(block+road)² (now a derived diagnostic, not an abstract knob).
    // Set road_w_x == road_w_y for an isotropic grid; differ them for directional
    // canyons (meaningful now that wind angle is varied against the fixed grid).
    double road_w_x, road_w_y;
    double base_height;       // FIXED at BASE_HEIGHT_M (v8.2) — kept for ABI, IGNORED by
                              // height_at (was redundant with the cbd_peak tail; ~4% variance).
    double cbd_x, cbd_y, cbd_peak, cbd_decay;
    double cbd_aspect;      // IGNORED (circular model) — kept for ABI
    double cbd_angle;       // IGNORED (circular model) — kept for ABI
    double biz_center_x, biz_center_y;
    double biz_inner_frac;    // FIXED to 0 (v8.3) — kept for ABI, IGNORED. Business is now
                              // a solid central circle (no void); spread is via `patchiness`.
    double biz_aspect;        // IGNORED (circular model) — kept for ABI
    double park_centrality;   // [0, 1] park placement bias (WHERE parks go)
    double park_fraction;     // [0, MAX_PARK_FRAC] fraction of blocks made parks
                              // (HOW MANY) — set directly; no open-space requirement.
    double roughness;         // [0, ~0.8] height heterogeneity ≈ std-dev of ln(height)
                              //  ≈ the height coefficient of variation σ_H/H̄. 0 = uniform
                              //  canopy (skimming flow); high = towers among low-rise.
    double patchiness;        // [0, 1] business-SPREAD (v8.3). 0 = tight concentric business
                              //  core; 1 = business follows a coherent noise field into
                              //  off-centre patches, spreading the work-population.
    // coverage RETIRED (v8): buildings now fill each lot to the setback line; plan
    // density is controlled by street width + block size, not an abstract fill ratio.
    double wind_direction;     // wind inflow angle RELATIVE to the grid-aligned street
                               // network (0=+x along streets; π/4=45° diagonal). By
                               // square symmetry the distinct range is [0, π/4].
    double source_x, source_y; // contaminant source position in meters
    double city_w, city_h;       // city footprint within domain (meters), 0=fill domain
    // FIXED total population (people), v8. Replaces target_density: population is no
    // longer hit by scaling building heights (which entangled the height and density
    // axes and drove the domain-NZ flip). Instead heights are pure morphology and
    // this fixed headcount is allocated across buildings by floor-area capacity in
    // generate() (Σ eff_pop ≡ population_total). Hold this constant across a sweep so
    // designs are compared at equal population (total exposure Σ w·C, not per-capita).
    double population_total;

    // Per-side buffer in meters. Set via default_buffers() or manually.
    // If any > 0, Sx/Sy are computed by compute_domain().
    // Set all to 0 for legacy centered behavior (Sx/Sy specified directly).
    double buf_xn, buf_xp;      // -x (left) and +x (right) buffers
    double buf_yn, buf_yp;      // -y (bottom) and +y (top) buffers
};

// ── Buffer helpers ──────────────────────────────────────────

// Standard buffer: 5× a typical urban max height. Adjust these if needed.
constexpr double BUF_REF_H  = 40.0;                  // reference max height (m)
constexpr double BUF_MULT   = 5.0;                    // multiplier
constexpr double BUF_DEFAULT = BUF_MULT * BUF_REF_H;  // 200m

// Uniform 200m buffer on all sides (the standard default)
inline void default_buffers(Params& p) {
    p.buf_xn = p.buf_xp = p.buf_yn = p.buf_yp = BUF_DEFAULT;
}

// Set buffers so that an external source has at least `pad` meters of clearance.
inline void ensure_source_buffer(Params& p, double pad = 20.0) {
    double city_xn = p.buf_xn;
    double city_xp = p.buf_xn + p.city_w;
    double city_yn = p.buf_yn;
    double city_yp = p.buf_yn + p.city_h;

    if (p.source_x < city_xn)
        p.buf_xn = std::max(p.buf_xn, (city_xn - p.source_x) + pad);
    if (p.source_x > city_xp)
        p.buf_xp = std::max(p.buf_xp, (p.source_x - city_xp) + pad);
    if (p.source_y < city_yn)
        p.buf_yn = std::max(p.buf_yn, (city_yn - p.source_y) + pad);
    if (p.source_y > city_yp)
        p.buf_yp = std::max(p.buf_yp, (p.source_y - city_yp) + pad);
}

// Next power of 2 >= n
inline int next_pow2(int n) {
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

// ── Minimal directional simulation domain (COST 732 / Tominaga et al. 2008) ──
// Size the domain to the bare minimum for the wind-direction SWEEP θ∈[0,π/4]
// (relative to the grid-aligned streets). Over that range the wind always has
// +x and +y components, so it enters from the −x/−y corner and leaves through the
// +x/+y corner; one FIXED domain must cover the whole sweep, so the inflow faces
// (−x,−y) take `inflow_H`·H and the outflow faces (+x,+y) take `outflow_H`·H.
// Standard guideline values: inflow_H≈5, outflow_H≈15 (the larger clearance is
// DOWNSTREAM, for wake recovery — Tominaga et al. 2008, JWEIA 96:1749; COST 732,
// Franke et al. 2007). H = tallest building height (m). Power-of-2 NOT enforced.
// Note: asymmetric buffers ⇒ the city is NOT centred; it sits toward the inflow
// corner with the extra room downstream (the intended CFD layout).
inline void minimal_domain(Params& p, double H, double inflow_H, double outflow_H){
    double cw = (p.city_w>0)?p.city_w:0.0, ch = (p.city_h>0)?p.city_h:0.0;
    // −x is always inflow, +x always outflow (wind keeps a +x component for θ∈[0,π/4]).
    p.buf_xn = inflow_H  * H;
    p.buf_xp = outflow_H * H;
    // ±y: only the OBLIQUE sweep (θ>0) makes +y part of the outflow corner. When the
    // wind is axis-aligned (θ=0, the only case currently run — wind_direction is
    // staged), ±y are LATERAL and need just inflow_H (5H), not 15H. This roughly
    // HALVES the y-extent and the cell count vs the corner sizing.
    bool oblique = (p.wind_direction > 1e-6);
    p.buf_yn = inflow_H * H;                              // −y lateral (inflow side)
    p.buf_yp = (oblique ? outflow_H : inflow_H) * H;     // +y: outflow corner only if oblique
    p.Sx = p.buf_xn + cw + p.buf_xp;                      // exact, no power-of-2
    p.Sy = p.buf_yn + ch + p.buf_yp;
    double ccx = p.buf_xn + cw*0.5, ccy = p.buf_yn + ch*0.5;
    p.cbd_x = ccx; p.cbd_y = ccy; p.biz_center_x = ccx; p.biz_center_y = ccy;
}
// Vertical cell count: 5H clear above the tallest building ⇒ domain height ≥6H.
// Raw (no power-of-2). Pass as nz_fixed to voxelize().
inline int nz_cost732(double H){ return std::max(2, (int)std::ceil(6.0*H/CELL)); }

// Compute Sx, Sy from buffers + city footprint. Snaps each axis to a
// power-of-2 cell count. Extra space goes to the +x / +y side.
// Call before generate().
inline void compute_domain(Params& p) {
    int raw_nx = (int)std::ceil((p.buf_xn + p.city_w + p.buf_xp) / CELL);
    int raw_ny = (int)std::ceil((p.buf_yn + p.city_h + p.buf_yp) / CELL);
    int nx = next_pow2(raw_nx);
    int ny = next_pow2(raw_ny);
    p.Sx = nx * CELL;
    p.Sy = ny * CELL;
}

struct Block {
    int x0,y0,x1,y1, bx0,by0,bx1,by1;
    int height_cells, max_height_cells;
    int n_floors;       // exact floor count (for population; height_cells is voxel approx)
    double center_x, center_y, footprint_m2, raw_height, dist_to_biz;
    int grid_ix, grid_iy; // position in block grid
    Usage usage;
    double eff_pop;      // residents who LIVE in this block
    double empl_cap;     // workers this block can HOLD (business floor area / BIZ_AREA_PW)
    double eff_inh;      // effective inhabitance (people present, time-averaged)
};

struct Result {
    std::vector<Block> blocks;
    double alpha, max_height, population, total_inhabitance;
    double biz_frac_actual;  // actual fraction of blocks that are business
    double biz_floor_area;   // total business floor area (m²)
    double worker_density;   // actual m² per worker (may differ from BIZ_AREA_PW)
    int counts[5], num_blocks, nudges;
    int nx_cells, ny_cells, num_blocks_x, num_blocks_y, offset_x, offset_y;
    double road_area_m2, park_area_m2, total_open_m2, open_space_ratio, park_frac;
    std::vector<bool> v_road_major, h_road_major;
};

// Skyline height field: a base height plus a radially-symmetric Gaussian "bump"
// peaking at the CBD:  h(x,y) = base + peak · exp(−decay · r²),  r² = (x−cx)²+(y−cy)².
// STRICTLY CIRCULAR (ring/central-core model): the earlier elliptical form
// (cbd_aspect/cbd_angle) is intentionally not used, so the tall-building core is
// concentric with the circular business core. (cbd_aspect/cbd_angle remain in
// Params for ABI but are ignored; restore the rotated/stretched r² here to revert.)
inline double height_at(const Params&p, double x, double y){
    double h=BASE_HEIGHT_M;                        // fixed baseline (p.base_height ignored)
    double dx=x-p.cbd_x, dy=y-p.cbd_y;
    double r2=dx*dx + dy*dy;
    h+=p.cbd_peak*std::exp(-p.cbd_decay*r2);
    return h;
}

inline double residents_per_m2(Usage u){
    if(u==RES_HIGH) return DENS_RES;                         // apartment tower: 1/35
    if(u==RES_LOW)  return DENS_RES_LOW;                     // house/walk-up: 1/80
    return 0; // business and park house nobody
}
inline double workers_per_m2(Usage u){
    if(u==BUSINESS) return DENS_BIZ;                         // all floors office
    return 0; // residential and park employ nobody
}

// Deterministic per-block "noise" in [−0.5, 0.5] from an integer hash (Knuth
// multiplicative + xorshift mixing). Reproducible jitter without a PRNG, used to
// break up the regular grid so zoning/heights look organic.
inline double block_noise(int idx){
    uint32_t h=(uint32_t)idx*2654435761u;
    h^=h>>13;h*=1274126177u;h^=h>>16;
    return((h&0xFFFF)/65535.0)-0.5;
}

// ── Deterministic COHERENT value noise for zoning patchiness ─────────────────
// Smooth (smoothstep-interpolated) hashed value lattice, a pure function of the
// block's PHYSICAL (x,y) in metres — so it is fully deterministic (same city ⇒
// same patches) and resolution-independent (warm-restart / optimizer safe). Used
// to SPREAD the business district: at patchiness=1 business follows this field
// into coherent off-centre patches instead of a single concentric core.
inline double hash01(int a,int b){
    uint32_t h=(uint32_t)(a*73856093) ^ (uint32_t)(b*19349663);
    h^=h>>13; h*=1274126177u; h^=h>>16;
    return (h & 0xFFFF)/65535.0;                    // [0,1]
}
inline double value_noise01(double x, double y, double scale){
    double fx=x/scale, fy=y/scale;
    int x0=(int)std::floor(fx), y0=(int)std::floor(fy);
    double tx=fx-x0, ty=fy-y0;
    tx=tx*tx*(3-2*tx); ty=ty*ty*(3-2*ty);           // smoothstep (C¹)
    double n00=hash01(x0,y0),   n10=hash01(x0+1,y0);
    double n01=hash01(x0,y0+1), n11=hash01(x0+1,y0+1);
    double nx0=n00+(n10-n00)*tx, nx1=n01+(n11-n01)*tx;
    return nx0+(nx1-nx0)*ty;                          // [0,1], spatially coherent
}
constexpr double PATCH_SCALE_M = 160.0;   // patch feature size ≈ 3–4 blocks

// ── Generate ─────────────────────────────────────────────────
inline Result generate(const Params& p){
    int Scx=(int)(p.Sx/CELL), Scy=(int)(p.Sy/CELL);
    // Snap physical dimensions to cell grid
    int bw=std::max(3,(int)std::round(p.block_w/CELL));
    int bd=std::max(3,(int)std::round(p.block_d/CELL));
    int ROAD_X=std::max(1,(int)std::round((p.road_w_x>0?p.road_w_x:ROAD_DEFAULT_M)/CELL));
    int ROAD_Y=std::max(1,(int)std::round((p.road_w_y>0?p.road_w_y:ROAD_DEFAULT_M)/CELL));
    int SETBACK=std::max(1,(int)std::round(SETBACK_M/CELL));
    // Grid-aligned street network: every vertical road (along y) is ROAD_X wide,
    // every horizontal road (along x) is ROAD_Y wide. No major/minor tiers.
    auto road_v=[ROAD_X](int){return ROAD_X;};
    auto road_h=[ROAD_Y](int){return ROAD_Y;};
    bool has_buf = (p.buf_xn > 0 || p.buf_xp > 0 || p.buf_yn > 0 || p.buf_yp > 0);
    // Fill the CITY footprint when buffered (else the whole domain). The last
    // block may abut the city edge with its trailing road falling in the buffer
    // (limit = city extent + one road), so the count is set by the city size, not
    // the domain — and centering below won't have to cull overhang asymmetrically.
    int cwx=(has_buf && p.city_w>0)?(int)std::round(p.city_w/CELL):Scx;
    int cwy=(has_buf && p.city_h>0)?(int)std::round(p.city_h/CELL):Scy;
    int limx=(has_buf && p.city_w>0)?(cwx+ROAD_X):Scx;
    int limy=(has_buf && p.city_h>0)?(cwy+ROAD_Y):Scy;
    // Compute block counts by accumulating variable pitch
    int nbx=0;
    {int x=road_v(0);
     while(x+bw+road_v(nbx+1)<=limx){x+=bw+road_v(nbx+1);nbx++;}
     nbx=std::max(1,nbx);}
    int nby=0;
    {int y=road_h(0);
     while(y+bd+road_h(nby+1)<=limy){y+=bd+road_h(nby+1);nby++;}
     nby=std::max(1,nby);}
    // Total grid extents
    int ux=0;for(int ix=0;ix<=nbx;++ix)ux+=road_v(ix); ux+=nbx*bw;
    int uy=0;for(int iy=0;iy<=nby;++iy)uy+=road_h(iy); uy+=nby*bd;
    int ox, oy, mg;
    if (has_buf) {
        ox = (int)std::round(p.buf_xn / CELL);
        oy = (int)std::round(p.buf_yn / CELL);
        mg = std::max(1, std::min({(int)(p.buf_xn/CELL), (int)(p.buf_xp/CELL),
                                   (int)(p.buf_yn/CELL), (int)(p.buf_yp/CELL)}));
    } else {
        ox = (Scx - ux) / 2;
        oy = (Scy - uy) / 2;
        mg = 5;
    }
    // Precompute positions
    std::vector<int> col_x(nbx), row_y(nby);
    {int x=ox+road_v(0);for(int ix=0;ix<nbx;++ix){col_x[ix]=x;x+=bw+road_v(ix+1);}}
    {int y=oy+road_h(0);for(int iy=0;iy<nby;++iy){row_y[iy]=y;y+=bd+road_h(iy+1);}}
    // Centre the grid on the city centre so the kept (post-cull) block set is
    // symmetric about the domain centre. This only SHIFTS positions — block count
    // is unchanged — and the CBD height field is anchored at the same centre, so
    // morphology stays aligned. (Left-anchoring left up to one pitch of margin on
    // only one side, biasing the city off-centre by ≤½ pitch.)
    if (has_buf) {
        int cxc = (int)std::round((p.buf_xn + (p.city_w>0?p.city_w:p.Sx)*0.5)/CELL);
        int cyc = (int)std::round((p.buf_yn + (p.city_h>0?p.city_h:p.Sy)*0.5)/CELL);
        int shx = cxc - (col_x.front()+col_x.back()+bw)/2;
        int shy = cyc - (row_y.front()+row_y.back()+bd)/2;
        for(auto&v:col_x) v+=shx;
        for(auto&v:row_y) v+=shy;
        ox+=shx; oy+=shy;
    }
    // No major/minor tiers anymore — kept (all false) for Result ABI / renderer.
    std::vector<bool> vmaj(nbx+1,false), hmaj(nby+1,false);
    double road_area_m2=((double)ux*uy-(double)nbx*nby*bw*bd)*CELL*CELL;

    // Generate blocks with coverage
    // v8: coverage retired — buildings fill each lot to the setback line (cov≡1).
    // cov kept as a local so the RES_LOW footprint-shrink math below is unchanged.
    double cov=1.0;
    double cs=std::sqrt(cov);
    std::vector<Block> blocks;
    for(int iy=0;iy<nby;++iy) for(int ix=0;ix<nbx;++ix){
        int bx0=col_x[ix], by0=row_y[iy];
        int bx1=bx0+bw, by1=by0+bd;
        if(bx0<mg||by0<mg||bx1>Scx-mg||by1>Scy-mg) continue;
        int x0f=bx0+SETBACK,y0f=by0+SETBACK,x1f=bx1-SETBACK,y1f=by1-SETBACK;
        if(x1f<=x0f||y1f<=y0f) continue;
        int bw_f=x1f-x0f,bd_f=y1f-y0f;
        int bw_c=std::max(1,(int)std::round(bw_f*cs));
        int bd_c=std::max(1,(int)std::round(bd_f*cs));
        // Snap parity so building centers symmetrically in available space
        if((bw_c&1)!=(bw_f&1)) bw_c=std::max(1,bw_c-1);
        if((bd_c&1)!=(bd_f&1)) bd_c=std::max(1,bd_c-1);
        int cx=(x0f+x1f)/2, cy=(y0f+y1f)/2;
        Block b;
        b.x0=cx-bw_c/2;b.x1=b.x0+bw_c;b.y0=cy-bd_c/2;b.y1=b.y0+bd_c;
        b.bx0=bx0;b.by0=by0;b.bx1=bx1;b.by1=by1;
        b.center_x=(bx0+bx1)*0.5*CELL;b.center_y=(by0+by1)*0.5*CELL;
        b.footprint_m2=(double)(b.x1-b.x0)*(b.y1-b.y0)*CELL*CELL;
        int md=std::min(b.x1-b.x0,b.y1-b.y0);
        b.max_height_cells=(int)(SLENDERNESS*md);
        b.raw_height=std::min(height_at(p,b.center_x,b.center_y),(double)b.max_height_cells*CELL);
        // Business-assignment metric. Base is the STRICTLY CIRCULAR radial distance
        // to the core (concentric CBD). `patchiness` blends this toward a coherent
        // value-noise field, so at patchiness→1 the "nearness" that wins a block its
        // business label follows off-centre patches instead of the centre — the
        // business district (and its work-population) SPREADS out. Deterministic and
        // resolution-independent (value_noise01 is a pure function of physical x,y).
        double dx=b.center_x-p.biz_center_x,dy=b.center_y-p.biz_center_y;
        double r=std::sqrt(dx*dx+dy*dy);
        double patch=std::max(0.0,std::min(1.0,p.patchiness));
        if(patch>0.0){
            double Rref=0.5*((p.city_w>0?p.city_w:p.Sx));           // characteristic radius
            double vn=value_noise01(b.center_x,b.center_y,PATCH_SCALE_M);  // [0,1] coherent
            r=(1.0-patch)*r + patch*vn*Rref;                        // blend distance ↔ noise
        }
        b.dist_to_biz=r;
        b.grid_ix=ix;b.grid_iy=iy;
        b.usage=RES_HIGH; // default
        // Cull blocks outside city footprint (if specified)
        double cw=p.city_w>0?p.city_w:p.Sx;
        double ch=p.city_h>0?p.city_h:p.Sy;
        double cx_dom = has_buf ? (p.buf_xn + cw*0.5) : (p.Sx*0.5);
        double cy_dom = has_buf ? (p.buf_yn + ch*0.5) : (p.Sy*0.5);
        if(std::abs(b.center_x-cx_dom)>cw*0.5 || std::abs(b.center_y-cy_dom)>ch*0.5)
            continue; // skip this block — outside city boundary
        blocks.push_back(b);
    }
    int N=(int)blocks.size();

    Result r;
    r.nx_cells=Scx;r.ny_cells=Scy;r.num_blocks_x=nbx;r.num_blocks_y=nby;
    r.offset_x=ox;r.offset_y=oy;r.num_blocks=N;r.nudges=0;r.road_area_m2=road_area_m2;
    for(int i=0;i<5;++i)r.counts[i]=0;
    if(N<3){r.alpha=0;r.max_height=0;r.population=0;r.blocks=blocks;
            r.park_frac=0;r.open_space_ratio=0;r.biz_frac_actual=0;return r;}

    double cw=p.city_w>0?p.city_w:p.Sx;
    double ch=p.city_h>0?p.city_h:p.Sy;
    // ── Step 1: Park count from the DIRECT park_fraction knob (v8) ──────
    // No open-space requirement / per-capita standard: park_fraction is the fraction
    // of city blocks made parks, set straight from the design vector and placed (not
    // merged) in Step 4. Clamped to [0, MAX_PARK_FRAC] for sanity only.
    double park_frac=std::max(0.0,std::min(MAX_PARK_FRAC,p.park_fraction));
    r.park_frac=park_frac;
    int n_park=(int)(N*park_frac+0.5);

    // ── Step 2: Sort blocks by distance to business center ──────
    std::vector<int> order(N);
    std::iota(order.begin(),order.end(),0);
    std::sort(order.begin(),order.end(),[&](int a,int b){
        return blocks[a].dist_to_biz<blocks[b].dist_to_biz;});

    // ── Step 3: Two-pass business assignment ────────────────
    // Pass 1: estimate with trial α → get biz count → binary search → get real α
    // Pass 2: re-assign with real α → final binary search
    int rs=0;   // no central void (v8.3): business fills from the lowest-metric block outward
                // (a solid central circle when patchiness=0). biz_inner_frac is ignored.
    auto assign_biz=[&](double trial_alpha) -> int {
        // Reset all non-park to residential
        for(int i=0;i<N;++i)
            if(blocks[i].usage==BUSINESS) blocks[i].usage=RES_HIGH;
        int nb=0;
        for(int iter=0;iter<200;++iter){
            double pop_est=0, biz_fa=0;
            for(int i=0;i<N;++i){auto&b=blocks[i];
                double h=b.raw_height*trial_alpha;
                h=std::max(CELL,std::min(h,(double)b.max_height_cells*CELL));
                if(b.usage==PARK)continue;
                if(b.usage==BUSINESS){
                    biz_fa+=b.footprint_m2*(h/BIZ_FLOOR_H);continue;
                }
                double fh=RES_FLOOR_H;double nf=h/fh;
                Usage eff_u=b.usage;double eff_fp=b.footprint_m2;
                if(eff_u==RES_HIGH&&(int)std::round(nf)<=RES_LOW_MAX_FLOORS){
                    eff_u=RES_LOW;eff_fp*=RES_LOW_COVERAGE/cov;
                }
                pop_est+=residents_per_m2(eff_u)*eff_fp*nf;
            }
            double biz_needed=pop_est*LABOR_RATE*BIZ_AREA_PW;
            if(biz_fa>=biz_needed*0.95)break;
            bool ok=false;
            for(int k=rs;k<N;++k){int idx=order[k];
                if(blocks[idx].usage!=RES_HIGH)continue;
                blocks[idx].usage=BUSINESS;nb++;ok=true;break;}
            if(!ok)break;
        }
        return nb;
    };
    // Pass 1: rough estimate with α=0.2
    int n_biz=assign_biz(0.2);

    // ── Step 4: Place parks — maximin dispersal with a smooth RADIAL centrality ──
    // Each park is placed at the block maximizing (spread-from-other-parks) plus a
    // centrality pull: park_centrality 0 → edge ring, 0.5 → evenly distributed,
    // 1 → central cluster. A position-hashed jitter (≪ block pitch) breaks the exact
    // grid ties symmetrically, so a symmetric city gets parks on all sides — the old
    // maximin+adjacency-clustering formed one-sided park WALLS (biased by block
    // aspect: left/right on wide blocks, top/bottom on tall ones).
    {
        double pc=std::max(0.0,std::min(1.0,p.park_centrality));
        std::vector<int> pool;
        for(int i=0;i<N;++i) if(blocks[i].usage==RES_HIGH||blocks[i].usage==RES_LOW) pool.push_back(i);
        int M=(int)pool.size();
        int np=std::min(n_park,M);
        double cx = has_buf ? (p.buf_xn + (p.city_w>0?p.city_w:p.Sx)*0.5) : (p.Sx*0.5);
        double cy = has_buf ? (p.buf_yn + (p.city_h>0?p.city_h:p.Sy)*0.5) : (p.Sy*0.5);
        auto distc=[&](int k){ auto&b=blocks[pool[k]];
            return std::sqrt((b.center_x-cx)*(b.center_x-cx)+(b.center_y-cy)*(b.center_y-cy)); };
        double Rref=1.0; for(int k=0;k<M;++k) Rref=std::max(Rref,distc(k));
        const double CEN_GAIN=3.5;                 // centrality strength vs dispersion
        std::vector<double> min_dp(M,1e18);
        for(int placed=0; placed<np; ++placed){
            int best=-1; double bs=-1e18;
            for(int k=0;k<M;++k){
                if(blocks[pool[k]].usage==PARK) continue;
                double disp = (placed==0)? 0.0 : min_dp[k];             // spread from existing parks
                double cen  = CEN_GAIN*(pc-0.5)*(Rref-distc(k));        // +central (pc>.5) / +edge (pc<.5)
                double jit  = 8.0*hash01(blocks[pool[k]].grid_ix, blocks[pool[k]].grid_iy);
                double score= disp + cen + jit;
                if(score>bs){bs=score;best=k;}
            }
            if(best<0)break;
            blocks[pool[best]].usage=PARK;
            auto&pb=blocks[pool[best]];
            for(int k=0;k<M;++k){ if(blocks[pool[k]].usage==PARK)continue; auto&b=blocks[pool[k]];
                double d=std::sqrt((b.center_x-pb.center_x)*(b.center_x-pb.center_x)
                                  +(b.center_y-pb.center_y)*(b.center_y-pb.center_y));
                min_dp[k]=std::min(min_dp[k],d); }
            min_dp[best]=0;
        }
    }
    // ── Step 5: Fix park footprints (no setback) ─────────────
    for(auto&b:blocks) if(b.usage==PARK){
        b.x0=b.bx0;b.y0=b.by0;b.x1=b.bx1;b.y1=b.by1;
        b.footprint_m2=(double)(b.x1-b.x0)*(b.y1-b.y0)*CELL*CELL;
    }

    // ── Steps 5b/5c/6 REMOVED (v8/8.1): no park merging or open-space shedding
    // (parks are direct block-sized parcels from Step 4), and mixed-use zoning is
    // retired. Blocks are purely PARK / BUSINESS / RES_HIGH / RES_LOW.

    // ── Step 7: Heights are PURE MORPHOLOGY (v8 — no population-driven scaling) ──
    // height_at() = BASE_HEIGHT_M + cbd_peak·exp(−cbd_decay·r²), with deterministic
    // log-normal per-block roughness. Total population is the FIXED input
    // p.population_total, allocated across buildings by floor-area capacity in Step 9.
    // This decouples the height axis from the density axis (v7 solved a global height
    // multiplier α to hit a density target; that is retired, r.alpha≡1).
    double rn=std::max(0.0,std::min(0.8,p.roughness));   // ≈ height CoV; up to very heterogeneous
    r.alpha=1.0;                 // morphology heights used as-is (field kept for ABI/reporting)

    // Re-assign business with final (morphology) heights — Pass 2, now at α=1.
    n_biz=assign_biz(r.alpha);
    r.biz_frac_actual=(double)n_biz/std::max(1,N);

    // ── Step 8: Snap morphology heights to whole floors ──────────
    for(int ki=0;ki<N;++ki){auto&b=blocks[ki];
        if(b.usage<5)r.counts[b.usage]++;
        if(b.usage==PARK){
            b.height_cells=std::max(1,(int)std::round(PARK_CANOPY_H/CELL));
            b.n_floors=0;b.eff_pop=0;b.empl_cap=0;b.eff_inh=0;
        }else{
            double raw_h=b.raw_height*std::exp(ROUGH_GAIN*rn*block_noise(ki)); // log-normal height scatter (α≡1)
            raw_h=std::max(CELL,std::min(raw_h,(double)b.max_height_cells*CELL));
            double fh=(b.usage==BUSINESS)?BIZ_FLOOR_H:RES_FLOOR_H;
            b.n_floors=std::max(1,(int)std::round(raw_h/fh));
            // Height-driven: short residential → low-density (houses with yards)
            if(b.usage==RES_HIGH && b.n_floors<=RES_LOW_MAX_FLOORS){
                r.counts[RES_HIGH]--; b.usage=RES_LOW; r.counts[RES_LOW]++;
                double scale=std::sqrt(RES_LOW_COVERAGE/cov);
                int cur_w=b.x1-b.x0, cur_d=b.y1-b.y0;
                int new_w=std::max(2,(int)(cur_w*scale));
                int new_d=std::max(2,(int)(cur_d*scale));
                int ccx=(b.x0+b.x1)/2, ccy=(b.y0+b.y1)/2;
                b.x0=ccx-new_w/2; b.x1=b.x0+new_w;
                b.y0=ccy-new_d/2; b.y1=b.y0+new_d;
                b.footprint_m2=(double)(b.x1-b.x0)*(b.y1-b.y0)*CELL*CELL;
                int md_lo=std::min(b.x1-b.x0,b.y1-b.y0);
                b.max_height_cells=(int)(SLENDERNESS*md_lo);
                if(b.height_cells>b.max_height_cells) b.height_cells=b.max_height_cells;
            }
            double phys_h=b.n_floors*fh;
            b.height_cells=std::max(1,(int)std::round(phys_h/CELL));
            b.height_cells=std::min(b.height_cells,b.max_height_cells);
            // CAPACITY weight (not final headcount): residents/workers the floor area
            // would hold at standard densities. Rescaled to the fixed total in Step 9.
            double fa=b.footprint_m2*b.n_floors;
            b.eff_pop=residents_per_m2(b.usage)*fa;
            b.empl_cap=workers_per_m2(b.usage)*fa;
            b.eff_inh=0;
        }
    }

    // ── Step 9: Employment balance (land-use mix), then allocate FIXED population ──
    // (a) Balance business vs residential floor area to the labor force implied by
    // the fixed total population (workers_needed = P_total·LABOR_RATE). This is a
    // land-use decision about the biz/res MIX, independent of the population SCALE,
    // and is carried over unchanged from v7 (center-out swaps at the biz boundary).
    double P_total=std::max(0.0,p.population_total);
    {
        auto sum_empl=[&]{double s=0;for(auto&b:blocks)s+=b.empl_cap;return s;};
        double workers_needed=P_total*LABOR_RATE;
        double total_empl=sum_empl();
        double ratio=total_empl/std::max(1.0,workers_needed);
        int swaps=0;
        while(ratio<0.9 && swaps<20){          // too few offices: res→biz, center-out
            int best=-1;double bdst=1e18;
            for(int i=0;i<N;++i){if(blocks[i].usage!=RES_HIGH)continue;
                if(blocks[i].dist_to_biz<bdst){bdst=blocks[i].dist_to_biz;best=i;}}
            if(best<0)break; auto&b=blocks[best];
            double fa=b.footprint_m2*b.n_floors;
            b.usage=BUSINESS;r.counts[2]--;r.counts[1]++;
            b.eff_pop=0;b.empl_cap=DENS_BIZ*fa;
            total_empl=sum_empl();ratio=total_empl/std::max(1.0,workers_needed);swaps++;
        }
        while(ratio>1.3 && swaps<20){          // too many offices: biz→res, edge-in
            int best=-1;double bdst=0;
            for(int i=0;i<N;++i){if(blocks[i].usage!=BUSINESS)continue;
                if(blocks[i].dist_to_biz>bdst){bdst=blocks[i].dist_to_biz;best=i;}}
            if(best<0)break; auto&b=blocks[best];
            double fa=b.footprint_m2*b.n_floors;
            b.usage=RES_HIGH;r.counts[1]--;r.counts[2]++;
            b.eff_pop=residents_per_m2(RES_HIGH)*fa;b.empl_cap=0;
            total_empl=sum_empl();ratio=total_empl/std::max(1.0,workers_needed);swaps++;
        }
        r.nudges=swaps;   // now counts land-use swaps, not height nudges
    }
    // (b) Dasymetric allocation: distribute the FIXED population across residential
    // capacity so Σ eff_pop ≡ P_total exactly. capacity_i = residents_per_m2(usage)·
    // footprint·floors keeps apartment-vs-house density realism in the DISTRIBUTION
    // while the TOTAL is pinned. (Floor-area / volume dasymetric mapping: Mennis,
    // Prof. Geographer 55(1):31–42, 2003; building occupant-load basis: IBC §1004.)
    double cap_sum=0; for(auto&b:blocks) cap_sum+=b.eff_pop;
    double pop=P_total;
    if(cap_sum>0) for(auto&b:blocks) b.eff_pop*=P_total/cap_sum;
    r.population=pop;

    // ── Step 10: Effective inhabitance from the 24-h NHAPS time budget ──────────
    // Distribution logic unchanged from v7; the SCALE is now the fixed P_total. The
    // building+park inhabitance sums to (TIME_HOME+TIME_WORK+TIME_PARK)·P_total; the
    // remaining TIME_STREET·P_total is placed on road cells by occupancy.h. Because
    // the NHAPS fractions partition the day (Σ ≈ 1.001), the receptor weight w over
    // ALL microenvironments sums to ≈P_total — the per-capita-vs-total question is
    // settled by holding P_total fixed, so no extra normalization is applied.
    r.max_height=0;r.park_area_m2=0;r.total_inhabitance=0;
    double total_res_fa=0,total_biz_fa=0,total_park_area=0;
    for(auto&b:blocks){
        if(b.height_cells*CELL>r.max_height)r.max_height=b.height_cells*CELL;
        double fa=b.footprint_m2*b.n_floors;
        if(b.usage==PARK){r.park_area_m2+=b.footprint_m2;total_park_area+=b.footprint_m2;}
        else if(b.usage==RES_HIGH||b.usage==RES_LOW) total_res_fa+=fa;
        else if(b.usage==BUSINESS) total_biz_fa+=fa;
    }
    double home_people=pop*TIME_HOME, work_people=pop*TIME_WORK, park_people=pop*TIME_PARK;
    for(auto&b:blocks){
        double fa=b.footprint_m2*b.n_floors; b.eff_inh=0;
        if(b.usage==PARK) b.eff_inh=(total_park_area>0)?park_people*(b.footprint_m2/total_park_area):0;
        else if(b.usage==RES_HIGH||b.usage==RES_LOW) b.eff_inh=(total_res_fa>0)?home_people*(fa/total_res_fa):0;
        else if(b.usage==BUSINESS) b.eff_inh=(total_biz_fa>0)?work_people*(fa/total_biz_fa):0;
        r.total_inhabitance+=b.eff_inh;
    }

    // Business capacity metrics
    r.biz_floor_area=total_biz_fa;
    double n_workers=r.population*LABOR_RATE;
    r.worker_density=(n_workers>0)?total_biz_fa/n_workers:0;
    r.total_open_m2=r.park_area_m2;
    r.open_space_ratio=(r.total_open_m2/SQM_PER_ACRE)/(r.population/1000.0);
    r.v_road_major=vmaj;r.h_road_major=hmaj;
    // Final recount and metrics
    for(int i=0;i<5;i++) r.counts[i]=0;
    r.num_blocks=(int)blocks.size();
    for(auto&b:blocks) if(b.usage>=0&&b.usage<5) r.counts[b.usage]++;
    r.biz_frac_actual=(double)r.counts[1]/std::max(1,r.num_blocks);
    r.biz_floor_area=total_biz_fa;
    double n_workers_f=r.population*LABOR_RATE;
    r.worker_density=(n_workers_f>0)?r.biz_floor_area/n_workers_f:0;
    r.blocks=std::move(blocks);
    return r;
}

inline void export_city(const char*fn,const Params&p,const Result&r){
    FILE*f=fopen(fn,"w");
    fprintf(f,"CELL %.1f\nDOMAIN %d %d\nSIZE %.0f %.0f\n",CELL,r.nx_cells,r.ny_cells,p.Sx,p.Sy);
    int bw_e=std::max(3,(int)std::round(p.block_w/CELL));
    int bd_e=std::max(3,(int)std::round(p.block_d/CELL));
    // v8: rmaj/rmin slots now carry the vertical/horizontal road widths (cells);
    // the old MAJ_INTERVAL slot is fixed at 1 (no major/minor tiers).
    int rwx_e=std::max(1,(int)std::round((p.road_w_x>0?p.road_w_x:ROAD_DEFAULT_M)/CELL));
    int rwy_e=std::max(1,(int)std::round((p.road_w_y>0?p.road_w_y:ROAD_DEFAULT_M)/CELL));
    int sb_e=(int)std::round(SETBACK_M/CELL);
    fprintf(f,"GRID %d %d %d %d %d %d %d %d %d %d\n",bw_e,bd_e,rwx_e,rwy_e,sb_e,
            r.offset_x,r.offset_y,r.num_blocks_x,r.num_blocks_y,1);
    fprintf(f,"STATS %.4f %.1f %.0f %d %d %d %d %d %d %d\n",r.alpha,r.max_height,r.population,r.num_blocks,
            r.counts[0],r.counts[1],r.counts[2],r.counts[3],r.counts[4],r.nudges);
    fprintf(f,"BUILDINGS %d\n",r.num_blocks);
    for(auto&b:r.blocks) fprintf(f,"B %d %d %d %d %d %d %d %d %d %d %.1f %.1f\n",
        b.x0,b.y0,b.x1,b.y1,b.bx0,b.by0,b.bx1,b.by1,b.height_cells,(int)b.usage,b.eff_pop,b.eff_inh);
    fclose(f);
}

inline void print_summary(const Params&p,const Result&r){
    const char*ulbl[]={"Park","Biz","Hi-R","Mix","Lo-R"};
    printf("  Domain: %.0f×%.0fm (%d×%d cells)\n",p.Sx,p.Sy,r.nx_cells,r.ny_cells);
    if (p.buf_xn > 0 || p.buf_xp > 0 || p.buf_yn > 0 || p.buf_yp > 0)
        printf("  Buffers: -x=%.0f +x=%.0f -y=%.0f +y=%.0fm\n",
               p.buf_xn, p.buf_xp, p.buf_yn, p.buf_yp);
    printf("  Blocks: %d —",r.num_blocks);
    for(int i=0;i<5;++i) if(r.counts[i]) printf(" %d %s/%.0f%%",r.counts[i],ulbl[i],r.counts[i]*100.0/r.num_blocks);
    printf("\n  Biz: %.0f%% of blocks, %.0f m² floor area, %.1f m²/worker (GSA std: %.0f)\n",
           r.biz_frac_actual*100,r.biz_floor_area,r.worker_density,BIZ_AREA_PW);
    // v8: population is a FIXED input allocated by floor area (no density target / α).
    printf("  Heights: max %.0fm (pure morphology) | Pop: %.0f (fixed), %d land-use swaps\n",
           r.max_height,r.population,r.nudges);
    printf("  Inhabitance: %.0f (%.1f%% of pop) | Open space: %.2f ac/1k\n",
           r.total_inhabitance,r.total_inhabitance/std::max(1.0,r.population)*100,r.open_space_ratio);
}

} // namespace city
