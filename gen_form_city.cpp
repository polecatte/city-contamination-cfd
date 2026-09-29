// gen_form_city.cpp — built-form-only city for the two-tier design (TWO_TIER_DESIGN.md).
//
//   g++ -O3 -std=c++17 -I. -DCELL_SIZE_M=4.0 gen_form_city.cpp -o gen_form_city
//   BLOCK_SIZE=36 STREET_W=12 ... OUT_DIR=geom_form ./gen_form_city
//
// The outer level of the design fixes geometry only. Nobody lives anywhere yet: population
// never sets a building's height, and every building is a use-neutral shell (one floor
// height) whose use (home or work) and occupancy the inner integer programme chooses later.
// Heights are shaped by the form parameters and then scaled so that the total floor area
// equals the fixed GFA = FAR x city area, so no design wins by building more or less.
//
// Outputs (OUT_DIR, the repo's 5-int header [nx,ny,nz,dx*1000,ncomp] + payload, z-major):
//   material_map.dat  int32  OpenLB materials (openlb_geometry.h; inlet on the x=0 face)
//   source_mask.u8    uint8  release cells: one per plan cell over city + environs, at ground
//                            level, or on the roof where a building stands, so the release is
//                            uniform per unit area and independent of the design
//   release_zone.u8   uint8  release tile 1..K (ZONES x ZONES over city + environs), 0 elsewhere
//   envelope.i32      int32  building id + 1 on the fluid cells touching that building's walls
//                            or roof (where its occupants' outdoor air is sampled), 0 elsewhere
//   dep_vel.f32       float  dry-deposition velocity of each surface cell for particle DEP_DP
//   geom_type.u8      uint8  raw voxel types (viz)
//   buildings.csv            one row per building: footprint, floors, floor area, capacities
//   meta_geom.txt            parameters, achieved GFA, population, domain
// Exit 0 = written; 3 = the design cannot hold the fixed GFA (reported, nothing written).

#include "openlb_geometry.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
using namespace city;

static double envd(const char* k, double d){ const char* e=getenv(k); return e? atof(e):d; }
static int    envi(const char* k, int    d){ const char* e=getenv(k); return e? atoi(e):d; }

// ── fixed ground rules (TWO_TIER_DESIGN.md) ──
constexpr double FLOOR_H_M   = 3.5;    // one floor height for every building: capacity is use-neutral
constexpr int    MIN_FLOORS  = 2;
constexpr double AREA_PP_RES = 35.0;   // m2 per resident  (city_builder7.h RES_AREA_PP)
constexpr double AREA_PP_JOB = 14.0;   // m2 per job       (city_builder7.h BIZ_AREA_PW)
constexpr double LABOR       = 0.47;   // jobs per resident (city_builder7.h LABOR_RATE)
constexpr double PARK_CANOPY_M = 10.0; // porous tree canopy height (city_builder7.h)

struct Bldg { int i, j; double X0, X1, Y0, Y1; int floors = 0, fmax = 0; double w = 0;
              int x0, x1, y0, y1, hc; long nenv = 0; };

// deterministic per-slot uniform in [0,1)
static double hash01(int a, int b, int seed) {
    uint64_t h = (uint64_t)(a * 73856093) ^ (uint64_t)(b * 19349663) ^ (uint64_t)(seed * 83492791);
    h ^= h >> 33; h *= 0xff51afd7ed558ccdULL; h ^= h >> 33; h *= 0xc4ceb9fe1a85ec53ULL; h ^= h >> 33;
    return (double)(h >> 11) / (double)(1ULL << 53);
}

// Positions of blocks along one axis (metres from the city edge): n blocks of width bw,
// ordinary streets st, and nc of the streets nearest the centre widened to corridor width cw.
static std::vector<std::pair<double,double>> axis_blocks(double L, double bw, double st, int nc, double cw) {
    int n = (int)std::floor((L + st) / (bw + st));
    auto width = [&](int n) { int k = std::min(nc, std::max(0, n - 1)); return n * bw + (n - 1 - k) * st + k * cw; };
    while (n > 1 && width(n) > L) --n;
    std::vector<double> gap(std::max(0, n - 1), st);
    // widen the gaps nearest the centre (1 corridor: middle gap; 2: the gaps at 1/3 and 2/3)
    for (int k = 0; k < std::min(nc, n - 1); ++k) {
        double target = (nc == 1) ? 0.5 : (k + 1.0) / 3.0;
        int g = std::min(n - 2, std::max(0, (int)std::lround(target * n - 1)));
        while (g < n - 1 && gap[g] > st) ++g;
        if (g < n - 1) gap[g] = cw;
    }
    double used = n * bw; for (double g : gap) used += g;
    double x = 0.5 * (L - used);
    std::vector<std::pair<double,double>> out;
    for (int i = 0; i < n; ++i) { out.push_back({x, x + bw}); x += bw; if (i < n - 1) x += gap[i]; }
    return out;
}

int main() {
    setvbuf(stdout, nullptr, _IOLBF, 0);
    const std::string OUT = getenv("OUT_DIR") ? getenv("OUT_DIR") : "geom_form";
    const double dx = CELL;

    // ── fixed scenario ──
    const double CITY_M   = envd("CITY_M", 600.0);
    const double FAR      = envd("FAR", 2.0);
    const double OCC      = envd("OCCUPANCY", 0.8);            // average occupancy the population is sized for
    const double ENV_FRAC = envd("ENVIRON_FRAC", 0.5);         // environs ring width / city size
    const double H_MAX    = envd("H_MAX", 100.0);              // tallest allowed building (m); sets the wake buffer
    const double BUF_IN   = envd("BUF_IN", 40.0);              // inlet to the upwind edge of the environs
    const double BUF_LAT  = envd("BUF_LAT", 36.0);             // lateral slip face to the environs edge
    const double WAKE_H   = envd("WAKE_H", 15.0);              // downstream buffer in units of H_MAX (COST 732)
    const double HEADROOM = envd("HEADROOM_H", 3.0);           // clear height above H_MAX, in units of H_MAX
    const int    ZONES    = envi("ZONES", 4);                  // release tiles per side
    const int    SEED     = envi("SEED", 1);                   // height-scatter noise
    const double DEP_DP   = envd("DEP_DP", 2.5e-6), DEP_RHO = envd("DEP_RHO", 1800.0), DEP_USTAR = envd("DEP_USTAR", 0.5);

    // ── outer design parameters ──
    const double BLOCK_SIZE  = envd("BLOCK_SIZE", 36.0);   // 24-60 m
    const double BLOCK_ASP   = envd("BLOCK_ASPECT", 1.0);  // 0.5-2 (x extent / y extent)
    const double STREET_W    = envd("STREET_W", 12.0);     // 12-30 m
    const int    CORRIDORS   = envi("CORRIDORS", 0);       // 0-2 wide streets per axis
    const double CORRIDOR_W  = envd("CORRIDOR_W", 40.0);   // 30-50 m
    const double H_CONC      = envd("HEIGHT_CONC", 0.5);   // 0 uniform canopy .. 1 concentrated cores
    const int    H_CORES     = envi("HEIGHT_CORES", 1);    // 1-3
    const double CORE_SPREAD = envd("CORE_SPREAD", 0.5);   // 0-1
    const double H_SCATTER   = envd("HEIGHT_SCATTER", 0.3);// sigma(ln h) 0-0.6
    const double PARK_FRAC   = envd("PARK_FRAC", 0.15);    // 0.05-0.30 of block slots
    const double PARK_LAYOUT = envd("PARK_LAYOUT", 0.5);   // 0 one park .. 1 dispersed pocket parks

    const double GFA_TARGET = FAR * CITY_M * CITY_M;
    const double POP  = GFA_TARGET * OCC / (AREA_PP_RES + LABOR * AREA_PP_JOB);
    const double JOBS = LABOR * POP;

    // ── block slots ──
    const double bw = BLOCK_SIZE * std::sqrt(BLOCK_ASP), bd = BLOCK_SIZE / std::sqrt(BLOCK_ASP);
    auto ax = axis_blocks(CITY_M, bw, STREET_W, CORRIDORS, CORRIDOR_W);
    auto ay = axis_blocks(CITY_M, bd, STREET_W, CORRIDORS, CORRIDOR_W);
    const int NI = (int)ax.size(), NJ = (int)ay.size(), NS = NI * NJ;
    if (NS < 4) { printf("[gen_form_city] INFEASIBLE: only %d block slots\n", NS); return 3; }

    // ── parks: greedy, blending "central" (layout 0) and "maximally dispersed" (layout 1) ──
    const double cxm = 0.5 * CITY_M, cym = 0.5 * CITY_M, dmax = CITY_M * std::sqrt(2.0);
    auto sx = [&](int i) { return 0.5 * (ax[i].first + ax[i].second); };
    auto sy = [&](int j) { return 0.5 * (ay[j].first + ay[j].second); };
    const int npark = std::min(NS - 2, (int)std::lround(PARK_FRAC * NS));
    std::vector<char> isPark(NS, 0);
    std::vector<double> dmin(NS, dmax);
    for (int k = 0; k < npark; ++k) {
        int best = -1; double bs = -1e30;
        for (int s = 0; s < NS; ++s) {
            if (isPark[s]) continue;
            const double dc = std::hypot(sx(s / NJ) - cxm, sy(s % NJ) - cym) / dmax;
            const double score = (1.0 - PARK_LAYOUT) * (1.0 - dc) + PARK_LAYOUT * dmin[s] / dmax;
            if (score > bs + 1e-12) { bs = score; best = s; }
        }
        isPark[best] = 1;
        for (int s = 0; s < NS; ++s)
            dmin[s] = std::min(dmin[s], std::hypot(sx(s / NJ) - sx(best / NJ), sy(s % NJ) - sy(best % NJ)));
    }

    // ── height shape: (1-c) + c * sum of Gaussian cores, times log-normal scatter ──
    std::vector<std::pair<double,double>> cores;
    const double Rc = CORE_SPREAD * 0.35 * CITY_M, sig = 0.2 * CITY_M;
    for (int k = 0; k < std::max(1, H_CORES); ++k) {
        if (H_CORES <= 1) { cores.push_back({cxm, cym}); break; }
        const double a = 2 * M_PI * k / H_CORES - M_PI / 2;
        cores.push_back({cxm + Rc * std::cos(a), cym + Rc * std::sin(a)});
    }
    const int fcap_h = (int)std::floor(H_MAX / FLOOR_H_M);
    std::vector<Bldg> B;
    for (int i = 0; i < NI; ++i) for (int j = 0; j < NJ; ++j) {
        if (isPark[i * NJ + j]) continue;
        Bldg b; b.i = i; b.j = j; b.X0 = ax[i].first; b.X1 = ax[i].second; b.Y0 = ay[j].first; b.Y1 = ay[j].second;
        double s = 0; for (auto& c : cores) { const double r2 = std::pow(sx(i) - c.first, 2) + std::pow(sy(j) - c.second, 2);
                                              s += std::exp(-r2 / (2 * sig * sig)); }
        b.w = ((1.0 - H_CONC) + H_CONC * s) * std::exp(std::sqrt(12.0) * H_SCATTER * (hash01(i, j, SEED) - 0.5));
        b.fmax = std::min(fcap_h, (int)std::floor(SLENDERNESS * std::min(bw, bd) / FLOOR_H_M));
        B.push_back(b);
    }
    const double A = bw * bd;                                  // every building has the same footprint
    const long floorsTarget = std::lround(GFA_TARGET / A);
    long fmaxSum = 0; for (auto& b : B) fmaxSum += b.fmax;
    if ((long)B.size() * MIN_FLOORS > floorsTarget || fmaxSum < floorsTarget) {
        printf("[gen_form_city] INFEASIBLE: %zu buildings of %.0f m2 need %ld floors in total; allowed %ld..%ld\n",
               B.size(), A, floorsTarget, (long)B.size() * MIN_FLOORS, fmaxSum);
        return 3;
    }
    // scale alpha so that sum clamp(alpha w, MIN, fmax) = target, then round by largest remainder
    auto total = [&](double a) { double t = 0; for (auto& b : B) t += std::clamp(a * b.w, (double)MIN_FLOORS, (double)b.fmax); return t; };
    double lo = 0, hi = 1; while (total(hi) < floorsTarget) hi *= 2;
    for (int it = 0; it < 200; ++it) { const double m = 0.5 * (lo + hi); (total(m) < floorsTarget ? lo : hi) = m; }
    const double alpha = hi;
    std::vector<double> frac(B.size());
    long sumF = 0;
    for (size_t k = 0; k < B.size(); ++k) {
        const double f = std::clamp(alpha * B[k].w, (double)MIN_FLOORS, (double)B[k].fmax);
        B[k].floors = (int)std::floor(f); frac[k] = f - B[k].floors; sumF += B[k].floors;
    }
    std::vector<size_t> ord(B.size()); std::iota(ord.begin(), ord.end(), 0);
    std::sort(ord.begin(), ord.end(), [&](size_t a, size_t b) { return frac[a] > frac[b]; });
    for (size_t k = 0; sumF < floorsTarget && k < ord.size(); ++k)
        if (B[ord[k]].floors < B[ord[k]].fmax) { ++B[ord[k]].floors; ++sumF; }
    const double GFA = A * sumF;
    int maxF = 0; for (auto& b : B) maxF = std::max(maxF, b.floors);

    // ── domain: inlet buffer | environs | city | wake buffer (>= environs) ; environs + lateral buffer on both sides ──
    const double RING = ENV_FRAC * CITY_M;
    const double down = std::max(RING, WAKE_H * H_MAX);
    const double ox = BUF_IN + RING, oy = BUF_LAT + RING;          // city origin in the domain
    const int nx = (int)std::lround((ox + CITY_M + down) / dx);
    const int ny = (int)std::lround((2 * oy + CITY_M) / dx);
    const int nz = (int)std::lround((1.0 + HEADROOM) * H_MAX / dx) + 1;
    const size_t N = (size_t)nx * ny * nz;
    VoxelGrid g; g.nx = nx; g.ny = ny; g.nz = nz; g.cell_size = dx;
    g.type.assign(N, CELL_FLUID); g.perm.assign(N, 0.f); g.inh.assign(N, 0.f); g.dep_vel.assign(N, 0.f);
    const float vd_urban = (float)dep::deposition_velocity(city::RES_HIGH, DEP_DP, DEP_RHO, DEP_USTAR);
    const float vd_park  = (float)dep::deposition_velocity(city::PARK,     DEP_DP, DEP_RHO, DEP_USTAR);
    for (int y = 0; y < ny; ++y) for (int x = 0; x < nx; ++x) { size_t id = g.idx(x, y, 0); g.type[id] = CELL_GROUND; g.dep_vel[id] = vd_urban; }
    auto cx_ = [&](double m) { return (int)std::lround((ox + m) / dx); };
    auto cy_ = [&](double m) { return (int)std::lround((oy + m) / dx); };
    std::vector<int32_t> env(N, 0);
    for (size_t k = 0; k < B.size(); ++k) {
        Bldg& b = B[k];
        b.x0 = cx_(b.X0); b.x1 = cx_(b.X1); b.y0 = cy_(b.Y0); b.y1 = cy_(b.Y1);
        b.hc = std::max(1, (int)std::lround(b.floors * FLOOR_H_M / dx));
        for (int z = 1; z <= b.hc && z < nz; ++z) for (int y = b.y0; y < b.y1; ++y) for (int x = b.x0; x < b.x1; ++x) {
            size_t id = g.idx(x, y, z); g.type[id] = CELL_SHELL; g.perm[id] = 0.f; g.dep_vel[id] = vd_urban; }
    }
    const int canopy = std::max(1, (int)std::lround(PARK_CANOPY_M / dx));
    for (int i = 0; i < NI; ++i) for (int j = 0; j < NJ; ++j) {
        if (!isPark[i * NJ + j]) continue;
        for (int z = 1; z <= canopy; ++z) for (int y = cy_(ay[j].first); y < cy_(ay[j].second); ++y)
            for (int x = cx_(ax[i].first); x < cx_(ax[i].second); ++x) {
                size_t id = g.idx(x, y, z); g.type[id] = CELL_SHELL; g.perm[id] = PERM_PARK; g.dep_vel[id] = vd_park; }
    }
    // envelope: fluid cells face-adjacent to a building's walls or roof, labelled with that building
    const int dnx[6] = {1,-1,0,0,0,0}, dny[6] = {0,0,1,-1,0,0}, dnz[6] = {0,0,0,0,1,-1};
    long envShared = 0;
    for (size_t k = 0; k < B.size(); ++k) {
        Bldg& b = B[k];
        for (int z = 1; z <= b.hc && z < nz; ++z) for (int y = b.y0; y < b.y1; ++y) for (int x = b.x0; x < b.x1; ++x)
            for (int d = 0; d < 6; ++d) {
                const int xx = x + dnx[d], yy = y + dny[d], zz = z + dnz[d];
                if (xx < 0 || xx >= nx || yy < 0 || yy >= ny || zz < 1 || zz >= nz) continue;
                const size_t id = g.idx(xx, yy, zz);
                if (g.type[id] != CELL_FLUID) continue;
                if (env[id] == 0) { env[id] = (int32_t)k + 1; ++b.nenv; }
                else if (env[id] != (int32_t)k + 1) ++envShared;
            }
    }

    // ── release: one cell per plan cell over city + environs; ground, or just above the roof ──
    std::vector<uint8_t> src(N, 0), zone(N, 0);
    const int rx0 = (int)std::lround(BUF_IN / dx), rx1 = cx_(CITY_M + RING);
    const int ry0 = (int)std::lround(BUF_LAT / dx), ry1 = cy_(CITY_M + RING);
    long nRel = 0, nRoof = 0;
    for (int y = std::max(1, ry0); y < std::min(ny - 1, ry1); ++y) for (int x = std::max(1, rx0); x < std::min(nx - 1, rx1); ++x) {
        int z = 1;
        while (z < nz - 1 && g.type[g.idx(x, y, z)] == CELL_SHELL && g.perm[g.idx(x, y, z)] < 0.5f) ++z;   // above a roof
        if (z >= nz - 1) continue;
        if (z > 1) ++nRoof;
        const size_t id = g.idx(x, y, z);
        src[id] = 1; ++nRel;
        const int zx = std::min(ZONES - 1, (x - rx0) * ZONES / std::max(1, rx1 - rx0));
        const int zy = std::min(ZONES - 1, (y - ry0) * ZONES / std::max(1, ry1 - ry0));
        zone[id] = (uint8_t)(1 + zx * ZONES + zy);
    }

    if (envi("STATS_ONLY", 0)) {           // layout + statistics only (parameter sweeps): nothing written
        int nmin = 1 << 30; for (auto& b : B) nmin = std::min<long>(nmin, b.nenv);
        printf("STATS buildings=%zu parks=%d floors_max=%d gfa_err=%.4f pop=%.0f env_shared=%ld env_min=%d nx=%d ny=%d nz=%d\n",
               B.size(), npark, maxF, GFA / GFA_TARGET - 1, POP, envShared, nmin, nx, ny, nz);
        return 0;
    }
    // ── materials (inlet on the x = 0 face) + reconciliation gate ──
    MaterialMap mm = build_material_map(g, 0.0);
    { std::string c = "mkdir -p '" + OUT + "'"; if (system(c.c_str())) {} }
    export_material_map((OUT + "/material_map.dat").c_str(), mm);
    const int h5[5] = {nx, ny, nz, (int)std::lround(dx * 1000.0), 1};
    auto wr = [&](const char* fn, const void* p, size_t sz) {
        FILE* f = fopen((OUT + "/" + fn).c_str(), "wb"); if (!f) return;
        fwrite(h5, sizeof(int), 5, f); fwrite(p, sz, N, f); fclose(f); };
    wr("source_mask.u8", src.data(), 1); wr("release_zone.u8", zone.data(), 1);
    wr("envelope.i32", env.data(), 4); wr("dep_vel.f32", g.dep_vel.data(), 4); wr("geom_type.u8", g.type.data(), 1);
    const bool gate = material_reconcile(g, mm, /*verbose=*/false);

    if (FILE* f = fopen((OUT + "/buildings.csv").c_str(), "w")) {
        fprintf(f, "id,slot_i,slot_j,x0,x1,y0,y1,cx_m,cy_m,footprint_m2,floors,height_m,height_cells,floor_area_m2,cap_residents,cap_jobs,envelope_cells\n");
        for (size_t k = 0; k < B.size(); ++k) { const Bldg& b = B[k]; const double fa = A * b.floors;
            fprintf(f, "%zu,%d,%d,%d,%d,%d,%d,%.1f,%.1f,%.1f,%d,%.1f,%d,%.1f,%.1f,%.1f,%ld\n", k, b.i, b.j, b.x0, b.x1, b.y0, b.y1,
                    ox + 0.5 * (b.X0 + b.X1), oy + 0.5 * (b.Y0 + b.Y1), A, b.floors, b.floors * FLOOR_H_M, b.hc, fa,
                    fa / AREA_PP_RES, fa / AREA_PP_JOB, b.nenv); }
        fclose(f);
    }
    if (FILE* f = fopen((OUT + "/meta_geom.txt").c_str(), "w")) {
        fprintf(f, "# gen_form_city: built-form-only city (TWO_TIER_DESIGN.md)\n");
        fprintf(f, "grid_nx %d\ngrid_ny %d\ngrid_nz %d\ndx_m %.4f\nwind_deg 0.0\n", nx, ny, nz, dx);
        fprintf(f, "city_m %.1f\ncity_origin_x_m %.1f\ncity_origin_y_m %.1f\nenviron_ring_m %.1f\nh_max_m %.1f\n", CITY_M, ox, oy, RING, H_MAX);
        fprintf(f, "block_size %.3f\nblock_aspect %.3f\nstreet_w %.3f\ncorridors %d\ncorridor_w %.3f\n", BLOCK_SIZE, BLOCK_ASP, STREET_W, CORRIDORS, CORRIDOR_W);
        fprintf(f, "height_conc %.3f\nheight_cores %d\ncore_spread %.3f\nheight_scatter %.3f\npark_frac %.3f\npark_layout %.3f\nseed %d\n",
                H_CONC, H_CORES, CORE_SPREAD, H_SCATTER, PARK_FRAC, PARK_LAYOUT, SEED);
        fprintf(f, "buildings %zu\nparks %d\nslots %d\nfootprint_m2 %.1f\nfloor_h_m %.2f\nmax_floors %d\n", B.size(), npark, NS, A, FLOOR_H_M, maxF);
        fprintf(f, "gfa_target_m2 %.0f\ngfa_m2 %.0f\npopulation %.0f\njobs %.0f\noccupancy %.3f\n", GFA_TARGET, GFA, POP, JOBS, OCC);
        fprintf(f, "release_cells %ld\nrelease_roof_cells %ld\nrelease_zones %d\nenvelope_shared_cells %ld\n", nRel, nRoof, ZONES * ZONES, envShared);
        fprintf(f, "dep_dp_m %.3g\nreconcile_gate %s\n", DEP_DP, gate ? "PASS" : "FAIL");
        fclose(f);
    }
    printf("[gen_form_city] %d x %d slots -> %zu buildings + %d parks; footprint %.0f m2 (%.1f x %.1f m); floors %d..%d\n",
           NI, NJ, B.size(), npark, A, bw, bd, MIN_FLOORS, maxF);
    printf("[gen_form_city] GFA %.0f m2 (target %.0f, %+.2f%%) -> population %.0f, jobs %.0f\n",
           GFA, GFA_TARGET, 100 * (GFA / GFA_TARGET - 1), POP, JOBS);
    printf("[gen_form_city] domain %d x %d x %d = %.1f M cells at dx %.0f m; release %ld cells (%ld on roofs) in %d zones; reconcile %s\n",
           nx, ny, nz, N / 1e6, dx, nRel, nRoof, ZONES * ZONES, gate ? "PASS" : "FAIL");
    return gate ? 0 : 1;
}
