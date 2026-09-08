#pragma once
// city_zoning.h — Mixed potential field zoning overlay for city_builder7.
//
// The idea: city_builder7's generate() assigns business and parks by ranking
// blocks on a single scalar — dist_to_biz = radial distance from center
// (optionally blended with coherent noise via patchiness). That produces
// purely concentric distributions.
//
// This header introduces a designed potential field Φ(x,y) built from a
// small orthogonal basis of named terms. Blocks are still ranked on Φ and
// filled greedily — the engine is unchanged — but Φ can now express sector
// zoning, twin-core, linear corridors, and blends thereof.
//
// Usage:
//   1. Call city::generate(p) normally → get block layout + morphology.
//   2. Set MixedZoningParams fields (all weights default to 0; set only what
//      you want active). Call city::rezone(p, result, mzp).
//   3. Export the returned Result with city::export_city as usual.
//
// city_builder7.h is NOT modified.
//
// ── Basis terms for business potential Φ (all normalized to ≈[0,1]) ──────
//
//   w_radial    · φ_r(x,y) = r/R
//               → monocentric core (current default)
//
//   w_grad_x    · φ_gx(x,y) = −(x−cx)/R
//               → business on the positive-x (right) half; gradient/sector
//
//   w_grad_y    · φ_gy(x,y) = −(y−cy)/R
//               → business on the positive-y (top) half
//
//   w_bipeak_x  · φ_2px(x,y) = −cos(2π(x−cx)/city_w)
//               → twin cores at x = cx ± city_w/4; polycentric along x
//
//   w_bipeak_y  · φ_2py(x,y) = −cos(2π(y−cy)/city_h)
//               → twin cores along y
//
//   w_corridor  · φ_cor(x,y) = |(x−cx)sinθ − (y−cy)cosθ| / R
//               → linear main-street corridor through center at angle θ;
//                 blocks ON the line get Φ=0 → assigned business first.
//                 θ=0 → E-W corridor; θ=π/2 → N-S corridor.
//
// ── Park potential Φ_park (SAME basis as business) ─────────────────────────
//
//   Parks are placed with the identical designed-field machinery as business:
//   a small Φ_park marks a preferred green site, blocks are ranked on it, and
//   park_scatter (like biz_scatter) chooses between one contiguous park mass
//   (0) and evenly dispersed pocket parks (1, the default). The same six basis
//   terms apply — radial, gradient, twin-peak, corridor — so a park corridor
//   term gives a single green spine, a twin-peak term gives two green bands,
//   and the radial term (driven by the legacy park_centrality knob) gives a
//   central park or an edge greenbelt ring.
//
//   This replaces the earlier special-case periodic "stripe" cosine field:
//   parks appeared "in stripes" only because that periodic function was the
//   sole spatial structure the park model had. A general potential subsumes it
//   (true periodic bands are just one basis term, not the whole model) and puts
//   parks on equal footing with business.
//
#include "city_builder7.h"
#include <numeric>
#include <cmath>

namespace city {

struct MixedZoningParams {
    // ── Business potential weights (any non-negative value; scale is relative)
    double w_radial    = 0; // radial (monocentric)
    double w_grad_x    = 0; // gradient along x (sector, right-heavy)
    double w_grad_y    = 0; // gradient along y (sector, top-heavy)
    double w_bipeak_x  = 0; // twin-peak along x (polycentric)
    double w_bipeak_y  = 0; // twin-peak along y
    double w_corridor  = 0; // linear corridor through center
    double corridor_angle = 0.0; // corridor axis angle (radians from +x)

    // ── Business assignment mode
    // 0 = greedy rank-fill: blocks sorted by Φ, filled lowest-first → contiguous zone.
    // 1 = maximin dispersal: each successive pick is as far as possible from all
    //     already-placed business blocks → individual buildings scattered across city.
    // Intermediate values blend both; Φ still biases where the scatter is denser.
    double biz_scatter = 0.0;

    // ── Park potential weights (SAME basis as business; small Φ_park → park-
    //    preferred). Parks use the identical designed-field machinery — no more
    //    special-case periodic stripe field.
    double park_w_radial   = 0; // extra radial bias (adds to park_centrality below)
    double park_w_grad_x   = 0; // sector gradient along x
    double park_w_grad_y   = 0; // sector gradient along y
    double park_w_bipeak_x = 0; // twin green bands along x
    double park_w_bipeak_y = 0; // twin green bands along y
    double park_w_corridor = 0; // single green corridor / spine through center
    double park_corridor_angle = 0.0; // green-corridor axis angle (radians from +x)

    // ── Park assignment mode (mirrors biz_scatter)
    // 0 = greedy rank-fill: one contiguous park mass at the Φ_park minimum.
    // 1 = maximin dispersal: pocket parks spread as evenly as possible.
    double park_scatter = 1.0; // default 1 → familiar even-dispersed pocket parks

    // ── Legacy radial centrality knob, folded into the park radial weight so old
    //    callers keep working: 0 = edge/greenbelt, 0.5 = neutral, 1 = central.
    double park_centrality = 0.5;

    // ── Reaction–diffusion zoning. When rd_enable is set, a Gray–Scott field
    //    REPLACES phi_biz/phi_park: business is drawn to the activator peaks and
    //    parks to the open valleys, so the RD morphology becomes the built form.
    bool     rd_enable = false;
    double   rd_F      = 0.035; // feed rate  (with rd_k selects spots/worms/maze/holes)
    double   rd_k      = 0.065; // kill rate
    double   rd_aniso  = 1.0;   // diffusion anisotropy (>1 → wind-aligned canyons)
    int      rd_scale  = 3;     // RD cells per block (feature ≈ 6/rd_scale blocks wide)
    int      rd_steps  = 6000;  // integration steps
    unsigned rd_seed   = 1u;

    // ── Density-field population zoning. When dens_enable is set, an employment
    //    density field and a residential density field JOINTLY drive land use,
    //    building heights, AND the population/receptor distribution, so the people
    //    (w) and the built form (which shapes C) move together. This replaces the
    //    potential-field / scatter / RD machinery above.
    //
    //    Business locates at the employment peaks and is tall; residential density
    //    follows a monocentric gradient modulated by jobs-housing mixing; parks
    //    fill the low-intensity valleys. Heights derive from the fields, so an
    //    employment centre is genuinely dense in the flow, not just re-labelled.
    bool     dens_enable      = false;
    int      emp_centers      = 1;    // number of employment centres (1..4)
    double   emp_centrality   = 0.8;  // 0 = centres on an outer ring (spread), 1 = all central
    double   emp_concentration= 0.5;  // 0 = broad centres, 1 = tight peaks
    double   res_gradient     = 1.5;  // residential falloff exp(−b·r/R); 0 = uniform sprawl
    double   jh_mix           = 0.5;  // 0 = segregated (housing avoids job cores), 1 = mixed-use
    double   peak_height      = 70.0; // building height (m) at the densest employment peak
};

// ── Potential field primitives ────────────────────────────────────────────

// Generic designed potential Φ from the orthogonal basis. Small → preferred
// (ranked first in the greedy fill). Each term is normalized to ≈[0,1] so the
// weights are directly comparable. Shared verbatim by business and parks.
inline double phi_field(double x, double y,
                        double cx, double cy,
                        double city_w, double city_h,
                        double w_radial, double w_grad_x, double w_grad_y,
                        double w_bipeak_x, double w_bipeak_y,
                        double w_corridor, double corridor_angle) {
    double R  = 0.5 * std::max(city_w, city_h);
    double dx = x - cx, dy = y - cy;
    double r  = std::sqrt(dx*dx + dy*dy);
    double phi = 0.0;

    phi += w_radial   *  (r / R);
    phi += w_grad_x   * -(dx / R);                        // small = right (+x)
    phi += w_grad_y   * -(dy / R);                        // small = top  (+y)
    phi += w_bipeak_x * -std::cos(2.0*M_PI*dx / city_w); // minima at ±city_w/4
    phi += w_bipeak_y * -std::cos(2.0*M_PI*dy / city_h);
    // Perpendicular distance to the line through center at angle corridor_angle.
    // Points on the line have perp=0 → preferred first.
    double th   = corridor_angle;
    double perp = std::abs(dx*std::sin(th) - dy*std::cos(th));
    phi += w_corridor * (perp / R);

    return phi;
}

// Business potential: small → more business-like → ranked first in fill.
inline double phi_biz(double x, double y,
                      double cx, double cy,
                      double city_w, double city_h,
                      const MixedZoningParams& m) {
    return phi_field(x, y, cx, cy, city_w, city_h,
                     m.w_radial, m.w_grad_x, m.w_grad_y,
                     m.w_bipeak_x, m.w_bipeak_y, m.w_corridor, m.corridor_angle);
}

// Park potential: same basis. The legacy park_centrality knob (0=edge,1=central)
// is folded into the radial weight — central parks want a small Φ_park at r=0, so
// centrality maps to a positive radial weight of magnitude up to 1 (comparable to
// a unit basis term). park_w_radial adds any further radial bias on top.
inline double phi_park(double x, double y,
                       double cx, double cy,
                       double city_w, double city_h,
                       const MixedZoningParams& m) {
    double w_rad = m.park_w_radial + 2.0*(m.park_centrality - 0.5); // 0→−1, 1→+1
    return phi_field(x, y, cx, cy, city_w, city_h,
                     w_rad, m.park_w_grad_x, m.park_w_grad_y,
                     m.park_w_bipeak_x, m.park_w_bipeak_y,
                     m.park_w_corridor, m.park_corridor_angle);
}

// ── Shared greedy / maximin ordering ───────────────────────────────────────
// Reorders `idxs` (block indices) for scatter-aware sequential fill, used by
// BOTH business and park assignment. px,py,phi are indexed by block id.
//   scatter=0 → idxs sorted by Φ ascending  → contiguous greedy fill.
//   scatter=1 → maximin dispersal: seed at the Φ-minimum, then each next pick
//               maximizes the min-distance to all already-picked blocks.
//   intermediate → convex blend  score = (1−sc)·(1−Φ_norm) + sc·(d_min/d_scale).
// Deterministic and identical for both land uses.
inline std::vector<int> scatter_order(const std::vector<double>& px,
                                      const std::vector<double>& py,
                                      const std::vector<double>& phi,
                                      std::vector<int> idxs,
                                      double scatter, double d_scale) {
    int n = (int)idxs.size();
    if (n < 2) return idxs;
    std::sort(idxs.begin(), idxs.end(),
              [&](int a, int b){ return phi[a] < phi[b]; });
    double sc = std::max(0.0, std::min(1.0, scatter));
    if (sc <= 1e-6) return idxs;                       // pure Φ-ascending

    double phi_lo = phi[idxs.front()], phi_hi = phi[idxs.back()];
    double phi_rng = std::max(1e-9, phi_hi - phi_lo);
    if (d_scale <= 0.0) d_scale = 1.0;

    std::vector<char>   taken(n, 0);
    std::vector<double> min_dp(n, 1e18);
    std::vector<int>    out; out.reserve(n);

    int seed = 0;                                      // idxs[0] = global Φ-minimum
    out.push_back(idxs[seed]); taken[seed] = 1;
    for (int j = 0; j < n; ++j) {
        double dx = px[idxs[j]] - px[idxs[seed]];
        double dy = py[idxs[j]] - py[idxs[seed]];
        min_dp[j] = std::sqrt(dx*dx + dy*dy);
    }
    min_dp[seed] = 0.0;

    for (int step = 1; step < n; ++step) {
        int best = -1; double bscore = -1e18;
        for (int j = 0; j < n; ++j) {
            if (taken[j]) continue;
            double phi_n = (phi[idxs[j]] - phi_lo) / phi_rng;
            double d_n   = min_dp[j] / d_scale;
            double score = (1.0 - sc)*(1.0 - phi_n) + sc*d_n;
            if (score > bscore) { bscore = score; best = j; }
        }
        if (best < 0) break;
        out.push_back(idxs[best]); taken[best] = 1;
        for (int j = 0; j < n; ++j) {
            if (taken[j]) continue;
            double dx = px[idxs[j]] - px[idxs[best]];
            double dy = py[idxs[j]] - py[idxs[best]];
            double d  = std::sqrt(dx*dx + dy*dy);
            if (d < min_dp[j]) min_dp[j] = d;
        }
    }
    return out;
}

// ── Gray–Scott reaction–diffusion field ────────────────────────────────────
// Integrates the Gray–Scott two-species system on an Rx×Ry periodic grid and
// returns the activator v (row-major, y*Rx+x). The caller samples it at block
// centres. Diffusion is trace-preserving anisotropic (ax+ay=2) so the explicit
// update stays stable for any anisotropy ratio. Seeding is deterministic (a hash
// of rd_seed), so the same params always give the same city.
inline std::vector<double> gray_scott_field(int Rx, int Ry,
                                            double F, double k, double aniso,
                                            int steps, unsigned seed) {
    int NN = Rx*Ry;
    std::vector<double> u(NN,1.0), v(NN,0.0), un(NN,0.0), vn(NN,0.0);
    auto rnd = [&](unsigned i)->double{
        unsigned h = seed*2654435761u + i*40503u;
        h ^= h>>13; h *= 0x5bd1e995u; h ^= h>>15;
        return (h & 0xffffffu)/double(0x1000000u); };
    int nseed = std::max(8, NN/60);
    for (int s=0; s<nseed; ++s) {
        int ix = (int)(rnd(2u*s)*(Rx-2))+1, iy = (int)(rnd(2u*s+1u)*(Ry-2))+1;
        for (int dy=-1; dy<=1; ++dy) for (int dx=-1; dx<=1; ++dx) {
            int xx=(ix+dx+Rx)%Rx, yy=(iy+dy+Ry)%Ry; v[yy*Rx+xx]=0.5; u[yy*Rx+xx]=0.25; }
    }
    for (int i=0;i<NN;++i){ u[i]+=0.01*rnd(1000u+i); v[i]+=0.01*rnd(500000u+i); }
    const double Du=0.16, Dv=0.08;
    double ax = 2.0*aniso/(1.0+aniso), ay = 2.0/(1.0+aniso);
    auto ID = [&](int x,int y){ return ((y+Ry)%Ry)*Rx + ((x+Rx)%Rx); };
    for (int t=0; t<steps; ++t) {
        for (int y=0; y<Ry; ++y) for (int x=0; x<Rx; ++x) {
            int c=y*Rx+x;
            double lu = -(2*ax+2*ay)*u[c] + ax*(u[ID(x+1,y)]+u[ID(x-1,y)]) + ay*(u[ID(x,y+1)]+u[ID(x,y-1)]);
            double lv = -(2*ax+2*ay)*v[c] + ax*(v[ID(x+1,y)]+v[ID(x-1,y)]) + ay*(v[ID(x,y+1)]+v[ID(x,y-1)]);
            double uvv = u[c]*v[c]*v[c];
            un[c] = u[c] + Du*lu - uvv + F*(1.0-u[c]);
            vn[c] = v[c] + Dv*lv + uvv - (F+k)*v[c];
        }
        u.swap(un); v.swap(vn);
    }
    return v;
}

// ── Density fields: employment E(x) and residential R(x) per block ──────────
// Both are normalized to [0,1]. Employment is a Gaussian mixture of emp_centers
// centres (monocentric → polycentric via emp_centrality/count); residential is a
// negative-exponential density gradient modulated by jobs-housing mixing: at
// jh_mix=0 housing avoids the job cores (segregated), at jh_mix=1 it concentrates
// with them (mixed-use). These feed both the receptor weighting w and the height
// field, so land use and morphology stay physically coupled.
inline void density_fields(const std::vector<Block>& blocks,
                           double cx, double cy, double cw, double ch,
                           const MixedZoningParams& m,
                           std::vector<double>& E, std::vector<double>& R) {
    int N = (int)blocks.size();
    E.assign(N, 0.0); R.assign(N, 0.0);
    double Rmax  = 0.5 * std::min(cw, ch);
    int    k     = std::max(1, std::min(4, m.emp_centers));
    double cen   = std::max(0.0, std::min(1.0, m.emp_centrality));
    double conc  = std::max(0.0, std::min(1.0, m.emp_concentration));
    double ring  = (1.0 - cen) * 0.42 * std::max(cw, ch);
    double sigma = std::max(cw, ch) * (0.34 - 0.24 * conc);   // broad → tight

    std::vector<std::pair<double,double>> ctr;
    if (k == 1) ctr.push_back({cx, cy});
    else for (int j = 0; j < k; ++j) {
        double a = 2.0*M_PI*j/k - M_PI/2;
        ctr.push_back({cx + ring*std::cos(a), cy + ring*std::sin(a)});
    }

    double emax = 1e-12;
    for (int i = 0; i < N; ++i) {
        double e = 0.0;
        for (auto& c : ctr) {
            double dx = blocks[i].center_x - c.first, dy = blocks[i].center_y - c.second;
            e += std::exp(-(dx*dx + dy*dy)/(2*sigma*sigma));
        }
        E[i] = e; emax = std::max(emax, e);
    }
    for (auto& e : E) e /= emax;

    double b   = std::max(0.0, m.res_gradient);
    double jh  = std::max(0.0, std::min(1.0, m.jh_mix));
    double rmx = 1e-12;
    for (int i = 0; i < N; ++i) {
        double dx = blocks[i].center_x - cx, dy = blocks[i].center_y - cy;
        double r  = std::sqrt(dx*dx + dy*dy) / std::max(1.0, Rmax);
        double rbase = std::exp(-b * r);
        // segregated (jh=0): housing ∝ (1−E) → avoids cores; mixed (jh=1): housing boosted at cores
        double reff  = rbase * ((1.0 - jh)*(1.0 - E[i]) + jh*(0.5 + 0.5*E[i]));
        R[i] = reff; rmx = std::max(rmx, reff);
    }
    for (auto& r : R) r /= rmx;
}

// ── rezone: re-assign business and parks on an existing Result ────────────
//
// Block positions, footprints, and raw_heights come from generate() and are
// not recomputed (morphology is independent of zoning). Everything from
// "sort by Φ" onward is redone here, including heights→floors, RES_LOW
// reclassification, employment balance, and NHAPS inhabitance — matching
// generate()'s steps 2-10 exactly.
//
inline Result rezone(const Params& p, Result r, const MixedZoningParams& mzp) {
    using std::max; using std::min; using std::sqrt; using std::exp;

    auto& blocks = r.blocks;
    int N = (int)blocks.size();
    if (N < 3) return r;

    bool has_buf = (p.buf_xn > 0 || p.buf_xp > 0 || p.buf_yn > 0 || p.buf_yp > 0);
    double cw = (p.city_w > 0) ? p.city_w : p.Sx;
    double ch = (p.city_h > 0) ? p.city_h : p.Sy;
    double cx = has_buf ? (p.buf_xn + cw*0.5) : (p.Sx*0.5);
    double cy = has_buf ? (p.buf_yn + ch*0.5) : (p.Sy*0.5);
    double cov = 1.0; // coverage fixed at 1 (v8)

    // ── Step A: Coordinate + potential arrays (shared by business & parks) ─
    // Optional reaction–diffusion field, sampled once per block. Business prefers
    // the activator peaks (Φ = −v); parks prefer the valleys (Φ_park = +v).
    std::vector<double> rd_block;
    if (mzp.rd_enable) {
        int gnx=0, gny=0;
        for (auto& b : blocks) { gnx=max(gnx,b.grid_ix+1); gny=max(gny,b.grid_iy+1); }
        int sc = max(1, mzp.rd_scale);
        int Rx = max(8, sc*gnx), Ry = max(8, sc*gny);
        std::vector<double> v = gray_scott_field(Rx, Ry, mzp.rd_F, mzp.rd_k,
                                                 mzp.rd_aniso, mzp.rd_steps, mzp.rd_seed);
        rd_block.assign(N, 0.0);
        for (int i=0;i<N;++i) {
            int sx = min(Rx-1, blocks[i].grid_ix*sc + sc/2);
            int sy = min(Ry-1, blocks[i].grid_iy*sc + sc/2);
            rd_block[i] = v[sy*Rx + sx];
        }
    }

    // Optional density-field population model. Computes employment/residential
    // density per block and OVERRIDES raw_height so the built form follows the
    // population distribution (dense job centres are genuinely tall; sprawl is low).
    std::vector<double> emp_field, res_field;
    if (mzp.dens_enable) {
        density_fields(blocks, cx, cy, cw, ch, mzp, emp_field, res_field);
        for (int i = 0; i < N; ++i) {
            double h = BASE_HEIGHT_M + mzp.peak_height*(0.78*emp_field[i] + 0.32*res_field[i]);
            blocks[i].raw_height = h;   // heights now derive from the density fields
        }
    }

    std::vector<double> px(N), py(N), phi_biz_v(N);
    for (int i = 0; i < N; ++i) {
        px[i] = blocks[i].center_x;
        py[i] = blocks[i].center_y;
        phi_biz_v[i] = mzp.dens_enable ? -emp_field[i]            // business at job peaks
                     : mzp.rd_enable   ? -rd_block[i]
                     : phi_biz(blocks[i].center_x, blocks[i].center_y, cx, cy, cw, ch, mzp);
        blocks[i].dist_to_biz = phi_biz_v[i];  // Step H employment balance ranks on this
    }
    double d_scale = std::max(cw, ch);

    // ── Step B: Business fill order via the shared scatter ordering ───────
    // biz_scatter=0 → Φ-ascending (contiguous CBD); =1 → maximin (scattered shops).
    std::vector<int> all_idx(N);
    std::iota(all_idx.begin(), all_idx.end(), 0);
    std::vector<int> order = scatter_order(px, py, phi_biz_v, all_idx,
                                           mzp.biz_scatter, d_scale);

    // ── Step C: Reset all to RES_HIGH, place parks, then assign business ─
    // Park count from direct park_fraction knob (same as generate()).
    double park_frac = max(0.0, min(MAX_PARK_FRAC, p.park_fraction));
    int n_park = (int)(N*park_frac + 0.5);

    for (auto& b : blocks) b.usage = RES_HIGH;

    // ── Step D: Business assignment — single pass at α=1 ─────────────────
    // Iteratively converts the nearest-Φ RES_HIGH block to BUSINESS until the
    // employment constraint (workers = pop·LABOR_RATE, area per worker = BIZ_AREA_PW)
    // is satisfied at ≥95%. Uses raw_height directly (morphology heights, α=1).
    auto assign_biz = [&]() -> int {
        for (auto& b : blocks)
            if (b.usage == BUSINESS) b.usage = RES_HIGH;
        int nb = 0;
        for (int iter = 0; iter < 200; ++iter) {
            double pop_est = 0, biz_fa = 0;
            for (auto& b : blocks) {
                double h = max(CELL, min(b.raw_height, (double)b.max_height_cells*CELL));
                if (b.usage == PARK)     { continue; }
                if (b.usage == BUSINESS) { biz_fa += b.footprint_m2*(h/BIZ_FLOOR_H); continue; }
                double fh = RES_FLOOR_H, nf = h/fh;
                Usage eu = b.usage; double efp = b.footprint_m2;
                if (eu==RES_HIGH && (int)std::round(nf) <= RES_LOW_MAX_FLOORS)
                    { eu=RES_LOW; efp *= RES_LOW_COVERAGE/cov; }
                pop_est += residents_per_m2(eu)*efp*nf;
            }
            double biz_needed = pop_est * LABOR_RATE * BIZ_AREA_PW;
            if (biz_fa >= biz_needed*0.95) break;
            bool ok = false;
            for (int k = 0; k < N; ++k) {
                int idx = order[k];
                if (blocks[idx].usage != RES_HIGH) continue;
                blocks[idx].usage = BUSINESS; nb++; ok = true; break;
            }
            if (!ok) break;
        }
        return nb;
    };
    assign_biz();

    // ── Step E: Park placement — same designed field + scatter as business ─
    // Parks are ranked on Φ_park (radial/centrality + gradient + twin-peak +
    // corridor) and filled with the shared scatter_order():
    //   park_scatter=0 → one contiguous green mass at the Φ_park minimum;
    //   park_scatter=1 → maximin-dispersed pocket parks.
    // A tiny deterministic jitter breaks ties on flat fields so an unstructured
    // (Φ_park≈const) dispersal stays non-degenerate.
    {
        std::vector<double> phi_park_v(N);
        for (int i = 0; i < N; ++i)
            phi_park_v[i] = (mzp.dens_enable
                             ? (emp_field[i] + res_field[i])   // parks in low-intensity valleys
                             : mzp.rd_enable
                             ? rd_block[i]     // parks prefer the RD valleys (low v)
                             : phi_park(blocks[i].center_x, blocks[i].center_y,
                                        cx, cy, cw, ch, mzp))
                          + 1e-3*hash01(blocks[i].grid_ix, blocks[i].grid_iy);

        std::vector<int> pool;
        for (int i = 0; i < N; ++i)
            if (blocks[i].usage == RES_HIGH || blocks[i].usage == RES_LOW)
                pool.push_back(i);
        int M = (int)pool.size();
        int np = min(n_park, M);

        std::vector<int> park_order = scatter_order(px, py, phi_park_v, pool,
                                                    mzp.park_scatter, d_scale);
        for (int k = 0; k < np && k < (int)park_order.size(); ++k)
            blocks[park_order[k]].usage = PARK;
    }

    // Fix park footprints (fill full block, no setback — same as generate() Step 5)
    for (auto& b : blocks) if (b.usage == PARK) {
        b.x0=b.bx0; b.y0=b.by0; b.x1=b.bx1; b.y1=b.by1;
        b.footprint_m2 = (double)(b.x1-b.x0)*(b.y1-b.y0)*CELL*CELL;
    }

    // ── Step F: Re-run business assignment after parks are placed ─────────
    // Parks now excluded from the pool, so business fills remaining blocks in Φ order.
    assign_biz();

    // ── Step G: Snap heights → floors, RES_LOW reclassification ─────────
    // (mirrors generate() Step 8 exactly)
    double rn = max(0.0, min(0.8, p.roughness));
    for (int ki = 0; ki < N; ++ki) {
        auto& b = blocks[ki];
        if (b.usage == PARK) {
            b.height_cells = max(1,(int)std::round(PARK_CANOPY_H/CELL));
            b.n_floors=0; b.eff_pop=0; b.empl_cap=0; b.eff_inh=0;
        } else {
            double raw_h = b.raw_height * exp(ROUGH_GAIN*rn*block_noise(ki));
            raw_h = max(CELL, min(raw_h, (double)b.max_height_cells*CELL));
            double fh = (b.usage==BUSINESS) ? BIZ_FLOOR_H : RES_FLOOR_H;
            b.n_floors = max(1,(int)std::round(raw_h/fh));
            if (b.usage==RES_HIGH && b.n_floors<=RES_LOW_MAX_FLOORS) {
                b.usage = RES_LOW;
                double scale = sqrt(RES_LOW_COVERAGE/cov);
                int cw_b=b.x1-b.x0, cd_b=b.y1-b.y0;
                int nw=max(2,(int)(cw_b*scale)), nd=max(2,(int)(cd_b*scale));
                int ccx=(b.x0+b.x1)/2, ccy=(b.y0+b.y1)/2;
                b.x0=ccx-nw/2; b.x1=b.x0+nw;
                b.y0=ccy-nd/2; b.y1=b.y0+nd;
                b.footprint_m2=(double)(b.x1-b.x0)*(b.y1-b.y0)*CELL*CELL;
                int md_lo=min(b.x1-b.x0,b.y1-b.y0);
                b.max_height_cells=(int)(SLENDERNESS*md_lo);
            }
            double phys_h = b.n_floors*fh;
            b.height_cells = max(1,(int)std::round(phys_h/CELL));
            b.height_cells = min(b.height_cells, b.max_height_cells);
            double fa = b.footprint_m2*b.n_floors;
            b.eff_pop   = residents_per_m2(b.usage)*fa;
            b.empl_cap  = workers_per_m2(b.usage)*fa;
            b.eff_inh   = 0;
        }
    }

    // ── Step H: Employment balance + dasymetric population ───────────────
    // (mirrors generate() Steps 9a+9b exactly)
    double P_total = max(0.0, p.population_total);
    {
        auto sum_empl=[&]{ double s=0; for(auto&b:blocks) s+=b.empl_cap; return s; };
        double workers_needed = P_total*LABOR_RATE;
        double total_empl = sum_empl();
        double ratio = total_empl / max(1.0, workers_needed);
        int swaps = 0;
        while (ratio < 0.9 && swaps < 20) {
            int best=-1; double bdst=1e18;
            for (int i=0;i<N;++i) {
                if (blocks[i].usage!=RES_HIGH) continue;
                if (blocks[i].dist_to_biz<bdst){bdst=blocks[i].dist_to_biz;best=i;}
            }
            if (best<0) break;
            auto& b=blocks[best];
            double fa=b.footprint_m2*b.n_floors;
            b.usage=BUSINESS; b.eff_pop=0; b.empl_cap=DENS_BIZ*fa;
            total_empl=sum_empl(); ratio=total_empl/max(1.0,workers_needed); swaps++;
        }
        while (ratio > 1.3 && swaps < 20) {
            int best=-1; double bdst=0;
            for (int i=0;i<N;++i) {
                if (blocks[i].usage!=BUSINESS) continue;
                if (blocks[i].dist_to_biz>bdst){bdst=blocks[i].dist_to_biz;best=i;}
            }
            if (best<0) break;
            auto& b=blocks[best];
            double fa=b.footprint_m2*b.n_floors;
            b.usage=RES_HIGH; b.eff_pop=residents_per_m2(RES_HIGH)*fa; b.empl_cap=0;
            total_empl=sum_empl(); ratio=total_empl/max(1.0,workers_needed); swaps++;
        }
        r.nudges = swaps;
    }
    double cap_sum=0;
    for (auto& b : blocks) cap_sum += b.eff_pop;
    if (cap_sum > 0)
        for (auto& b : blocks) b.eff_pop *= P_total/cap_sum;
    r.population = P_total;

    // ── Step I: NHAPS time-budget inhabitance ────────────────────────────
    // (mirrors generate() Step 10 exactly)
    r.max_height=0; r.park_area_m2=0; r.total_inhabitance=0;
    double total_res_fa=0, total_biz_fa=0, total_park_area=0;
    for (auto& b : blocks) {
        if (b.height_cells*CELL > r.max_height) r.max_height=b.height_cells*(double)CELL;
        double fa=b.footprint_m2*b.n_floors;
        if (b.usage==PARK)                      { r.park_area_m2+=b.footprint_m2; total_park_area+=b.footprint_m2; }
        else if(b.usage==RES_HIGH||b.usage==RES_LOW) total_res_fa+=fa;
        else if(b.usage==BUSINESS)               total_biz_fa+=fa;
    }
    double home_ppl=P_total*TIME_HOME, work_ppl=P_total*TIME_WORK, park_ppl=P_total*TIME_PARK;
    for (auto& b : blocks) {
        double fa=b.footprint_m2*b.n_floors; b.eff_inh=0;
        if      (b.usage==PARK)
            b.eff_inh=(total_park_area>0)?park_ppl*(b.footprint_m2/total_park_area):0;
        else if (b.usage==RES_HIGH||b.usage==RES_LOW)
            b.eff_inh=(total_res_fa>0)?home_ppl*(fa/total_res_fa):0;
        else if (b.usage==BUSINESS)
            b.eff_inh=(total_biz_fa>0)?work_ppl*(fa/total_biz_fa):0;
        r.total_inhabitance+=b.eff_inh;
    }

    // ── Final metrics + counts ────────────────────────────────────────────
    r.biz_floor_area=total_biz_fa;
    double n_workers_f=r.population*LABOR_RATE;
    r.worker_density=(n_workers_f>0)?r.biz_floor_area/n_workers_f:0;
    r.total_open_m2=r.park_area_m2;
    r.open_space_ratio=(r.total_open_m2/SQM_PER_ACRE)/(r.population/1000.0);
    r.park_frac=park_frac;

    for (int i=0;i<5;++i) r.counts[i]=0;
    r.num_blocks=(int)blocks.size();
    for (auto& b : blocks) if(b.usage>=0&&b.usage<5) r.counts[b.usage]++;
    r.biz_frac_actual=(double)r.counts[1]/max(1,r.num_blocks);
    r.v_road_major=r.v_road_major; r.h_road_major=r.h_road_major; // carry through from generate()

    return r;
}

} // namespace city
