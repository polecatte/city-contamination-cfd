// tail_replay.cpp — replay a finished burst's time series through tail_fit.h's rule.
//
//   g++ -O2 -std=c++17 -I. tests/tail_replay.cpp -o tail_replay
//   ./tail_replay OUT_DIR/exposure_timeseries.csv [TAIL_FROM]
//
// The run must have gone well past the stopping levels tested (CLEAR_FRAC 0.001, fine
// TS_EVERY). For each stopping level L (airborne mass < L x peak) it walks the rows as the
// solver would, stops at the first row where the fit passes, and compares
//   J(t_stop)              (no tail: what a plain stop at that level reports)
//   J(t_stop) + rate/k     (with the tail)
// against the reference: J at the run's last row plus that row's own tail (the residual at
// 0.1 % is already tiny). Also prints the burst steps saved. TAIL_SPEC.md reports the result.
#include "tail_fit.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>

struct Row { int step; double t, air, J; };

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s exposure_timeseries.csv [TAIL_FROM]\n", argv[0]); return 2; }
    std::ifstream in(argv[1]);
    if (!in) { std::fprintf(stderr, "cannot read %s\n", argv[1]); return 2; }
    std::string line; std::getline(in, line);                       // header
    std::vector<Row> rows;
    while (std::getline(in, line)) {
        std::stringstream ss(line); std::string f; std::vector<double> v;
        while (std::getline(ss, f, ',')) v.push_back(std::atof(f.c_str()));
        if (v.size() >= 10) rows.push_back({(int)v[0], v[1], v[2], v[9]});
    }
    if (rows.size() < 20) { std::fprintf(stderr, "too few rows (%zu)\n", rows.size()); return 2; }
    const double from = argc > 2 ? std::atof(argv[2]) : 0.2;
    double peak = 0; for (auto& r : rows) peak = std::max(peak, r.air);
    // pulse end: first row after the airborne maximum (the solver samples only after the pulse)
    size_t i0 = 0; for (size_t i = 0; i < rows.size(); ++i) if (rows[i].air == peak) { i0 = i; break; }

    // reference: last row + its own tail
    tailfit::Tracker ref; ref.from = from;
    for (size_t i = i0; i < rows.size(); ++i) ref.add(rows[i].t, rows[i].J, rows[i].air);
    const tailfit::Fit rf = ref.fit(peak);
    const double Jref = rows.back().J + (rf.k > 0 ? rf.tail : 0);
    std::printf("[tail] %zu rows, last at %.0f s (step %d), airborne %.2e of peak; reference J = %.6e "
                "(last row %.6e + own tail %.3f %%, k = %.4g/s, R2 %.4f)\n",
                rows.size(), rows.back().t, rows.back().step, rows.back().air / peak, Jref, rows.back().J,
                100 * (rf.k > 0 ? rf.tail / Jref : 0), rf.k, rf.r2);
    std::printf("[tail] %-8s %-10s %-10s %-10s %-9s %-7s %-7s %-6s\n", "stop at", "t_stop s", "steps", "no-tail", "with tail", "R2", "e^span", "frac");
    for (double L : {0.30, 0.20, 0.10, 0.05, 0.02, 0.01}) {
        tailfit::Tracker tr; tr.from = from;
        bool done = false;
        for (size_t i = i0; i < rows.size() && !done; ++i) {
            tr.add(rows[i].t, rows[i].J, rows[i].air);
            if (rows[i].air >= L * peak) continue;
            const tailfit::Fit f = tr.fit(peak);
            if (!f.ok) continue;
            const double eNo = rows[i].J / Jref - 1, eTail = (rows[i].J + f.tail) / Jref - 1;
            std::printf("[tail] %-8.2f %-10.0f %-10d %+9.2f%% %+8.2f%% %-7.4f %-7.2f %5.1f%%\n",
                        L, rows[i].t, rows[i].step, 100 * eNo, 100 * eTail, f.r2, f.span, 100 * f.frac);
            done = true;
        }
        if (!done) std::printf("[tail] %-8.2f fit never passed before the run ended\n", L);
    }
    std::printf("[tail] plain stop at the solver's default 1 %% (no tail): ");
    for (auto& r : rows) if (r.t > rows[i0].t && r.air < 0.01 * peak) {
        std::printf("t %.0f s, J %+.2f %% vs reference\n", r.t, 100 * (r.J / Jref - 1)); break; }
    return 0;
}
