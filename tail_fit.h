#pragma once
// tail_fit.h — exponential-tail extrapolation of the burst's exposure integral.
// Spec and validation: TAIL_SPEC.md. Used by urban_flow (TAIL=1) and tests/tail_replay.cpp,
// so the replay of a finished run uses exactly the solver's rule.
//
// After the pulse, the receptor-weighted dose rate r(t) = d<w,Theta>/dt = <w, C(t)> decays,
// at late times, at the rate k of the slowest flushing mode (the concentration field keeps its
// shape and shrinks). Then the dose still to come after a stop at t_s is
//     integral_{t_s}^inf r dt = r(t_s) / k,
// the same extrapolation as the area under a concentration-time curve to infinity in
// pharmacokinetics (AUC_inf = AUC_t + C_last / lambda_z; Gibaldi & Perrier 1982) and the
// exponential tail of a tracer-decay measurement. k comes from a least-squares line through
// ln r(t) over the late window; the solver adds C(x, t_s)/k to Theta in every cell, so
// <w, Theta> gains <w, C(t_s)>/k = r(t_s)/k.
#include <vector>
#include <cmath>

namespace tailfit {

struct Sample { double t, wtheta, air; };   // time (s), <w,Theta> so far, airborne mass in the CV

struct Fit {
    bool   ok = false;     // passes every acceptance test
    int    n = 0;          // rate samples in the window
    double k = 0;          // decay rate (1/s)
    double r2 = 0;         // of the ln-rate line
    double rate = 0;       // fitted dose rate at the last sample
    double tail = 0;       // rate / k: <w,Theta> still to come
    double frac = 0;       // tail / (<w,Theta> so far + tail)
    double span = 0;       // ln(rate at window start / rate at end): how much decay the window saw
};

struct Tracker {
    double from    = 0.2;   // window: samples once airborne mass <= from x peak
    double r2min   = 0.98;  // the late decay must be close to a single exponential...
    int    nmin    = 8;     // ...over enough samples...
    double spanmin = 0.7;   // ...covering at least a factor e^0.7 = 2 of decay
    double maxfrac = 0.15;  // and the extrapolated part must stay a minor share of J
    std::vector<Sample> s;

    void add(double t, double wtheta, double air) { s.push_back({t, wtheta, air}); }

    Fit fit(double peakAir) const {
        Fit f;
        std::vector<double> X, Y;
        for (size_t i = 1; i < s.size(); ++i) {
            if (s[i - 1].air > from * peakAir) continue;           // both ends inside the window
            const double dt = s[i].t - s[i - 1].t;
            if (dt <= 0) continue;
            const double r = (s[i].wtheta - s[i - 1].wtheta) / dt;
            if (!(r > 0)) continue;
            X.push_back(0.5 * (s[i].t + s[i - 1].t));
            Y.push_back(std::log(r));
        }
        f.n = (int)X.size();
        if (f.n < 3) return f;
        double mx = 0, my = 0;
        for (int i = 0; i < f.n; ++i) { mx += X[i]; my += Y[i]; }
        mx /= f.n; my /= f.n;
        double sxx = 0, sxy = 0, syy = 0;
        for (int i = 0; i < f.n; ++i) {
            sxx += (X[i] - mx) * (X[i] - mx); sxy += (X[i] - mx) * (Y[i] - my); syy += (Y[i] - my) * (Y[i] - my);
        }
        if (sxx <= 0 || syy <= 0) return f;
        const double b = sxy / sxx, a = my - b * mx;
        f.k = -b;
        f.r2 = sxy * sxy / (sxx * syy);
        f.span = -b * (X.back() - X.front());
        if (!(f.k > 0)) return f;
        f.rate = std::exp(a + b * s.back().t);
        f.tail = f.rate / f.k;
        const double J = s.back().wtheta;
        f.frac = f.tail / (J + f.tail);
        f.ok = f.n >= nmin && f.r2 >= r2min && f.span >= spanmin && f.frac <= maxfrac;
        return f;
    }
};

} // namespace tailfit
