// urban_les.h evaluates the WALE powers through sqrt (x^2.5 = x^2 sqrt x, x^1.25 = x sqrt sqrt x,
// x^1.5 = x sqrt x) instead of pow. This checks, on random velocity gradients over the range the
// lattice sees (|g| 1e-7 .. 0.3, lattice units), that the two forms agree to round-off in double
// and in float, and how far the float result is from the double one (the single-precision build).
// The collision uses omega_eff = 1/(1/omega + tau_t), so that is what is compared; tau_t itself
// only where it moves omega_eff by more than 1e-6 (below that, tau_t ~ |g|^2 and its powers sit
// near float's denormal range, |g| ~ 1e-7 gives (S:S)^2.5 ~ 1e-35, and do not matter).
//   g++ -O2 -std=c++17 pow_sqrt_check.cpp -o pow_sqrt_check && ./pow_sqrt_check
#include <cmath>
#include <cstdio>
#include <random>
#include <algorithm>

template <typename V>
void invariants(const double vg[9], V& ss, V& sdsd) {   // urban_les.h, the same loops
  V g[3][3], g2[3][3];
  for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) g[i][j] = V(vg[3 * i + j]);
  for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) {
    g2[i][j] = 0; for (int k = 0; k < 3; ++k) g2[i][j] += g[i][k] * g[k][j]; }
  const V tr = (g2[0][0] + g2[1][1] + g2[2][2]) / V(3);
  ss = 0; sdsd = 0;
  for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) {
    const V sd = V(0.5) * (g2[i][j] + g2[j][i]) - (i == j ? tr : V(0));
    const V s = V(0.5) * (g[i][j] + g[j][i]);
    sdsd += sd * sd; ss += s * s; }
}
template <typename V> V tauPow(V ss, V sdsd, V cw) {
  const V den = std::pow(ss, V(2.5)) + std::pow(sdsd, V(1.25));
  return den > V(0) ? V(3) * cw * cw * std::pow(sdsd, V(1.5)) / den : V(0);
}
template <typename V> V tauSqrt(V ss, V sdsd, V cw) {
  const V rs = std::sqrt(ss), rd = std::sqrt(sdsd);
  const V den = ss * ss * rs + sdsd * std::sqrt(rd);
  return den > V(0) ? V(3) * cw * cw * (sdsd * rd) / den : V(0);
}
// omega_eff = 1/(1/omega + tau_t): what the collision actually uses (tau 0.505 -> omega 1.98)
template <typename V> V omEff(V tau, V omega) { return V(1) / (V(1) / omega + tau); }

int main() {
  std::mt19937_64 rng(7);
  std::normal_distribution<double> N(0, 1);
  std::uniform_real_distribution<double> L(-7, std::log10(0.3));
  const double cw = 0.325, omega = 1 / 0.505;
  double dD = 0, dF = 0, dDom = 0, dFom = 0, fd = 0, fdOm = 0, tmax = 0;
  long n = 0, nz = 0, nBig = 0;
  for (int k = 0; k < 200000; ++k) {
    double vg[9]; const double s = std::pow(10.0, L(rng));
    for (double& v : vg) v = s * N(rng);
    if (k % 7 == 0) for (int i = 0; i < 9; ++i) if (i % 4) vg[i] = 0;      // diagonal (pure strain)
    if (k % 11 == 0) { for (double& v : vg) v = 0; vg[2] = s; }             // pure shear: g^2 = 0, tau_t = 0
    double ssd, sdd; float ssf, sdf;
    invariants(vg, ssd, sdd); invariants(vg, ssf, sdf);
    const double pd = tauPow(ssd, sdd, cw), qd = tauSqrt(ssd, sdd, cw);
    const float pf = tauPow(ssf, sdf, float(cw)), qf = tauSqrt(ssf, sdf, float(cw));
    if (!(pd > 0)) { ++nz; if (qd != 0 || qf != 0) { std::printf("zero mismatch at %d\n", k); return 1; } continue; }
    tmax = std::max(tmax, pd);
    const double od = omEff(pd, omega);
    dDom = std::max(dDom, std::fabs(omEff(qd, omega) - od) / od);
    dFom = std::max(dFom, std::fabs(double(omEff(qf, float(omega))) - double(omEff(pf, float(omega))))
                          / double(omEff(pf, float(omega))));
    fdOm = std::max(fdOm, std::fabs(double(omEff(qf, float(omega))) - od) / od);
    if (pd > 1e-6) {                                  // tau_t that moves omega_eff by > ~1e-6
      dD = std::max(dD, std::fabs(qd - pd) / pd);
      dF = std::max(dF, std::fabs(double(qf) - double(pf)) / double(pf));
      fd = std::max(fd, std::fabs(double(qf) - qd) / qd);
      ++nBig;
    }
    ++n;
  }
  std::printf("%ld gradients (%ld with tau_t = 0; %ld with tau_t > 1e-6), max tau_t %.3g\n", n, nz, nBig, tmax);
  std::printf("sqrt vs pow, double: omega_eff %.2e, tau_t %.2e   (round-off: < 1e-14)\n", dDom, dD);
  std::printf("sqrt vs pow, float : omega_eff %.2e, tau_t %.2e   (round-off: < 1e-5)\n", dFom, dF);
  std::printf("float vs double (sqrt form): omega_eff %.2e, tau_t %.2e\n", fdOm, fd);
  const bool ok = dDom < 1e-14 && dD < 1e-14 && dFom < 1e-5 && dF < 1e-5 && fdOm < 1e-5;
  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
