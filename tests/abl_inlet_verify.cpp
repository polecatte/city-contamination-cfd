// abl_inlet_verify.cpp — numerical verification of the ABL/RFG inlet (abl_inlet.h)
// BEFORE it is wired into OpenLB. This is the Step-3 gate check §6.1 calls for:
// "validate it first (mean-shear + turbulence-intensity profiles)".
//
// Checks, all purely on the analytical inlet field (no LBM needed):
//   1. MEAN PROFILE  — velocity() horizontal speed vs the log law U(z)=(u*/k)ln((z+z0)/z0)
//   2. TURB INTENSITY — empirical sigma_u,v,w over a sampled inlet plane × time vs the
//                       prescribed surface-layer targets 2.5/1.9/1.25 · u*
//   3. SOLENOIDALITY  — RMS divergence of the fluctuation field, normalized by the RMS
//                       velocity-gradient magnitude (the "divergence-free by construction"
//                       claim; the header honestly notes a small residual — we measure it)
//   4. DETERMINISM    — identical output for identical (x,y,z,t) on repeated calls
//
// Build:  g++ -O2 -std=c++17 abl_inlet_verify.cpp -o abl_inlet_verify
// Run:    ./abl_inlet_verify   (writes abl_inlet_profile.csv for plotting)

#include "abl_inlet.h"
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>

int main() {
    using abl::ABLInlet; using abl::KAPPA;

    // Operating point matching forward_city.cpp's config factory:
    //   U_inlet 4 m/s at z_ref 4 m, z0 0.045 m, L_turb 20 m, 100 modes,
    //   sigma ratios 2.5 / 1.9 / 1.25.
    const double U_REF = 4.0, Z_REF = 4.0;
    ABLInlet in;
    in.z0 = 0.045; in.d = 0.0; in.wind_angle = 0.0;
    in.L_turb = 20.0; in.n_modes = 100;
    in.sigma_u_ratio = 2.5; in.sigma_v_ratio = 1.9; in.sigma_w_ratio = 1.25;
    in.enable_turb = true;
    in.init(U_REF, Z_REF, /*seed=*/1000u);

    printf("=== ABL/RFG inlet verification ===\n");
    printf("U_ref=%.2f m/s @ z_ref=%.2f m  ->  u*=%.4f m/s  (z0=%.3f, kappa=%.2f)\n",
           U_REF, Z_REF, in.u_star, in.z0, KAPPA);

    // ── 1. MEAN PROFILE vs analytic log law ──
    // Compare the STREAMWISE-component mean (unbiased, since the fluctuation is
    // zero-mean) to the log law. NOTE: the mean SPEED |u_h| is a different quantity —
    // it is biased above U near the ground because sigma_u is comparable to U there
    // (Jensen's inequality). We report both so the distinction is explicit.
    printf("\n[1] mean streamwise speed vs log law (and mean |u_h| for reference)\n");
    printf("   %6s  %10s  %10s  %8s  %10s\n", "z(m)", "mean_ux", "U_loglaw", "rel.err", "mean|u_h|");
    double max_rel = 0.0;
    FILE* csv = fopen("abl_inlet_profile.csv", "w");
    if (csv) fprintf(csv, "z_m,mean_ux,U_loglaw,mean_speed,sig_u,sig_v,sig_w,ti_u\n");
    std::vector<double> zs = {1,2,4,8,16,32,64,100};
    // Randomized (y,t) sampling of the inlet plane. A structured grid aliases against
    // the Fourier modes and fakes a nonzero fluctuation mean, so we hash-sample instead.
    auto hrand = [](uint64_t& s){ s^=s<<13; s^=s>>7; s^=s<<17; return (s>>11)*(1.0/9007199254740992.0); };
    for (double z : zs) {
        double sumSpeed=0; int M = 200000;
        double sxx=0, syy=0, szz=0, mx=0, my=0, mz=0;
        uint64_t rs = 0x243F6A8885A308D3ull ^ (uint64_t)(z*10007);
        for (int m = 0; m < M; ++m) {
            double x = 0.0;                          // inlet plane x=0
            double y = hrand(rs) * 4000.0;           // random span position (m)
            double t = hrand(rs) * 4000.0;           // random time (s)
            double ux,uy,uz; in.velocity(x, y, z, t, ux, uy, uz);
            sumSpeed += std::sqrt(ux*ux + uy*uy);
            mx += ux; my += uy; mz += uz;
            sxx += ux*ux; syy += uy*uy; szz += uz*uz;
        }
        mx/=M; my/=M; mz/=M;
        double Umeas = mx;                           // streamwise mean (unbiased)
        double Uspeed = sumSpeed / M;
        double Ulog  = in.mean_speed(z);
        double rel = std::fabs(Umeas - Ulog) / std::max(1e-9, Ulog);
        max_rel = std::max(max_rel, rel);
        double sig_u = std::sqrt(std::max(0.0, sxx/M - mx*mx));
        double sig_v = std::sqrt(std::max(0.0, syy/M - my*my));
        double sig_w = std::sqrt(std::max(0.0, szz/M - mz*mz));
        printf("   %6.1f  %10.4f  %10.4f  %7.2f%%  %10.4f\n", z, Umeas, Ulog, 100.0*rel, Uspeed);
        if (csv) fprintf(csv, "%.3f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f\n",
                         z, Umeas, Ulog, Uspeed, sig_u, sig_v, sig_w, sig_u/std::max(1e-9,Ulog));
    }
    if (csv) fclose(csv);
    printf("   -> max streamwise mean-profile error vs log law: %.2f%%  [%s]\n",
           100.0*max_rel, max_rel < 0.02 ? "PASS <2%" : "CHECK");

    // ── 2. TURBULENCE INTENSITY: empirical sigmas vs prescribed targets ──
    printf("\n[2] turbulence intensity (target sigma = ratio * u*)\n");
    {
        double tgt_u = in.sigma_u_ratio * in.u_star;
        double tgt_v = in.sigma_v_ratio * in.u_star;
        double tgt_w = in.sigma_w_ratio * in.u_star;
        // sample fluct() over a large space-time volume; it is unit-variance by design
        long M = 400000; double sx=0,sy=0,sz=0;
        for (long m = 0; m < M; ++m) {
            double x = (m % 97) * 2.3, y = (m % 89) * 1.7, z = 2.0 + (m % 53) * 1.9;
            double t = (m % 131) * 0.11;
            double fx,fy,fz; in.fluct(x,y,z,t,fx,fy,fz);
            sx += fx*fx; sy += fy*fy; sz += fz*fz;
        }
        double vu = std::sqrt(sx/M), vv = std::sqrt(sy/M), vw = std::sqrt(sz/M);
        printf("   unit-field RMS (should be ~1.0):  u=%.3f  v=%.3f  w=%.3f\n", vu, vv, vw);
        printf("   scaled sigma_u=%.4f (target %.4f, %.1f%%)\n", vu*in.sigma_u_ratio*in.u_star, tgt_u, 100.0*vu);
        printf("   scaled sigma_v=%.4f (target %.4f, %.1f%%)\n", vv*in.sigma_v_ratio*in.u_star, tgt_v, 100.0*vv);
        printf("   scaled sigma_w=%.4f (target %.4f, %.1f%%)\n", vw*in.sigma_w_ratio*in.u_star, tgt_w, 100.0*vw);
        double err = std::max({std::fabs(vu-1),std::fabs(vv-1),std::fabs(vw-1)});
        printf("   -> max unit-variance error: %.2f%%  [%s]\n", 100.0*err,
               err < 0.05 ? "PASS <5%" : "CHECK");
    }

    // ── 3. SOLENOIDALITY: RMS divergence / RMS gradient magnitude ──
    printf("\n[3] divergence of the fluctuation field (solenoidal claim)\n");
    {
        const double h = 1e-3;   // finite-difference step (m)
        long M = 60000; double sum_div2 = 0.0, sum_grad2 = 0.0;
        for (long m = 0; m < M; ++m) {
            double x = (m % 91) * 2.1, y = (m % 83) * 1.9, z = 3.0 + (m % 61) * 1.3;
            double t = (m % 127) * 0.07;
            double fpx,fpy,fpz, fmx,fmy,fmz;
            in.fluct(x+h,y,z,t,fpx,fpy,fpz); in.fluct(x-h,y,z,t,fmx,fmy,fmz);
            double dfx_dx = (fpx-fmx)/(2*h);
            in.fluct(x,y+h,z,t,fpx,fpy,fpz); in.fluct(x,y-h,z,t,fmx,fmy,fmz);
            double dfy_dy = (fpy-fmy)/(2*h);
            in.fluct(x,y,z+h,t,fpx,fpy,fpz); in.fluct(x,y,z-h,t,fmx,fmy,fmz);
            double dfz_dz = (fpz-fmz)/(2*h);
            double div = dfx_dx + dfy_dy + dfz_dz;
            // gradient-magnitude scale for normalization
            double gmag2 = dfx_dx*dfx_dx + dfy_dy*dfy_dy + dfz_dz*dfz_dz;
            sum_div2 += div*div; sum_grad2 += gmag2;
        }
        double rms_div = std::sqrt(sum_div2/M);
        double rms_grad = std::sqrt(sum_grad2/M);
        double ratio = rms_div / std::max(1e-12, rms_grad);
        printf("   RMS(div f) / RMS(|diag grad|) = %.4f  (0 = perfectly solenoidal)\n", ratio);
        printf("   -> residual divergence: %.1f%%  [%s]\n", 100.0*ratio,
               ratio < 0.25 ? "acceptable (header notes small residual; pressure cleans it)"
                            : "LARGE — investigate anisotropy normalization");
    }

    // ── 4. DETERMINISM (CPU/GPU parity prerequisite) ──
    printf("\n[4] determinism\n");
    {
        double a1,a2,a3,b1,b2,b3;
        in.velocity(12.3, 45.6, 7.8, 3.21, a1,a2,a3);
        in.velocity(12.3, 45.6, 7.8, 3.21, b1,b2,b3);
        bool same = (a1==b1 && a2==b2 && a3==b3);
        printf("   repeated velocity(12.3,45.6,7.8,3.21) identical: %s\n", same?"YES [PASS]":"NO [FAIL]");
    }

    printf("\nwrote abl_inlet_profile.csv\n");
    return 0;
}
