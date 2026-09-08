// metrics.h — Chang & Hanna (2004) statistical performance metrics for
// atmospheric dispersion model evaluation, with the COST-732 / VDI 3783-9
// acceptance criteria. Used to score the forward scalar against measured urban
// dispersion data (MUST; Yee & Biltoft 2004).
//
//   FB    fractional bias           ideal 0     accept |FB| <= 0.3   (>0 => model under-predicts)
//   NMSE  normalized mean sq error  ideal 0     accept NMSE <= 1.5   (Chang & Hanna; 4 is looser)
//   FAC2  fraction within factor 2  ideal 1     accept FAC2 >= 0.5
//   MG    geometric mean bias       ideal 1     accept 0.7 <= MG <= 1.3
//   VG    geometric variance        ideal 1     accept VG <= 1.6..4
//   R     Pearson correlation       ideal 1
//   HR    hit rate (VDI 3783-9)     ideal 1     accept HR >= 0.66  (Dq=0.25 rel, Wq abs)
//
// Refs: Chang & Hanna, Meteorol. Atmos. Phys. 87 (2004) 167; VDI 3783 Part 9
// (2005); COST Action 732 (Franke et al. 2007; Di Sabatino et al. 2011).
#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdio>

namespace metrics {

struct Scores {
    int n;
    double FB, NMSE, FAC2, MG, VG, R, HR;
    double mean_obs, mean_pred;
};

// obs/pred paired at the receptors. `thr` = data threshold: pairs where BOTH obs
// and pred are below it are dropped (below detection limit), matching COST-732
// practice; MG/VG additionally need positivity so any remaining <=0 is floored.
// Dq (rel tol) and Wq (abs tol, same units as C) define the VDI hit rate.
inline Scores evaluate(const std::vector<double>& obs, const std::vector<double>& pred,
                       double thr=0.0, double Dq=0.25, double Wq=0.0) {
    std::vector<double> o, p;
    for(size_t i=0;i<obs.size();++i)
        if(!(obs[i]<thr && pred[i]<thr)){ o.push_back(obs[i]); p.push_back(pred[i]); }
    Scores S{}; S.n=(int)o.size();
    if(S.n==0){ return S; }
    double mo=0,mp=0; for(int i=0;i<S.n;++i){ mo+=o[i]; mp+=p[i]; }
    mo/=S.n; mp/=S.n; S.mean_obs=mo; S.mean_pred=mp;

    double denom=0.5*(mo+mp);
    S.FB = (denom!=0)? (mo-mp)/denom : 0.0;

    double mse=0; for(int i=0;i<S.n;++i){ double d=o[i]-p[i]; mse+=d*d; } mse/=S.n;
    S.NMSE = (mo*mp>0)? mse/(mo*mp) : (mse==0?0.0:INFINITY);

    int f2=0; for(int i=0;i<S.n;++i){ if(o[i]>0&&p[i]>0){ double r=p[i]/o[i]; if(r>=0.5&&r<=2.0) ++f2; } }
    S.FAC2 = (double)f2/S.n;

    // geometric metrics on positive data (floor tiny values to the smallest positive)
    double floor=INFINITY; for(int i=0;i<S.n;++i){ if(o[i]>0) floor=std::min(floor,o[i]); if(p[i]>0) floor=std::min(floor,p[i]); }
    if(!std::isfinite(floor)) floor=1e-30; floor*=1e-3;
    double slo=0,slp=0,svg=0;
    for(int i=0;i<S.n;++i){ double lo=std::log(std::max(o[i],floor)), lp=std::log(std::max(p[i],floor));
        slo+=lo; slp+=lp; svg+=(lo-lp)*(lo-lp); }
    slo/=S.n; slp/=S.n; svg/=S.n;
    S.MG = std::exp(slo-slp);        // >1 => under-prediction
    S.VG = std::exp(svg);

    double so=0,sp=0,soo=0,spp=0,sop=0;
    for(int i=0;i<S.n;++i){ so+=o[i];sp+=p[i];soo+=o[i]*o[i];spp+=p[i]*p[i];sop+=o[i]*p[i]; }
    double cov=sop/S.n-mo*mp, vo=soo/S.n-mo*mo, vp=spp/S.n-mp*mp;
    S.R = (vo>0&&vp>0)? cov/std::sqrt(vo*vp) : 0.0;

    // VDI 3783-9 hit rate: |pred-obs|/|obs| <= Dq  OR  |pred-obs| <= Wq
    int hit=0; for(int i=0;i<S.n;++i){ double a=std::fabs(p[i]-o[i]);
        if((o[i]!=0 && a/std::fabs(o[i])<=Dq) || a<=Wq) ++hit; }
    S.HR = (double)hit/S.n;
    return S;
}

inline void report(const Scores& S, FILE* f=stdout){
    auto ok=[&](bool b){ return b?"PASS":"----"; };
    fprintf(f,"  n=%d  mean_obs=%.3g  mean_pred=%.3g\n", S.n, S.mean_obs, S.mean_pred);
    fprintf(f,"  FAC2 = %.3f  (accept >=0.50)  %s\n", S.FAC2, ok(S.FAC2>=0.5));
    fprintf(f,"  FB   = %+.3f (accept |FB|<=0.30) %s\n", S.FB, ok(std::fabs(S.FB)<=0.3));
    fprintf(f,"  NMSE = %.3f  (accept <=1.50)  %s\n", S.NMSE, ok(S.NMSE<=1.5));
    fprintf(f,"  MG   = %.3f  (accept .7-1.3)  %s\n", S.MG, ok(S.MG>=0.7&&S.MG<=1.3));
    fprintf(f,"  VG   = %.3f  (accept <=1.60)  %s\n", S.VG, ok(S.VG<=1.6));
    fprintf(f,"  R    = %.3f\n", S.R);
    fprintf(f,"  HR   = %.3f  (accept >=0.66)  %s\n", S.HR, ok(S.HR>=0.66));
}

} // namespace metrics
