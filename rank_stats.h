// rank_stats.h — rank-correlation statistics shared by the diagnostic tools.
// Spearman ρ, Kendall τ, Pearson r, average ranks (tie-aware), and best-k overlap.
#pragma once
#include <vector>
#include <algorithm>
#include <cmath>

namespace rstats {

inline void average_ranks(const std::vector<double>& x, std::vector<double>& rk){
    int n=(int)x.size(); std::vector<int> idx(n); for(int i=0;i<n;++i) idx[i]=i;
    std::sort(idx.begin(),idx.end(),[&](int a,int b){return x[a]<x[b];});
    rk.assign(n,0);
    int i=0;
    while(i<n){ int j=i; while(j+1<n && x[idx[j+1]]==x[idx[i]]) ++j;
        double r=0.5*((i+1)+(j+1)); for(int k=i;k<=j;++k) rk[idx[k]]=r; i=j+1; }
}
inline double pearson(const std::vector<double>& a,const std::vector<double>& b){
    int n=(int)a.size(); if(n<2) return 0.0;
    double ma=0,mb=0; for(int i=0;i<n;++i){ma+=a[i];mb+=b[i];} ma/=n;mb/=n;
    double sab=0,sa=0,sb=0;
    for(int i=0;i<n;++i){ double da=a[i]-ma,db=b[i]-mb; sab+=da*db; sa+=da*da; sb+=db*db; }
    return (sa>0&&sb>0)? sab/std::sqrt(sa*sb) : 0.0;
}
inline double spearman(const std::vector<double>& a,const std::vector<double>& b){
    std::vector<double> ra,rb; average_ranks(a,ra); average_ranks(b,rb); return pearson(ra,rb);
}
inline double kendall(const std::vector<double>& a,const std::vector<double>& b){
    int n=(int)a.size(); long con=0,dis=0;
    for(int i=0;i<n;++i)for(int j=i+1;j<n;++j){
        double da=a[i]-a[j], db=b[i]-b[j];
        if(da*db>0) ++con; else if(da*db<0) ++dis;
    }
    long tot=con+dis; return tot? (double)(con-dis)/tot : 0.0;
}
// fraction of the best-k (lowest value) items both rankings agree on
inline double topk_overlap(const std::vector<double>& a,const std::vector<double>& b,int k){
    int n=(int)a.size(); k=std::min(k,n); if(k<=0) return 1.0;
    auto bestk=[&](const std::vector<double>& x){
        std::vector<int> idx(n); for(int i=0;i<n;++i)idx[i]=i;
        std::partial_sort(idx.begin(),idx.begin()+k,idx.end(),[&](int p,int q){return x[p]<x[q];});
        return std::vector<int>(idx.begin(),idx.begin()+k); };
    auto A=bestk(a), B=bestk(b); int hit=0;
    for(int x:A) if(std::find(B.begin(),B.end(),x)!=B.end()) ++hit;
    return (double)hit/k;
}

} // namespace rstats
