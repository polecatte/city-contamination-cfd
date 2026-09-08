// must_score.cpp - score predicted vs measured receptor concentrations with the
// Chang & Hanna (2004) / COST-732 / VDI 3783-9 metric suite (see metrics.h).
// Input CSVs: one value per line "id,concentration" (or just "concentration");
// rows are paired by order. Non-dimensionalize both identically before scoring.
//   usage: ./must_score observed.csv predicted.csv [threshold] [Dq] [Wq]
#include "metrics.h"
#include <cstdio>
#include <vector>
#include <string>
#include <cstring>
static std::vector<double> load(const char* p){
    std::vector<double> v; FILE* f=fopen(p,"r"); if(!f){fprintf(stderr,"cannot open %s\n",p);return v;}
    char line[512];
    while(fgets(line,sizeof line,f)){
        if(line[0]=='#'||line[0]=='\n') continue;
        const char* c=strrchr(line,','); double x=atof(c?c+1:line); v.push_back(x);
    }
    fclose(f); return v;
}
int main(int argc,char**argv){
    if(argc<3){ fprintf(stderr,"usage: %s observed.csv predicted.csv [thr] [Dq] [Wq]\n",argv[0]); return 2; }
    auto obs=load(argv[1]), pred=load(argv[2]);
    if(obs.size()!=pred.size()||obs.empty()){ fprintf(stderr,"size mismatch: %zu obs vs %zu pred\n",obs.size(),pred.size()); return 2; }
    double thr=(argc>3)?atof(argv[3]):0.0, Dq=(argc>4)?atof(argv[4]):0.25, Wq=(argc>5)?atof(argv[5]):0.0;
    printf("MUST scorecard (Chang & Hanna 2004; VDI 3783-9 hit rate)  %zu receptors\n",obs.size());
    metrics::report(metrics::evaluate(obs,pred,thr,Dq,Wq));
    return 0;
}
