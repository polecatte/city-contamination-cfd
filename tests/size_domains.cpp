// Minimal directional domains for the example cities, foregoing power-of-2.
// Compares: previous uniform 15H (centred, wasteful) vs minimal asymmetric under
// (a) COST-732 standard  inflow 5H / outflow 15H, and
// (b) the as-stated assignment inflow 15H / outflow 5H (reverse of the standard).
#include "city_builder7.h"
#include <cstdio>
#include <string>
using namespace city;

static Params mkp(double sx,double sy,double bh,double cp,double cd,double pf){
    Params p{};
    p.city_w=480; p.city_h=480; p.block_w=48; p.block_d=48; p.base_height=bh;
    p.cbd_peak=cp; p.cbd_decay=cd; p.cbd_aspect=1; p.cbd_angle=0;
    p.biz_inner_frac=0.05; p.biz_aspect=1;
    p.park_centrality=0.5; p.park_fraction=pf; p.roughness=0.15;
    p.road_w_x=sx; p.road_w_y=sy; p.population_total=20000; p.wind_direction=0;
    p.buf_xn=p.buf_xp=p.buf_yn=p.buf_yp=200;            // probe domain
    p.Sx=p.buf_xn+p.city_w+p.buf_xp; p.Sy=p.buf_yn+p.city_h+p.buf_yp;
    double cx=p.buf_xn+p.city_w*0.5, cy=p.buf_yn+p.city_h*0.5;
    p.cbd_x=cx; p.cbd_y=cy; p.biz_center_x=cx; p.biz_center_y=cy;
    p.source_x=p.buf_xn-20; p.source_y=cy;
    return p;
}
static double cells_M(const Params& p,int nz){ return (double)(p.Sx/CELL)*(p.Sy/CELL)*nz/1e6; }

int main(){
    struct C{const char*n; double sx,sy,bh,cp,cd,pf;};
    C cs[]={ {"A baseline",20,20,12,40,8e-6,0.10},{"B narrow",8,8,12,40,8e-6,0.10},
             {"C wide",40,40,12,40,8e-6,0.10},{"D anisotropic",40,8,12,40,8e-6,0.10},
             {"E high parks",20,20,12,40,8e-6,0.30},{"F tall CBD",20,20,10,120,2.5e-5,0.08} };
    printf("%-13s %4s | %-22s | %-30s | %-30s\n","city","H",
           "uniform 15H (centred)","COST-732 min (5H in/15H out)","as-stated (15H in/5H out)");
    printf("%s\n", std::string(110,'-').c_str());
    for(auto&c:cs){
        Params p0=mkp(c.sx,c.sy,c.bh,c.cp,c.cd,c.pf); double H=generate(p0).max_height;
        int nz=nz_cost732(H);
        Params u=mkp(c.sx,c.sy,c.bh,c.cp,c.cd,c.pf); u.buf_xn=u.buf_xp=u.buf_yn=u.buf_yp=15*H;
        u.Sx=u.buf_xn+u.city_w+u.buf_xp; u.Sy=u.buf_yn+u.city_h+u.buf_yp;
        Params a=mkp(c.sx,c.sy,c.bh,c.cp,c.cd,c.pf); minimal_domain(a,H,5,15);
        Params b=mkp(c.sx,c.sy,c.bh,c.cp,c.cd,c.pf); minimal_domain(b,H,15,5);
        printf("%-13s %3.0fm | %5.0fx%-5.0f %5.1fM | %5.0fx%-5.0f x%d  %6.1fM | %5.0fx%-5.0f x%d  %6.1fM\n",
               c.n,H, u.Sx,u.Sy,cells_M(u,nz),
               a.Sx,a.Sy,nz,cells_M(a,nz), b.Sx,b.Sy,nz,cells_M(b,nz));
    }
    printf("\n(rows: domain X x Y in m, then xNZ cells, then Mcells. NZ=6H/cell, no pow2.)\n");
    return 0;
}
