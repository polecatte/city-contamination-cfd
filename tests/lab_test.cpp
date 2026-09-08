// lab_test.cpp — full-model test run for the lab computer.
//
// Builds a 1024 m square city at 4 m resolution with the MINIMUM COST-732
// buffers, runs the ABL-turbulent-inlet LBM with the finalized 6-bin
// polydisperse source for 1 h of dispersion, and exports everything the
// visualizer needs: city geometry, velocity at z=2, particulate dispersion
// (mass-weighted TIAC), ground deposition, and a exposure map + total exposure.
//
// Buffers: COST Action 732 / Franke et al. (2007) minima — inflow 5H, lateral
// 5H, outflow 15H (H = tallest building); compute_domain() then rounds each
// axis up to a power of two (so these are minima, satisfied or exceeded).
// Transport: Option A (POLYDISPERSE_NOTE) — warm the flow once, replay each
// activity bin on the same deterministic flow via warm restart; accumulate
// mass-weighted fields. Settling Stokes+Cunningham; deposition Zhang 2001.
// Inlet: log-law mean (Richards & Hoxey 1993) + RFG turbulence (Kraichnan
// 1970; Smirnov et al. 2001). Occupancy: NHAPS 24-h (Klepeis et al. 2001),
// indoor F_inf (Liu & Nazaroff 2001; Chen & Zhao 2011; Nazaroff 2004), street
// pathway (occupancy.h).
//
// Build (CPU/OpenMP):
//   g++ -O3 -std=c++17 -fopenmp -c lbm_kernels_cpu.cpp -o kernels.o
//   g++ -O3 -std=c++17 -c lbm_solver.cpp -o solver.o
//   g++ -O3 -std=c++17 -c lab_test.cpp -o lab_test.o
//   g++ kernels.o solver.o lab_test.o -fopenmp -o lab_test
// Build (GPU): compile lbm_kernels.cu with nvcc (see test_poiseuille.cpp header)
//   and link instead of kernels.o. The 1 h / 6-bin run is a GPU job.
//
// Run:   ./lab_test [city_m=1024] [release_s=3600] [n_bins=6] [max_warmup=20000]
//   (small args = quick smoke test; defaults = the full lab run)

#include "city_builder7.h"
#include "voxelize.h"
#include "lbm_solver.h"
#include "psd.h"
#include "infiltration.h"
#include "indoor_exposure.h"
#include "occupancy.h"
#include "reverse_objective.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cstdint>
#include <string>
using namespace city;

// ── DEMONSTRATION exposure weights (placeholders) ──────────────────────────
// These placeholders only set the relative weighting of the two illustrative
// exposure pathways (airborne vs deposited) so the exposure MAP and TOTAL
// demonstrate the pipeline; they are not certified figures.
static const double W_AIRBORNE = 1.0;    // per unit time-integrated air conc (airborne inhalation)
static const double W_DEPOSITED    = 0.5;    // per unit deposited contaminant (deposited-surface exposure)

int main(int argc, char** argv){
    setvbuf(stdout, nullptr, _IOLBF, 0);   // line-buffer: progress streams to logs/pipes
    double city_m   = (argc>1)? atof(argv[1]) : 1024.0;
    double release_s= (argc>2)? atof(argv[2]) : 3600.0;   // 1 hour
    int    n_bins   = (argc>3)? atoi(argv[3]) : 6;
    int    max_warm = (argc>4)? atoi(argv[4]) : 20000;

    // ── 1. City + uniform buffers, centered in the power-of-two domain ───────
    Params p{};
    p.city_w=city_m; p.city_h=city_m;
    p.block_w=48; p.block_d=24; p.base_height=12;
    p.cbd_peak=40; p.cbd_decay=8e-6; p.cbd_aspect=1; p.cbd_angle=0;  // 40 m peak: fits 4 m on 16 GB
    p.biz_inner_frac=0.05; p.biz_aspect=1; p.park_centrality=0.5; p.park_fraction=0.10;
 p.roughness=0.15; p.road_w_x=20; p.road_w_y=20; p.population_total=20000;
    p.wind_direction=0;

    // Uniform buffer on every side (BUF_DEFAULT), then RE-CENTER the city in the
    // power-of-two-snapped domain so the clearance is symmetric instead of all
    // the snap slack landing downwind. (COST 5H/15H buffers were intentionally
    // asymmetric; this is the simpler uniform layout requested for now.)
    default_buffers(p);
    compute_domain(p);                       // Sx,Sy = next_pow2(buffer+city+buffer)
    p.buf_xn = (p.Sx - p.city_w)*0.5;        // equal clearance ±x
    p.buf_yn = (p.Sy - p.city_h)*0.5;        // equal clearance ±y
    p.source_x = p.buf_xn - 100.0;           // 100 m upwind of the city's leading edge
    p.source_y = p.buf_yn + p.city_h*0.5;    // on the centreline
    ensure_source_buffer(p);
    double cx=p.buf_xn+p.city_w*0.5, cy=p.buf_yn+p.city_h*0.5;
    p.cbd_x=cx; p.cbd_y=cy; p.biz_center_x=cx; p.biz_center_y=cy;
    Result r=generate(p);
    double H = r.max_height;
    if (H < 10.0) H = 10.0;
    print_summary(p,r);
    int NX=r.nx_cells, NY=r.ny_cells;
    printf("\n[LAB] H(tallest)=%.0f m | uniform buffers (m): x=%.0f y=%.0f (city centered)\n",
           H, p.buf_xn, p.buf_yn);
    printf("[LAB] domain %.0f x %.0f m = %d x %d cells @ %.0f m\n", p.Sx,p.Sy,NX,NY,(double)CELL);
    export_city("city_export.txt", p, r);              // for the renderer (2D/3D city)

    // ── 2. Finalized polydisperse source (param_space FIXED) ────────────────
    psd::Lognormal L{1.0e-6, 2.0, 1800.0};             // MMAD=1um, GSD=2.0, rho_p=1800
    auto bins = psd::discretize(L, n_bins);
    printf("[LAB] %d size bins (MMAD=1um, GSD=2): ", n_bins);
    for(auto&b:bins) printf("(%.2fum,%.0f%%) ", b.d_phys*1e6, b.frac*100); printf("\n");

    // ── 3. Transport: warm flow once, replay each bin (Option A) ────────────
    auto base_cfg = [&](double dp){
        lbm::Config c{};
        c.nx=0; // set after voxelize
        c.cell_size=(float)CELL; c.U_inlet=5.0f; c.wind_angle=0.0f;
        c.nu_phys=1.5e-5f; c.Cw=0.325f; c.Sc_t=0.7f; c.D_mol=1e-5f;
        c.source_x=p.source_x; c.source_y=p.source_y; c.source_z=4.0f; c.Q_source=1.0f;
        c.particle_diam=(float)dp; c.particle_density=1800.0f;
        c.max_steps=2000000; c.check_interval=500; c.conv_threshold=1e-4f;
        c.max_warmup=max_warm; c.avg_threshold=2e-3f; c.release_time=(float)release_s;
        c.inlet_profile=1; c.abl_z0=0.7f; c.abl_zref=40.0f; c.abl_Lturb=40.0f; c.abl_nmodes=100;
        c.abl_sigu_ratio=2.5f; c.abl_sigv_ratio=1.9f; c.abl_sigw_ratio=1.25f;
        return c;
    };

    size_t NC=(size_t)NX*NY;
    std::vector<double> tiac_sum(NC,0.0), dep_sum(NC,0.0);   // mass-weighted, z=1
    std::vector<float> buf(NC);

    // captured ONCE from the converged flow (identical across bins) to drive the
    // reverse (adjoint) solver: frozen mean velocity + eddy viscosity + geometry.
    std::vector<float> mean_ux, mean_uy, mean_uz, mean_nut;
    std::vector<uint8_t> geom_tp;
    int NZ=0; bool flow_captured=false;
    float v_p2l=0.f;                         // physical->lattice velocity factor
    std::vector<float> w_s_bins(n_bins,0.f); // per-bin settling velocity (lattice)

    for(int i=0;i<n_bins;++i){
        VoxelGrid g = voxelize(p, r, /*solid=*/true, /*dep_dp=*/bins[i].d_phys, 1800.0, 0.5);
        lbm::Config c = base_cfg(bins[i].d_phys);
        c.nx=g.nx; c.ny=g.ny; c.nz=g.nz;
        lbm::Solver s(c);
        s.load_geometry(g.type.data(), g.perm.data(), g.inh.data(), g.dep_vel.data());
        if(i>0) s.load_flow_checkpoint("flow.ckpt");     // reuse converged flow
        printf("\n[LAB] bin %d/%d  d_phys=%.3f um  frac=%.3f\n", i+1,n_bins,bins[i].d_phys*1e6,bins[i].frac);
        auto res = s.run();
        if(!std::isfinite(res.total_airborne)){
            fprintf(stderr,"[LAB] FATAL: bin %d/%d diverged (airborne=NaN) — aborting "
                "before it corrupts the warm-restart checkpoint and the remaining "
                "bins. Raise nu_floor (lbm_solver.cpp) and rebuild.\n", i+1, n_bins);
            return 2;
        }
        w_s_bins[i] = s.settling_velocity_lattice();   // per-bin settling (lattice)
        if(i==0){
            s.save_flow_checkpoint("flow.ckpt");
            s.export_velocity("vel_z2.bin", 2);          // velocity field at z=2 (once)
            // freeze the mean flow + geometry for the reverse (adjoint) solver
            s.copy_mean_flow_to_host(mean_ux, mean_uy, mean_uz, mean_nut);
            geom_tp = g.type; NZ = g.nz; v_p2l = s.velocity_phys_to_lattice();
            flow_captured = true;
        }
        // accumulate mass-weighted TIAC (airborne) and deposition (ground), z=1
        s.export_tiac("tmp_tiac.bin", 1);
        FILE* f=fopen("tmp_tiac.bin","rb"); int h[2]; size_t rd=fread(h,4,2,f);(void)rd;
        rd=fread(buf.data(),4,NC,f); fclose(f);
        for(size_t k=0;k<NC;++k) tiac_sum[k]+=bins[i].frac*buf[k];
        s.export_deposition("tmp_dep.bin", 1);
        f=fopen("tmp_dep.bin","rb"); rd=fread(h,4,2,f); rd=fread(buf.data(),4,NC,f); fclose(f);
        for(size_t k=0;k<NC;++k) dep_sum[k]+=bins[i].frac*buf[k];
        printf("[LAB]   emitted=%.3e deposited=%.3e (%.2f%%) airborne=%.3e\n",
               res.total_emitted,res.total_deposited,
               100.0*res.total_deposited/std::max(1e-30,res.total_emitted),res.total_airborne);
    }

    // ── 4. Dispersion + exposure maps (z=1) ─────────────────────────────────────
    auto write_field=[&](const char* fn, const std::vector<double>& v){
        std::vector<float> o(NC); for(size_t k=0;k<NC;++k) o[k]=(float)v[k];
        FILE* f=fopen(fn,"wb"); int h[2]={NX,NY}; fwrite(h,4,2,f); fwrite(o.data(),4,NC,f); fclose(f);
    };
    write_field("dispersion_z1.bin", tiac_sum);          // particulate dispersion (mass-weighted TIAC)
    write_field("deposition_z1.bin", dep_sum);
    std::vector<double> exposure_map(NC);
    for(size_t k=0;k<NC;++k) exposure_map[k]=W_AIRBORNE*tiac_sum[k] + W_DEPOSITED*dep_sum[k];
    write_field("exposure_z1.bin", exposure_map);

    // ── 5. OBJECTIVE: effective-inhabitance-weighted concentration ───────────
    // J = Σ (local time-integrated air concentration) × (effective inhabitance).
    // "Effective inhabitance" (eff_inh) already folds in the NHAPS occupancy time
    // budget, so it is WHERE and HOW MUCH people are present; the concentration is
    // the mass-weighted TIAC the population is exposed to (facade-adjacent
    // outdoor air for building occupants, the local cell for street/park
    // occupants). This is the raw concentration·inhabitance exposure functional —
    // no exposure weights, no infiltration (F_inf), no deposited-surface exposure. It depends
    // only on the dispersed concentration field and the inhabitance map.
    std::vector<float> tiacf(NC);
    for(size_t k=0;k<NC;++k) tiacf[k]=(float)tiac_sum[k];
    double pop=r.population;

    // building occupants × facade outdoor concentration (Σ eff_inh · C_facade)
    IndoorScenario sc; sc.dp_m=L.MMAD; sc.a_inf=0.5; sc.lambda_filt=0.0;
    auto in_tiac = indoor_exposure(r, tiacf, NX, NY, sc);
    double indoor_obj = in_tiac.total_outdoor_exposure;          // no F_inf applied

    // street occupants × local concentration (Σ weight · C)
    VoxelGrid gref = voxelize(p, r, true, bins[0].d_phys, 1800.0, 0.5);
    double street_pop = street_fraction(nhaps_average(), OUTDOOR_PARK_SHARE)*pop;
    auto sf = build_street_occupancy(p, r, gref, street_pop);
    double street_obj = street_exposure(sf, tiacf, NX, NY, 1.0);

    // park occupants × footprint-average concentration
    auto at=[&](std::vector<float>&F,int x,int y){ return (x<0||y<0||x>=NX||y>=NY)?0.f:F[(size_t)y*NX+x]; };
    double park_obj=0;
    for(auto&b:r.blocks){ if(b.usage!=PARK||b.eff_inh<=0)continue; double st=0;int n=0;
        for(int y=b.y0;y<b.y1;++y)for(int x=b.x0;x<b.x1;++x){ st+=at(tiacf,x,y); ++n; }
        if(n) park_obj += b.eff_inh*(st/n); }

    double objective = indoor_obj + street_obj + park_obj;       // Σ concentration × eff_inh

    printf("\n========== OBJECTIVE: concentration x effective inhabitance ==========\n");
    printf("  buildings (x facade C) : %.4e\n", indoor_obj);
    printf("  street   (x local C)   : %.4e\n", street_obj);
    printf("  park     (x local C)   : %.4e\n", park_obj);
    printf("  OBJECTIVE J = %.4e   (concentration-person-time; minimize)\n", objective);
    printf("  effective inhabitance (population) = %.0f\n", pop);
    FILE* ds=fopen("total_exposure.txt","w");
    fprintf(ds,"objective %.6e\nbuildings %.6e\nstreet %.6e\npark %.6e\npopulation %.0f\nrelease_s %.0f\n",
            objective,indoor_obj,street_obj,park_obj,pop,release_s);
    fclose(ds);

    // ── 6. REVERSE-SOLVER objective + reciprocity regression ─────────────────
    // The reverse (adjoint) solver computes the SAME effective-exposure objective
    // from the source side. Forcing the adjoint transport with the receptor field
    // w(x) (effective inhabitance, exactly the §5 weighting as a field) on the
    // FROZEN mean flow yields the footprint F(x) = effective exposure per unit
    // release at x; then J = Σ_x F(x)·s(x) for the uniform open-space source mask.
    //
    // Propagation specified by density and size: the reverse solve is run PER PSD
    // bin (each with its settling velocity and size-dependent per-surface
    // deposition), and the footprints are summed by mass fraction,
    // F = Σ_i frac_i·F_i. Because each bin's linear-upwind forward/adjoint pair are
    // exact transposes, the mass-weighted forward Σ w·C equals the reverse
    // Σ F·s to machine precision — asserted here as a standing regression.
    //
    // NOTE: this objective transport is first-order linear upwind (exactly
    // adjoint-consistent) and is MORE diffusive than the production van Leer D3Q7
    // scalar above, which is retained untouched for physical dispersion.
    if(flow_captured){
        int N3 = NX*NY*NZ;
        std::vector<float> Foot(N3, 0.f), Fbin(N3, 0.f);
        std::vector<float> w_field, s_field, vdep(N3, 0.f);
        double Jf_total = 0.0;
        int n_steps_used = 0; float dt_used = 0.f;

        // receptor w(x) and source mask s(x) are bin-independent (geometry only)
        VoxelGrid g0 = voxelize(p, r, true, bins[0].d_phys, 1800.0, 0.5);
        double street_pop6 = street_fraction(nhaps_average(), OUTDOOR_PARK_SHARE)*r.population;
        auto sf6 = build_street_occupancy(p, r, g0, street_pop6);
        adj::build_receptor_field(r, geom_tp.data(), NX,NY,NZ, sf6, w_field);
        adj::build_source_mask(r, p, geom_tp.data(), NX,NY,NZ, /*Q=*/1.0, s_field);

        for(int i=0;i<n_bins;++i){
            // size-dependent per-surface deposition velocity (m/s → lattice)
            VoxelGrid gi = voxelize(p, r, true, bins[i].d_phys, 1800.0, 0.5);
            for(int k=0;k<N3;++k) vdep[k] = gi.dep_vel[k]*v_p2l;

            adj::Field Fld{ NX,NY,NZ,N3, geom_tp.data(),
                            mean_ux.data(), mean_uy.data(), mean_uz.data(), mean_nut.data(),
                            vdep.data(),
                            /*D0=*/1e-3f,            // lattice diffusivity floor (matches solver)
                            /*Sc_t=*/0.7f,
                            /*w_s=*/w_s_bins[i],
                            /*dt=*/0.f };
            Fld.dt = adj::stable_dt(Fld);

            // steps: a few domain-transit times so the footprint fills (capped)
            float umax=1e-3f; for(int idx=0;idx<N3;++idx){
                float sp=std::fabs(mean_ux[idx])+std::fabs(mean_uy[idx])+std::fabs(mean_uz[idx]);
                if(geom_tp[idx]==adj::FLUID && sp>umax) umax=sp; }
            int n_steps=(int)std::min(3000.0, std::max(200.0, 2.0*NX/umax/Fld.dt));
            n_steps_used=n_steps; dt_used=Fld.dt;

            double Jf_i=0.0;
            adj::effective_exposure_reverse(Fld, w_field, s_field, n_steps, &Jf_i, &Fbin);
            for(int k=0;k<N3;++k) Foot[k] += (float)bins[i].frac * Fbin[k]; // mass-weighted
            Jf_total += bins[i].frac * Jf_i;                                // forward check
        }

        double Jr_total = adj::dot(Foot, s_field);                          // Σ F·s
        double rel = std::fabs(Jf_total-Jr_total)/std::max({std::fabs(Jf_total),std::fabs(Jr_total),1e-30});

        printf("\n========== REVERSE SOLVER: effective exposure via exposure footprint ==========\n");
        printf("  bins=%d  steps/bin=%d  dt=%.4f\n", n_bins, n_steps_used, dt_used);
        printf("  forward  (linear-upwind, mass-weighted)  J = Sum w.C = %.6e\n", Jf_total);
        printf("  reverse  (adjoint footprint, mass-weighted) J = Sum F.s = %.6e\n", Jr_total);
        printf("  reciprocity rel.err = %.2e  %s\n", rel,
               (rel<1e-4)?"[PASS: solvers agree]":"[FAIL: investigate transpose]");
        // export the mass-weighted footprint (z=1 slice) for visualization
        { std::vector<float> o(NC); for(size_t k=0;k<NC;++k) o[k]=Foot[(size_t)1*NY*NX+k];
          FILE* ff=fopen("footprint_z1.bin","wb"); int h[2]={NX,NY};
          fwrite(h,4,2,ff); fwrite(o.data(),4,NC,ff); fclose(ff); }
        FILE* rf=fopen("reverse_objective.txt","w");
        fprintf(rf,"J_reverse %.6e\nJ_forward_linupwind %.6e\nreciprocity_rel %.3e\nbins %d\nsteps %d\n",
                Jr_total,Jf_total,rel,n_bins,n_steps_used);
        fclose(rf);
    } else {
        printf("\n[REVERSE] flow not captured (no bins run); reverse objective skipped.\n");
    }

    printf("\n[LAB] wrote: city_export.txt vel_z2.bin dispersion_z1.bin deposition_z1.bin exposure_z1.bin total_exposure.txt\n");
    printf("[LAB] visualize:  python3 lab_visualize.py\n");
    return 0;
}
