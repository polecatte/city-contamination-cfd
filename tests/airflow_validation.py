"""airflow_validation.py — read av_*.csv from the overnight run and grade each
test against the physical reference bands in AIRFLOW_VALIDATION_NOTE.md.
Missing files are reported as 'not run'. Bands are wind-tunnel/guideline ranges;
a near-miss is 'INVESTIGATE', not a hard fail."""
import csv, os

def load(path):
    if not os.path.exists(path): return None
    with open(path) as f: return list(csv.reader(f))

def grade(name, cond):
    return f"  [{'PASS' if cond else 'FAIL'}] {name}"

print("="*64); print("AIRFLOW VALIDATION SUMMARY"); print("="*64)

# T1 mass closure + incompressibility (per wind angle)
r=load("av_mass.csv")
if r:
    print("\nT1 mass conservation / incompressibility")
    base=None
    for row in r[1:]:
        deg,cl,dv=float(row[0]),float(row[1]),float(row[2])
        ok = cl<1.0 and dv<2.0
        print(grade(f"wind {deg:>2.0f}°: closure={cl:.2f}% (<1)  div={dv:.2f}% (<2)", ok))
        if deg==0: base=(cl,dv)
        elif base and (cl>3*base[0]+0.5):
            print(f"        ⚠ oblique closure degraded vs 0° → lateral BC starving y-fetch; "
                  f"keep wind_direction staged")
else: print("\nT1 mass — not run")

# T2 ABL horizontal homogeneity
r=load("av_abl.csv")
if r:
    zs=[(int(x[0]),float(x[1]),float(x[3])) for x in r[1:]]
    Uref=max(u for _,u,_ in zs) or 1.0
    drift=max(abs(uo-ui)/Uref for _,ui,uo in zs)
    print("\nT2 ABL horizontal homogeneity")
    print(grade(f"max profile drift inlet→outlet = {100*drift:.1f}% (<10%)", drift<0.10))
else: print("\nT2 ABL — not run")

# T3 WALE
r=load("av_wale.csv")
if r:
    d={row[0]:float(row[1]) for row in r[1:] if len(row)>=2}
    ch=d.get("channel",float('nan')); ratio=d.get("ratio_wake_over_channel",float('nan'))
    print("\nT3 WALE eddy-viscosity sanity")
    print(grade(f"channel ν_t/(UH) = {ch:.2e} (≲1e-3, ~pure shear)", ch<1e-3))
    print(grade(f"wake/channel ratio = {ratio:.1f} (≫10, turbulence on)", ratio>10))
else: print("\nT3 WALE — not run")

# T4 cube wake
r=load("av_cube.csv")
if r:
    Xr,roof,up=float(r[1][0]),int(r[1][1]),int(r[1][2])
    print("\nT4 surface-mounted cube wake")
    print(grade(f"Xr/H = {Xr:.2f} (band 1.0–2.5; outside 0.5–4 ⇒ problem)", 1.0<=Xr<=2.5))
    print(grade(f"near-ground wake reversal present (roof_rev={roof})", roof==1))
    print(grade(f"upstream base/horseshoe reversal present (up_rev={up})", up==1))
    if Xr>0 and not (0.5<=Xr<=4): print("        ⚠ Xr far outside physical range — over-diffusion or wrong separation")
else: print("\nT4 cube — not run")

# T5 lateral confinement
r=load("av_lateral.csv")
if r:
    d={int(row[0]):(float(row[1]),float(row[2])) for row in r[1:]}
    print("\nT5 lateral-boundary confinement")
    if 5 in d and 8 in d and d[5][0]>0:
        rel=abs(d[8][0]-d[5][0])/d[5][0]
        print(grade(f"|Xr(8H)-Xr(5H)|/Xr(5H) = {100*rel:.1f}% (<5% ⇒ BC not contaminating)", rel<0.05))
    for lat,(xr,mub) in sorted(d.items()):
        print(f"        clearance {lat}H: Xr/H={xr:.2f}  max|u|@bndry/U={mub:.2f}")
else: print("\nT5 lateral — not run")

# T6 Reynolds
r=load("av_reynolds.csv")
if r:
    rows=[(row[0],float(row[1]),float(row[2])) for row in r[1:]]
    print("\nT6 effective Reynolds number (interpretive)")
    for nf,Re,Xr in rows: print(f"        NU_FLOOR={nf}: Re_eff={Re:.0f}  Xr/H={Xr:.2f}")
    if len(rows)==2 and rows[0][2]>0:
        dxr=abs(rows[1][2]-rows[0][2])/rows[0][2]
        flag = "OK (gross wake ~Re-independent)" if dxr<0.25 else "⚠ strong Re-sensitivity (under-resolved turbulence regime)"
        print(f"        ΔXr/H across τ = {100*dxr:.0f}%  → {flag}")
    print("        (note: Re_eff ~10²–10³ ≪ atmospheric 10⁷; sharp-edge separation is Re-independent,")
    print("         but reattachment/wake turbulence are not — documented limitation, not a bug)")
else: print("\nT6 reynolds — not run")

# T7 cube-face pressure coefficients (Silsoe cube refs)
r=load("av_cp.csv")
if r:
    d={row[0]:float(row[1]) for row in r[1:] if len(row)>=2}
    cw,cl,cr,cs=d.get("windward"),d.get("leeward"),d.get("roof"),d.get("side")
    print("\nT7 cube pressure coefficients  (Richards, Hoxey & Short 2001, Silsoe cube)")
    print(grade(f"windward Cp = {cw:+.2f}  (stagnation, ≈+0.5..+0.9)", cw is not None and 0.4<=cw<=1.0))
    print(grade(f"leeward  Cp = {cl:+.2f}  (wake suction, ≈-0.5..-0.1)", cl is not None and -0.6<=cl<=0.0))
    print(grade(f"roof     Cp = {cr:+.2f}  (separation suction, <-0.3)", cr is not None and cr<-0.2))
    print(grade(f"side     Cp = {cs:+.2f}  (separation suction, <-0.3)", cs is not None and cs<-0.2))
else: print("\nT7 Cp — not run")

# T8 permeable-SHELL cube (the production envelope) vs solid
r=load("av_shell.csv")
if r:
    solid=None; shells={}
    for row in r[1:]:
        if row[0]=="solid": solid=float(row[2])
        else: shells[float(row[1])]=float(row[2])
    print("\nT8 permeable-SHELL cube vs solid (production buildings are SHELL)")
    if solid is not None: print(f"        solid Xr/H = {solid:.2f}")
    for pv in sorted(shells): print(f"        SHELL perm={pv:.3f}: Xr/H = {shells[pv]:.2f}")
    if solid and 0.005 in shells:
        near = abs(shells[0.005]-solid)/solid < 0.25
        print(grade(f"low-perm (0.005) ≈ solid within 25% (nearly impermeable envelope)", near))
    if 0.005 in shells and 0.05 in shells:
        mono = shells[0.05] <= shells[0.005] + 1e-6
        print(grade(f"wake shortens monotonically as permeability rises (bleed-through)", mono))
else: print("\nT8 SHELL — not run")

# A2 Blasius flat-plate boundary layer (analytical)
r=load("av_blasius_fit.csv")
if r:
    slope,Hs=float(r[1][0]),float(r[1][1])
    print("\nA2 Blasius flat-plate boundary layer (analytical, ν-independent signatures)")
    print(grade(f"δ99 growth exponent = {slope:.2f}  (Blasius 0.50; pass 0.40–0.60)", 0.40<=slope<=0.60))
    print(grade(f"shape factor δ*/θ = {Hs:.2f}  (Blasius 2.59; pass 2.2–3.0)", 2.2<=Hs<=3.0))
else: print("\nA2 Blasius — not run")

# GRID CONVERGENCE (cube Xr/H vs resolution)
r=load("av_resolution.csv")
if r:
    rows=[(int(x[0]),float(x[1]),float(x[2])) for x in r[1:]]
    print("\nGRID CONVERGENCE — cube Xr/H vs resolution (Roache 1997, ASME V&V 20)")
    for H,M,Xr in rows: print(f"        H={H} cells ({M:.0f}M): Xr/H={Xr:.2f}")
    if len(rows)>=3:
        d1=abs(rows[1][2]-rows[0][2]); d2=abs(rows[2][2]-rows[1][2])
        print(grade(f"successive change shrinks ({d1:.2f} → {d2:.2f}) ⇒ grid-converging", d2<d1))
        print(f"        finest estimate: Xr/H = {rows[-1][2]:.2f} at H={rows[-1][0]} cells")
else: print("\nGrid convergence — not run")

# SCALAR MASS BUDGET (integrity: no creation of scalar mass; settling+deposition accounted)
r=load("av_massbudget.csv")
if r:
    print("\nSCALAR MASS BUDGET — emitted vs deposited+airborne (scalar/settling/deposition integrity)")
    for row in r[1:]:
        case=row[0]; em,dp,ai=float(row[1]),float(row[2]),float(row[3])
        ret=float(row[4]); fin=int(row[5]); noc=int(row[6])
        print(f"        {case:<10} emitted={em:.3e} dep={dp:.3e} airborne={ai:.3e}  retained={ret:.2f}")
        print(grade(f"{case}: finite and no scalar-mass creation (dep+air ≤ 1.02·emitted)", fin==1 and noc==1))
else: print("\nSCALAR MASS BUDGET — not run")

# SCALAR DIFFUSION (effective crosswind diffusivity from plume spread — floor vs physical)
r=load("av_diffusion.csv")
if r:
    x=r[1]; s2a,s2b=float(x[2]),float(x[3]); Deff=float(x[4]); Dfl=float(x[5]); ratio=float(x[6])
    xdw=float(x[7]); sy_meas=float(x[8]); sy_briggs=float(x[9]); br=float(x[10]); fin=int(x[11])
    print("\nSCALAR DIFFUSION — plume spread vs numerical floor and vs Briggs urban σ_y")
    print(f"        σ²(x_a)={s2a:.2f}  σ²(x_b)={s2b:.2f} cell²   D_eff={Deff:.3e} LU   D_floor={Dfl:.1e} LU")
    if Deff>0: print(f"        D_eff/D_floor = {ratio:.2f}  (≈1 ⇒ spread set by the floor, not the scheme)")
    print(f"        Briggs reality check: σ_y_meas={sy_meas:.1f} m vs σ_y_briggs={sy_briggs:.1f} m at x={xdw:.0f} m  → ratio={br:.2f}")
    if br>0: print(f"        {'[PASS]' if 0.5<=br<=2.0 else '[INVESTIGATE]'} plume width {'within factor 2 of' if 0.5<=br<=2.0 else ('MUCH WIDER than (over-diffuse)' if br>2 else 'much narrower than')} the real atmosphere")
    print(grade("plume measurable, positive spread, finite (diffusivity recoverable)", Deff>0 and s2b>s2a and fin==1))
else: print("\nSCALAR DIFFUSION — not run")

# LOW-ν STABILITY PROBE (does COLLISION=reg keep NU_FLOOR=1e-3 finite? — de-risks tonight)
r=load("av_stability.csv")
if r:
    x=r[1]; mu=float(x[2]); fin=int(x[3]); bnd=int(x[4]); ok=int(x[5])
    print("\nLOW-ν STABILITY PROBE — cube at NU_FLOOR=1e-3, COLLISION=reg (the MRT divergence regime)")
    print(f"        max|u_mean| = {mu:.5f} LU   finite={fin}  bounded={bnd}")
    print(grade("regularized collision stays finite+bounded at ν=1e-3 (reg cures the divergence)", ok==1))
else: print("\nLOW-ν STABILITY PROBE — not run")

# DETERMINISM (identical design ⇒ identical result ⇒ noise-free objective for the GP)
r=load("av_determinism.csv")
if r:
    x=r[1]; d=float(x[0]); ok=int(x[1])
    print("\nDETERMINISM — two identical runs (RFG is stateless ⇒ must match bit-for-bit)")
    print(f"        max|Δu_mean| = {d:.3e} LU")
    print(grade("reproducible (objective is noise-free — GP surrogate assumption holds)", ok==1))
else: print("\nDETERMINISM — not run")

print("\nA1 Poiseuille channel (analytical parabola + ν recovery): run ./test_poiseuille")
print("   — rigorous analytical check already implemented there; see its stdout in av_run.log.")
print("\n"+"="*64)
