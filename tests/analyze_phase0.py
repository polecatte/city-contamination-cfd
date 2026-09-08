#!/usr/bin/env python3
"""analyze_phase0.py - Phase 0 floor pick + verdict.

Physics split (learned from the first H100 run, where steady inflow gave a
laminar wake, nu_t~0, and therefore no Reynolds-independence plateau):

  * Steady inflow (ABL_RFG=0) is used ONLY for the STABILITY WALL and GRID
    CONVERGENCE (verification). It has no turbulent plateau, so we do NOT try to
    read one from it. The chosen floor = reg's LOWEST steady-stable floor (the
    highest Re reg can hold), optionally +MARGIN steps.
  * The reg-vs-HRR REGIME decision is made on a well-averaged TURBULENT (RFG-on)
    case at that floor, judged against the cube reattachment band [XR_LO, XR_HI].

Core (pick + verdict) is stdlib-only; the plot is optional/guarded.

Modes:
  --pick A.csv [--margin N]                 -> print ONLY the chosen floor (or 'none')
  A.csv [--sweepB B] [--conv C] [--fine F]  -> print verdict; write fig_phase0.png
"""
import sys, csv, argparse

XR_LO, XR_HI = 1.0, 2.3          # cube reattachment benchmark band (turbulent, ref ~1.6)

def load(path):
    rows=[]
    try:
        with open(path) as f:
            for r in csv.DictReader(f):
                def num(k):
                    try: return float(r[k])
                    except (KeyError, ValueError): return float("nan")
                def i(k, d):
                    v=num(k); return int(v) if v==v else d
                rows.append(dict(H=i("H",0), nf=num("nu_floor"), tau=num("tau"),
                                 stable=i("stable",0), Xr=num("Xr"), Cd=num("Cd"),
                                 nut=num("nut_mean"), n_avg=num("n_avg"), rfg=i("rfg",1),
                                 collision=(r.get("collision") or "?").strip()))
    except FileNotFoundError:
        pass
    return rows

def rel(a,b):
    if b==0 or b!=b or a!=a: return float("inf")
    return abs(a-b)/abs(b)

def stable_floors(rows):
    return sorted([r for r in rows if r["stable"]==1], key=lambda r:r["nf"])  # ascending nu

def pick_floor(rowsA, margin):
    """Chosen floor = reg's lowest steady-stable floor (max Re), stepped up by
    `margin` floors for stability headroom. Returns (floor_str, info) or (None, why)."""
    st=stable_floors(rowsA)
    if not st:
        return None, "no steady-stable floor in the swept range - nothing runs"
    idx=min(margin, len(st)-1)
    return ("%g"%st[idx]["nf"], "lowest steady-stable nu=%.1e (tau=%.4f)"%(st[0]["nf"],st[0]["tau"]))

def fmt(x, s="%.1e"):
    return s%x if x==x else "nan"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("sweepA")
    ap.add_argument("--pick", action="store_true")
    ap.add_argument("--sweepB"); ap.add_argument("--conv"); ap.add_argument("--fine")
    ap.add_argument("--floor"); ap.add_argument("--margin", type=int, default=0)
    ap.add_argument("--max-warmup", type=int, default=30000)
    a=ap.parse_args()

    rowsA=load(a.sweepA)
    floor, info = pick_floor(rowsA, a.margin)

    if a.pick:
        print(floor if floor else "none")
        if not floor: sys.stderr.write("[analyze] %s\n"%info)
        return

    if a.floor and floor:          # report the floor B/C actually ran at
        floor = a.floor

    print("="*70)
    print("PHASE 0 - stability wall + grid convergence (steady) + regime decision (turbulent)")
    print("="*70)

    # stationarity guard (any stage): n_avg at the cap => average not converged
    allrows=rowsA + (load(a.sweepB) if a.sweepB else []) + (load(a.conv) if a.conv else [])
    capped=[r for r in allrows if r["stable"]==1 and r["n_avg"]==r["n_avg"] and r["n_avg"]>=0.95*a.max_warmup]
    if capped:
        hs=sorted(set((r["H"],r["rfg"]) for r in capped))
        print("STATIONARITY WARNING: %d run(s) hit the averaging cap (~%d): %s"
              %(len(capped), a.max_warmup, ", ".join("H%d/rfg%d"%(h,g) for h,g in hs)))
        print("  -> those Cd/Xr are not converged; raise MAX_WARMUP / SPINUP_FT and re-run before trusting them.")

    # ---- Stage A: stability wall (steady) ----
    st=stable_floors(rowsA)
    div=sorted([r for r in rowsA if r["stable"]==0], key=lambda r:-r["nf"])
    if not st:
        print("Stage A: NO stable floor in the swept range - check the setup (nothing ran).")
        print("VERDICT: cannot proceed.")
        print("="*70); return
    wall_lo=st[0]; first_div=div[0] if div else None
    # operator name comes from the data, not a hardcoded literal: a run of a
    # different collision must not be reported as "reg".
    OP = (rowsA[0].get("collision") or "?") if rowsA else "?"
    print("Stage A (steady, stability + convergence):")
    print("  %s stable down to nu=%.1e (tau=%.4f)%s"
          %(OP, wall_lo["nf"], wall_lo["tau"],
            ("; diverges at nu=%.1e (tau=%.4f)"%(first_div["nf"],first_div["tau"]) if first_div else "")))
    if not div:
        print("  NOTE: %s never diverged in the swept range - the stability wall is BELOW"%OP)
        print("        the lowest floor tried (nu=%.1e). The kernel clamps nu_eff to 1e-3,"%wall_lo["nf"])
        print("        so this is the floor of what the solver permits: %s holds everywhere."%OP)
    print("  chosen floor for Stages B/C = nu=%s (highest Re %s holds)"%(floor,OP))
    if wall_lo["nut"]==wall_lo["nut"] and wall_lo["nut"]<1e-5:
        print("  note: steady wake is laminar here (nu_t/(UH)=%.1e ~ 0) - expected; regime call is Stage B."
              %wall_lo["nut"])

    # ---- Stage B: turbulent regime decision (RFG-on at the chosen floor) ----
    B=load(a.sweepB) if a.sweepB else []
    if B:
        b=B[0]
        print("Stage B (regime decision, RFG-on at nu=%s):"%floor)
        if b["stable"]!=1:
            print("  turbulent inflow DIVERGED at %s's floor limit."%OP)
            if OP.lower().startswith("hrr"):
                print("  VERDICT: HRR INSUFFICIENT at this floor - the turbulent case diverges even")
                print("           under HRR. Try a higher floor, larger sigma margin, or check the")
                print("           inflow (RFG amplitude) before concluding the operator is at fault.")
            else:
                print("  VERDICT: %s INSUFFICIENT - cannot run the turbulent flow at max Re => HRR."%OP)
        else:
            eng = "engaged" if (b["nut"]==b["nut"] and b["nut"]>1e-5) else "weak (~laminar)"
            print("  stable | Xr/H=%.3f (benchmark band %.1f-%.1f) | nu_t/(UH)=%s (SGS %s)"
                  %(b["Xr"], XR_LO, XR_HI, fmt(b["nut"]), eng))
            if b["Xr"]==b["Xr"] and b["Xr"]<=XR_HI:
                print("  VERDICT: %s SUFFICES at nu=%s - turbulent wake reaches the benchmark band."%(OP,floor))
            else:
                print("  VERDICT: %s INSUFFICIENT - Xr=%.2f exceeds the band even at its floor limit"%(OP,b["Xr"]))
                print("           (can't lower nu further: %s diverges below %.1e)%s"
                      %(OP, wall_lo["nf"], " => HRR." if not OP.lower().startswith("hrr") else "."))
    else:
        print("Stage B: (not run) - no turbulent case; regime undecided.")

    # ---- Stage C: grid convergence (steady, verification) ----
    C=load(a.conv) if a.conv else []
    cc=sorted([r for r in C if r["Cd"]==r["Cd"]], key=lambda r:r["H"])
    if len(cc)>=2:
        d=rel(cc[-1]["Cd"], cc[0]["Cd"])
        print("Stage C (steady grid convergence): " +
              "  ".join("Cd(H%d)=%.4f"%(r["H"],r["Cd"]) for r in cc) + "  |dCd|=%.1f%%"%(d*100))
        print("  -> %s"%("clean trend - discretization converging."
                          if d<0.10 else "ragged - refine averaging/spin-up before Phase 1."))
    # ---- Stage C2: fine-grid stability ----
    F=load(a.fine) if a.fine else []
    if F:
        fr=F[0]
        print("Stage C2 (fine grid H=%d): %s at nu=%s%s"
              %(fr["H"], "STABLE" if fr["stable"]==1 else "DIVERGED", floor,
                " - Phase-1 fine grid de-risked." if fr["stable"]==1
                else " - chosen floor does NOT hold at fine grid; raise it or HRR."))
    print("="*70)

    # optional plot
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig,ax=plt.subplots(1,2,figsize=(12,4.4))
        for r in rowsA:
            c="#2ca02c" if r["stable"] else "#d62728"
            if r["Cd"]==r["Cd"]: ax[0].scatter(r["nf"], r["Cd"], c=c, s=90, edgecolors="k", zorder=3)
        if floor:
            ax[0].axvline(float(floor), ls="--", color="b", label="chosen floor")
            ax[0].legend(fontsize=8)
        ax[0].set_xscale("log"); ax[0].invert_xaxis()
        ax[0].set_xlabel("nu_floor (lower => higher Re)"); ax[0].set_ylabel("Cd (steady)")
        ax[0].set_title("Stage A: stability + steady drag (green=stable)")
        if len(cc)>=2:
            ax[1].plot([1.0/r["H"] for r in cc], [r["Cd"] for r in cc], "o-")
            ax[1].set_xlabel("1/H"); ax[1].set_ylabel("Cd"); ax[1].set_title("Stage C: grid convergence")
        fig.tight_layout(); fig.savefig("fig_phase0.png", dpi=130); print("wrote fig_phase0.png")
    except Exception as e:
        print("(plot skipped: %s)"%e)

if __name__=="__main__":
    main()
