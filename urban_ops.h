#pragma once
// urban_ops.h — the per-step work of urban_flow as OpenLB operators (Phase 8, G2).
//
// Every operator here is `any_platform`: the same code runs in OpenLB's CPU loops (OpenMP) and
// as a CUDA kernel. They replace the host loops that urban_flow ran every time step
// (VeloGradRefresh, RoughWall, TopStress, the inlet's defineU sweep, TimeMean::sample and the
// Step-4 inject/deposit/accumulate/flux loops), which on a GPU build would read stale host
// copies or force a full device<->host copy per step.
//
// Correspondence rule: each operator is a line-for-line transcription of the host loop it
// replaces, same arithmetic in the same order, applied to exactly the same cells (the cell
// lists are built on the host once and registered per cell). On a CPU build the two paths
// must therefore agree bitwise; urban_flow keeps the host loops behind HOST_OPS=1 as the
// reference that proves it (tests/device_parity.sh STRICT=1: all outputs identical). What
// remains between CPU and GPU is rounding: nvcc contracts a*b+c into fused multiply-adds and
// the device sin/cos/log differ from glibc's in the last ULPs. The flow amplifies that, so the
// GPU is checked against the CPU build with tolerances (tests/device_parity.sh, lab box).
//
// Data residency: after start-up the DEVICE is the source of truth. The host only reads
// (setProcessingContext(Evaluation), per field where it can) and never pushes lattice state
// during time stepping: OpenLB's Simulation context copies whole arrays host->device with no
// dirty tracking, so a push after the device has advanced would silently roll it back.
//
// Single cuboid only: the neighbour reads (+-4 for the WALE stencil, -2 for the specular
// partner) rely on the block's own padding, not on inter-block communication.

namespace urbanops {

using namespace olb;

// ── per-cell fields (dynamic: allocated on first getField, not part of the descriptor) ──
struct UF_MASK  : public descriptors::FIELD_BASE<1> { };   // bit-packed per-cell flags, see below
struct UF_VBUF  : public descriptors::FIELD_BASE<3> { };   // lattice velocity for the WALE stencil
struct RW_SAVE  : public descriptors::FIELD_BASE<4> { };   // rough-wall snapshot of up-going slots
struct TM_SUM   : public descriptors::FIELD_BASE<6> { };   // time mean: sum u (3), sum u^2 (3), m/s
struct RFG_MODE : public descriptors::FIELD_BASE<10> { };  // one inlet Fourier mode (array element)

// UF_MASK bits
enum : int {
  MASK_FLUID  = 1,        // cell carries fluid (VeloGradRefresh::carriesFluid)
  MASK_WIDE_X = 2,        // the +-4 stencil lies inside [-1, n] along x (index condition only)
  MASK_WIDE_Y = 4,
  MASK_WIDE_Z = 8,
  MASK_RW0    = 16,       // rough-wall partner valid for horizontal direction d (4 bits)
};

// ── parameters ──
struct P_TIME    : public descriptors::FIELD_BASE<1> { };  // inlet: physical time
struct P_RAMP    : public descriptors::FIELD_BASE<1> { };  // inlet: start-up ramp factor
struct P_ABL     : public descriptors::FIELD_BASE<12> { }; // inlet: packed scalars, see InletOp
struct P_NMODES  : public descriptors::TYPED_FIELD_BASE<std::size_t,1> { };
struct P_MODES   : public descriptors::TEMPLATE_FIELD_BASE<std::add_pointer_t,10> { };
struct P_LNZ     : public descriptors::FIELD_BASE<1> { };  // rough wall: ln(z_P/z0)
struct P_KAPPA   : public descriptors::FIELD_BASE<1> { };
struct P_TOPDU   : public descriptors::FIELD_BASE<1> { };  // top stress: u*_lb^2 (ramped)
struct P_TOPDIR  : public descriptors::FIELD_BASE<2> { };
struct P_CVEL    : public descriptors::FIELD_BASE<1> { };  // velocity conversion factor

// ── stages (each holds one operator; executed explicitly, in urban_flow's order) ──
namespace ustage {
struct Inlet    { };
struct WaleVel  { };
struct WaleGrad { };
struct RwSave   { };
struct RwApply  { };
struct Top      { };
struct TMean    { };
}

template <typename DESCRIPTOR>
inline int maskOf(double v) any_platform { return (int)(v + 0.5); }

// D3Q19 slots used by the rough wall: up[d] = (dh_d, +1), upOpp[d] = (-dh_d, +1),
// dh = {(1,0),(-1,0),(0,1),(0,-1)} — the same enumeration as RoughWall::init.
template <typename DESCRIPTOR>
inline int slotOf(int cx, int cy, int cz) any_platform {
  for (int i = 0; i < DESCRIPTOR::q; ++i)
    if (descriptors::c<DESCRIPTOR>(i,0)==cx && descriptors::c<DESCRIPTOR>(i,1)==cy
        && descriptors::c<DESCRIPTOR>(i,2)==cz) return i;
  return -1;
}
// (a function, not a table: a namespace-scope array is host memory, unreadable in device code)
template <typename = void>
constexpr int rwDh(int d, int c) any_platform { return c == 0 ? (d == 0 ? 1 : d == 1 ? -1 : 0)
                                                               : (d == 2 ? 1 : d == 3 ? -1 : 0); }

// ─────────────────────────────────────────────────────────────────────────────
// Inlet: the ABL/RFG velocity (abl_inlet.h ABLInlet::velocity) evaluated per inlet cell on
// the device, then cell.defineU — replaces setBoundaryValues' host defineU sweep.
// P_ABL = { u_star, z0, d, kappa, cos(wind), sin(wind), sig_u*u*, sig_v*u*, sig_w*u*,
//           L_turb, T_scale, conversion factor velocity }, gains in P_ABL? no: see GAINS below.
// ─────────────────────────────────────────────────────────────────────────────
struct P_GAIN : public descriptors::FIELD_BASE<3> { };     // g*gx, g*gy, g*gz (ABLInlet::fluct)

struct InletOp {
  static constexpr OperatorScope scope = OperatorScope::PerCellWithParameters;
  using parameters = meta::list<P_TIME,P_RAMP,P_ABL,P_GAIN,P_NMODES,P_MODES>;
  int getPriority() const { return 0; }

  template <typename CELL, typename PARAMETERS>
  void apply(CELL& cell, PARAMETERS& parameters) any_platform {
    using V = typename CELL::value_t;
    const V t    = parameters.template get<P_TIME>();
    const V ramp = parameters.template get<P_RAMP>();
    const auto a = parameters.template get<P_ABL>();
    const auto g = parameters.template get<P_GAIN>();
    const std::size_t nm = parameters.template get<P_NMODES>();
    const auto m = parameters.template get<P_MODES>();
    const auto x = cell.template getField<descriptors::LOCATION>();
    // mean_speed(z)
    V zz = x[2] - a[2]; if (zz < V(0.1)) zz = V(0.1);
    const V U = (a[0] / a[3]) * util::log((zz + a[1]) / a[1]);
    V ux = U * a[4], uy = U * a[5], uz = V(0);
    // fluct(x,y,z,t)
    V fx = 0, fy = 0, fz = 0;
    const V xs = x[0] / a[9], ys = x[1] / a[9], zs = x[2] / a[9], ts = t / a[10];
    for (std::size_t n = 0; n < nm; ++n) {
      const V arg = m[0][n]*xs + m[1][n]*ys + m[2][n]*zs + m[9][n]*ts;
      const V c = util::cos(arg), s = util::sin(arg);
      fx += m[3][n]*c + m[6][n]*s;
      fy += m[4][n]*c + m[7][n]*s;
      fz += m[5][n]*c + m[8][n]*s;
    }
    fx *= g[0]; fy *= g[1]; fz *= g[2];
    ux += a[6] * fx; uy += a[7] * fy; uz += a[8] * fz;
    V u[3] = { ramp * (ux / a[11]), ramp * (uy / a[11]), ramp * (uz / a[11]) };
    cell.defineU(u);
  }
};

// ─────────────────────────────────────────────────────────────────────────────
// WALE velocity-gradient refresh = VeloGradRefresh, in two passes.
// ─────────────────────────────────────────────────────────────────────────────
struct WaleVelOp {                       // pass 1: u into UF_VBUF on fluid-carrying cells
  static constexpr OperatorScope scope = OperatorScope::PerCell;
  int getPriority() const { return 0; }
  template <typename CELL>
  void apply(CELL& cell) any_platform {
    using V = typename CELL::value_t;
    V u[3] = {0,0,0};
    if (maskOf<void>(cell.template getField<UF_MASK>()) & MASK_FLUID) cell.computeU(u);
    Vector<V,3> uu(u[0],u[1],u[2]);
    cell.template setField<UF_VBUF>(uu);
  }
};

struct WaleGradOp {                      // pass 2: differences into VELO_GRAD on MAT_FLUID
  static constexpr OperatorScope scope = OperatorScope::PerCell;
  int getPriority() const { return 0; }
  template <typename CELL>
  void apply(CELL& cell) any_platform {
    using V = typename CELL::value_t;
    const int mk = maskOf<void>(cell.template getField<UF_MASK>());
    auto U  = [&](int ox, int oy, int oz, int i) -> V {
      return cell.neighbor({ox,oy,oz}).template getField<UF_VBUF>()[i]; };
    auto F  = [&](int ox, int oy, int oz) -> bool {
      return maskOf<void>(cell.neighbor({ox,oy,oz}).template getField<UF_MASK>()) & MASK_FLUID; };
    const auto u0 = cell.template getField<UF_VBUF>();
    Vector<V,9> g;
    for (int j = 0; j < 3; ++j) {
      const int e[3] = { j==0, j==1, j==2 };
      bool wide = mk & (MASK_WIDE_X << j);
      for (int st = 1; st <= 4 && wide; ++st)
        if (!F(st*e[0],st*e[1],st*e[2]) || !F(-st*e[0],-st*e[1],-st*e[2])) wide = false;
      const bool fp = F(e[0],e[1],e[2]), fn = F(-e[0],-e[1],-e[2]);
      for (int i = 0; i < 3; ++i) {
        V d = 0;
        if (wide)
          d = (V(672)*(U(e[0],e[1],e[2],i)-U(-e[0],-e[1],-e[2],i))
             + V(168)*(U(-2*e[0],-2*e[1],-2*e[2],i)-U(2*e[0],2*e[1],2*e[2],i))
             + V(32)*(U(3*e[0],3*e[1],3*e[2],i)-U(-3*e[0],-3*e[1],-3*e[2],i))
             + V(3)*(U(-4*e[0],-4*e[1],-4*e[2],i)-U(4*e[0],4*e[1],4*e[2],i))) / V(840);
        else if (fp && fn) d = V(0.5)*(U(e[0],e[1],e[2],i)-U(-e[0],-e[1],-e[2],i));
        else if (fp)       d = U(e[0],e[1],e[2],i)-u0[i];
        else if (fn)       d = u0[i]-U(-e[0],-e[1],-e[2],i);
        g[3*i+j] = d;
      }
    }
    cell.template setField<descriptors::VELO_GRAD>(g);
  }
};

// ─────────────────────────────────────────────────────────────────────────────
// Rough wall = RoughWall::apply, in two passes (the remap reads partners' pre-remap values).
// ─────────────────────────────────────────────────────────────────────────────
struct RwSaveOp {
  static constexpr OperatorScope scope = OperatorScope::PerCell;
  int getPriority() const { return 0; }
  template <typename CELL>
  void apply(CELL& cell) any_platform {
    using V = typename CELL::value_t;
    using DESCRIPTOR = typename CELL::descriptor_t;
    Vector<V,4> s;
    for (int d = 0; d < 4; ++d) s[d] = cell[slotOf<DESCRIPTOR>(-rwDh(d,0), -rwDh(d,1), 1)];
    cell.template setField<RW_SAVE>(s);
  }
};

struct RwApplyOp {
  static constexpr OperatorScope scope = OperatorScope::PerCellWithParameters;
  using parameters = meta::list<P_LNZ,P_KAPPA>;
  int getPriority() const { return 0; }
  template <typename CELL, typename PARAMETERS>
  void apply(CELL& cell, PARAMETERS& parameters) any_platform {
    using V = typename CELL::value_t;
    using DESCRIPTOR = typename CELL::descriptor_t;
    const int mk = maskOf<void>(cell.template getField<UF_MASK>());
    // (1b) specular remap: slot (c_h,+1) <- partner P-2c_h's saved (-c_h,+1)
    for (int d = 0; d < 4; ++d)
      if (mk & (MASK_RW0 << d))
        cell[slotOf<DESCRIPTOR>(rwDh(d,0), rwDh(d,1), 1)] =
          cell.neighbor({-2*rwDh(d,0), -2*rwDh(d,1), 0}).template getField<RW_SAVE>()[d];
    // (2) log-law wall stress by exact-difference forcing
    const V lnz = parameters.template get<P_LNZ>(), kappa = parameters.template get<P_KAPPA>();
    V rho, u[3]; cell.computeRhoU(rho, u);
    const V ut = util::sqrt(u[0]*u[0] + u[1]*u[1]);
    if (!(ut > 0)) return;
    const V us = kappa*ut/lnz;
    const V du = util::min(us*us, ut);
    V u2[3] = { u[0] - du*u[0]/ut, u[1] - du*u[1]/ut, u[2] };
    for (int i = 0; i < DESCRIPTOR::q; ++i)
      cell[i] += equilibrium<DESCRIPTOR>::secondOrder(i, rho, u2)
               - equilibrium<DESCRIPTOR>::secondOrder(i, rho, u);
  }
};

// Top stress = TopStress::apply
struct TopOp {
  static constexpr OperatorScope scope = OperatorScope::PerCellWithParameters;
  using parameters = meta::list<P_TOPDU,P_TOPDIR>;
  int getPriority() const { return 0; }
  template <typename CELL, typename PARAMETERS>
  void apply(CELL& cell, PARAMETERS& parameters) any_platform {
    using V = typename CELL::value_t;
    using DESCRIPTOR = typename CELL::descriptor_t;
    const V du = parameters.template get<P_TOPDU>();
    const auto dir = parameters.template get<P_TOPDIR>();
    V rho, u[3]; cell.computeRhoU(rho, u);
    V u2[3] = { u[0] + du*dir[0], u[1] + du*dir[1], u[2] };
    for (int i = 0; i < DESCRIPTOR::q; ++i)
      cell[i] += equilibrium<DESCRIPTOR>::secondOrder(i, rho, u2)
               - equilibrium<DESCRIPTOR>::secondOrder(i, rho, u);
  }
};

// Time-mean sample = TimeMean::sample
struct TMeanOp {
  static constexpr OperatorScope scope = OperatorScope::PerCellWithParameters;
  using parameters = meta::list<P_CVEL>;
  int getPriority() const { return 0; }
  template <typename CELL, typename PARAMETERS>
  void apply(CELL& cell, PARAMETERS& parameters) any_platform {
    using V = typename CELL::value_t;
    const V cv = parameters.template get<P_CVEL>();
    V u[3] = {0,0,0}; cell.computeU(u);
    auto s = cell.template getField<TM_SUM>();
    for (int c = 0; c < 3; ++c) { const V v = u[c]*cv; s[c] += v; s[3+c] += v*v; }
    cell.template setField<TM_SUM>(s);
  }
};

// ══════════════════════════ Step 4 (advection-diffusion lattice) ═══════════════════════════
struct AD_SEL   : public descriptors::FIELD_BASE<1> { };  // bit0: Omega source; bits1..: flux code
struct AD_RATE  : public descriptors::FIELD_BASE<1> { };  // deposition fraction per step (S2)
struct AD_THETA : public descriptors::FIELD_BASE<1> { };  // Theta = sum C dt  [lattice C * s]
struct AD_DEP   : public descriptors::FIELD_BASE<1> { };  // deposited mass
struct AD_FLUX  : public descriptors::FIELD_BASE<1> { };  // this cell's share of the CV face flux

struct P_Q     : public descriptors::FIELD_BASE<1> { };
struct P_DT    : public descriptors::FIELD_BASE<1> { };
struct P_WS    : public descriptors::FIELD_BASE<1> { };

namespace ustage {
struct AdFlux    { };
struct AdInject  { };
struct AdDeposit { };
struct AdTheta   { };
struct AdSettle  { };
}

// flux codes (Ops::faceFlux): which post-stream slot this cell contributes, with sign
enum : int { FLUX_NONE = 0, FLUX_DN_PLUS = 1, FLUX_DN_MINUS = 2, FLUX_UP_PLUS = 3, FLUX_UP_MINUS = 4 };

template <typename CELL>
inline typename CELL::value_t adConc(CELL& cell) any_platform {   // Ops::conc: C = sum f~ + 1
  using V = typename CELL::value_t;
  V s = 1; for (int i = 0; i < CELL::descriptor_t::q; ++i) s += cell[i]; return s;
}

struct AdFluxOp {
  static constexpr OperatorScope scope = OperatorScope::PerCell;
  int getPriority() const { return 0; }
  template <typename CELL>
  void apply(CELL& cell) any_platform {
    using V = typename CELL::value_t;
    using DESCRIPTOR = typename CELL::descriptor_t;
    const int code = maskOf<void>(cell.template getField<AD_SEL>()) >> 1;
    const int ip = slotOf<DESCRIPTOR>(1,0,0), im = slotOf<DESCRIPTOR>(-1,0,0);
    V add = 0;
    if      (code == FLUX_DN_PLUS)  add =  (cell[ip] + descriptors::t<V,DESCRIPTOR>(ip));
    else if (code == FLUX_DN_MINUS) add = -(cell[im] + descriptors::t<V,DESCRIPTOR>(im));
    else if (code == FLUX_UP_PLUS)  add =  (cell[im] + descriptors::t<V,DESCRIPTOR>(im));
    else if (code == FLUX_UP_MINUS) add = -(cell[ip] + descriptors::t<V,DESCRIPTOR>(ip));
    cell.template setField<AD_FLUX>(cell.template getField<AD_FLUX>() + add);
  }
};

struct AdInjectOp {
  static constexpr OperatorScope scope = OperatorScope::PerCellWithParameters;
  using parameters = meta::list<P_Q>;
  int getPriority() const { return 0; }
  template <typename CELL, typename PARAMETERS>
  void apply(CELL& cell, PARAMETERS& parameters) any_platform {
    using V = typename CELL::value_t;
    using DESCRIPTOR = typename CELL::descriptor_t;
    const V q = parameters.template get<P_Q>();
    for (int i = 0; i < DESCRIPTOR::q; ++i) cell[i] += descriptors::t<V,DESCRIPTOR>(i)*q;
  }
};

struct AdDepositOp {
  static constexpr OperatorScope scope = OperatorScope::PerCell;
  int getPriority() const { return 0; }
  template <typename CELL>
  void apply(CELL& cell) any_platform {
    using V = typename CELL::value_t;
    using DESCRIPTOR = typename CELL::descriptor_t;
    const V C = adConc(cell), r = cell.template getField<AD_RATE>(), removed = r*C;
    for (int i = 0; i < DESCRIPTOR::q; ++i) {
      const V ti = descriptors::t<V,DESCRIPTOR>(i); cell[i] = (1-r)*(cell[i]+ti)-ti; }
    cell.template setField<AD_DEP>(cell.template getField<AD_DEP>() + removed);
  }
};

struct AdThetaOp {
  static constexpr OperatorScope scope = OperatorScope::PerCellWithParameters;
  using parameters = meta::list<P_DT>;
  int getPriority() const { return 0; }
  template <typename CELL, typename PARAMETERS>
  void apply(CELL& cell, PARAMETERS& parameters) any_platform {
    using V = typename CELL::value_t;
    const V C = adConc(cell);
    cell.template setField<AD_THETA>(cell.template getField<AD_THETA>() + C*parameters.template get<P_DT>());
  }
};

struct AdSettleOp {
  static constexpr OperatorScope scope = OperatorScope::PerCellWithParameters;
  using parameters = meta::list<P_WS>;
  int getPriority() const { return 0; }
  template <typename CELL, typename PARAMETERS>
  void apply(CELL& cell, PARAMETERS& parameters) any_platform {
    auto u = cell.template getField<descriptors::VELOCITY>();
    u[2] -= parameters.template get<P_WS>();
    cell.template setField<descriptors::VELOCITY>(u);
  }
};

} // namespace urbanops
