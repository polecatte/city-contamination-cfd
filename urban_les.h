#pragma once
// urban_les.h — WALE subgrid model with OpenLB 1.8's inner products corrected.
//
// OpenLB 1.8 (dynamics/collisionLES.h, detail::WaleEffectiveOmega) has three slips in the
// WALE eddy viscosity of Nicoud & Ducros (1999),
//     nu_t = (C_w Delta)^2 (Sd:Sd)^(3/2) / ( (S:S)^(5/2) + (Sd:Sd)^(5/4) ),
//     Sd_ij = 1/2 (g_ik g_kj + g_jk g_ki) - 1/3 delta_ij g_kl g_lk ,  S = (g + g^T)/2 :
//   - Sd:Sd is accumulated with "=" instead of "+=", so only the (2,2) term survives;
//   - S:S the same;
//   - the trace removed from the squared gradient is sum_i g_ii^2 / 3 instead of
//     g_kl g_lk / 3 (off-diagonal products missing).
// So the stock model's nu_t depends on the zz components alone. This is the same wrapper with
// the sums done properly; it also stores the effective relaxation frequency in
// EFFECTIVE_OMEGA (when the descriptor has it) so the scalar can use the local nu_t.
// Lattice units, Delta = 1, so tau_t = 3 nu_t; LES::SMAGORINSKY carries C_w.

namespace urbanles {

using namespace olb;

template <typename COLLISION, typename DESCRIPTOR, typename MOMENTA, typename EQUILIBRIUM>
struct WaleCorrectedImpl {
  using CollisionO = typename COLLISION::template type<DESCRIPTOR, MOMENTA, EQUILIBRIUM>;

  template <typename CELL, typename PARAMETERS, typename V = typename CELL::value_t>
  V computeEffectiveOmega(CELL& cell, PARAMETERS& parameters) any_platform {
    const V omega = parameters.template get<descriptors::OMEGA>();
    const V cw = parameters.template get<collision::LES::SMAGORINSKY>();
    const auto vg = cell.template getField<descriptors::VELO_GRAD>();   // vg[3i+j] = du_i/dx_j
    V g[3][3];
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) g[i][j] = vg[3 * i + j];
    V g2[3][3];                                       // (g^2)_ij = g_ik g_kj
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) {
      g2[i][j] = 0;
      for (int k = 0; k < 3; ++k) g2[i][j] += g[i][k] * g[k][j];
    }
    const V tr = (g2[0][0] + g2[1][1] + g2[2][2]) / V(3);
    V sdsd = 0, ss = 0;
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) {
      const V sd = V(0.5) * (g2[i][j] + g2[j][i]) - (i == j ? tr : V(0));
      const V s  = V(0.5) * (g[i][j] + g[j][i]);
      sdsd += sd * sd; ss += s * s;
    }
    const V den = util::pow(ss, V(2.5)) + util::pow(sdsd, V(1.25));
    V tauTurb = (den > V(0)) ? V(3) * cw * cw * util::pow(sdsd, V(1.5)) / den : V(0);
    if (!(tauTurb > V(0))) tauTurb = 0;              // also catches NaN
    return V(1) / (V(1) / omega + tauTurb);
  }

  template <typename CELL, typename PARAMETERS, typename V = typename CELL::value_t>
  CellStatistic<V> apply(CELL& cell, PARAMETERS& parameters) any_platform {
    const V omegaEff = computeEffectiveOmega(cell, parameters);
    if constexpr (DESCRIPTOR::template provides<descriptors::EFFECTIVE_OMEGA>()) {
      cell.template setField<descriptors::EFFECTIVE_OMEGA>(omegaEff);
    }
    parameters.template set<descriptors::OMEGA>(omegaEff);
    return CollisionO().apply(cell, parameters);
  }
};

template <typename COLLISION>
struct WaleCorrected {
  using parameters = typename COLLISION::parameters::template include<
    descriptors::OMEGA, collision::LES::SMAGORINSKY>;
  static_assert(COLLISION::parameters::template contains<descriptors::OMEGA>(),
                "COLLISION must be parametrized using relaxation frequency OMEGA");
  static std::string getName() { return "WaleCorrected<" + COLLISION::getName() + ">"; }
  template <typename DESCRIPTOR, typename MOMENTA, typename EQUILIBRIUM>
  using type = WaleCorrectedImpl<COLLISION, DESCRIPTOR, MOMENTA, EQUILIBRIUM>;
};

// Velocity coupling (as OpenLB's NavierStokesAdvectionDiffusionVelocityCoupling) plus a per-cell
// scalar diffusivity from the flow's local eddy viscosity, the usual LES closure for a passive
// scalar:  D(x) = D_mol + (nu_0 + nu_t(x)) / Sc_t,  nu_0 + nu_t = (1/omega_eff - 1/2)/3 read from
// the NSE cell's EFFECTIVE_OMEGA (written by WaleCorrected). Cells without a valid
// EFFECTIVE_OMEGA (0 before the first WALE collision, boundary dynamics) use the base omega,
// i.e. nu_t = 0. tau_D is floored at TAU_MIN for stability and written into the AD cell's OMEGA
// for dynamics::ParameterFromCell<OMEGA, ...>; with TRT_MAGIC > 0 it is converted to the even
// rate the same way step4::prepare does (OpenLB's TRT takes OMEGA = 1/tau+).
struct VelocityEddyDiffusivityCoupling {
  static constexpr OperatorScope scope = OperatorScope::PerCellWithParameters;
  struct OMEGA_NSE : public descriptors::FIELD_BASE<1> { };   // base NSE omega (nu_0)
  struct D_MOL_LB  : public descriptors::FIELD_BASE<1> { };   // molecular diffusivity, lattice
  struct INV_SC_T  : public descriptors::FIELD_BASE<1> { };   // 1 / Sc_t
  struct TAU_MIN   : public descriptors::FIELD_BASE<1> { };   // floor on tau_D
  struct TRT_MAGIC : public descriptors::FIELD_BASE<1> { };   // 0: BGK
  using parameters = meta::list<OMEGA_NSE, D_MOL_LB, INV_SC_T, TAU_MIN, TRT_MAGIC>;

  template <typename CELLS, typename PARAMETERS>
  void apply(CELLS& cells, PARAMETERS& parameters) any_platform {
    using V = typename CELLS::template value_t<names::NavierStokes>::value_t;
    using NSE = typename CELLS::template value_t<names::NavierStokes>::descriptor_t;
    using ADE = typename CELLS::template value_t<names::Concentration0>::descriptor_t;
    auto& cellNSE = cells.template get<names::NavierStokes>();
    auto& cellADE = cells.template get<names::Concentration0>();
    V u[NSE::d] { };
    cellNSE.computeU(u);
    cellADE.template setField<descriptors::VELOCITY>(u);

    const V w0 = parameters.template get<OMEGA_NSE>();
    V w = cellNSE.template getField<descriptors::EFFECTIVE_OMEGA>();
    if (!(w > V(0) && w <= w0)) w = w0;                     // also catches NaN
    const V nu = (V(1) / w - V(0.5)) / V(3);
    V tauD = V(0.5) + descriptors::invCs2<V,ADE>() *
             (parameters.template get<D_MOL_LB>() + nu * parameters.template get<INV_SC_T>());
    const V tauMin = parameters.template get<TAU_MIN>();
    if (tauD < tauMin) tauD = tauMin;
    const V magic = parameters.template get<TRT_MAGIC>();
    const V om = magic > V(0) ? V(1) / (V(0.5) + magic / (tauD - V(0.5))) : V(1) / tauD;
    cellADE.template setField<descriptors::OMEGA>(om);
  }
};

} // namespace urbanles
