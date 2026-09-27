#pragma once
// abl_inlet_olb.h — OpenLB binding for the verified ABL/RFG inlet (abl_inlet.h).
//
// Wraps abl::ABLInlet as an OpenLB AnalyticalF3D<T,T> so it can drive a velocity
// boundary via sLattice.defineU(superGeometry, MAT_INLET, ablF) each timestep. The
// underlying field is stateless/closed-form and was numerically verified (abl_inlet_verify):
//   mean profile 0.10% vs log law · divergence 5.2% · unit-variance within 7% · deterministic.
//
// Only this thin adapter is OpenLB-specific; the physics lives in abl_inlet.h unchanged.
// Compiles inside the OpenLB 1.8 build tree (needs olb3D.h). Flags marked CONFIRM 1.8
// are the version-sensitive spots — verify against the 1.8 User Guide.

#include "abl_inlet.h"

#ifdef URBAN_FLOW_WITH_OPENLB
namespace bridge {

// AnalyticalF3D that returns the inlet velocity in LATTICE units at a given physical
// time. Rebuild/refresh the time each step with setTime(physTime) before defineU.
//
// The base is `olb::AnalyticalF3D<T,T>` -- FULLY QUALIFIED. urban_flow.cpp includes this
// header BEFORE its `using namespace olb;`, so nothing from olb is visible unqualified here
// and an unqualified AnalyticalF3D fails with "expected template-name before '<' token".
// That error is about qualification, not about the name: 1.8 does provide AnalyticalF3D<T,T>
// (examples/turbulence/nozzle3d/nozzle3d.cpp:244 uses it). Keeping this header
// self-contained rather than depending on the includer's using-directives.
// operator() signature is `bool operator()(T output[], const T input[])`.
// `input` carries PHYSICAL coordinates when the functor is applied via defineU over a
// SuperGeometry material (check whether your defineU passes physical or lattice coords;
// if lattice, multiply input by converter.getPhysDeltaX()).
template <typename T, typename DESCRIPTOR, typename CONVERTER>
class AblVelocityF3D : public olb::AnalyticalF3D<T,T> {
public:
    AblVelocityF3D(const CONVERTER& converter, const abl::ABLInlet& inlet,
                   T physTime = 0, T ramp = 1)
        : olb::AnalyticalF3D<T,T>(3), _conv(converter), _inlet(inlet), _t(physTime), _ramp(ramp) {}

    void setTime(T physTime) { _t = physTime; }
    // CORRECTIVE — startup ramp: scale the whole inlet by a smoothstep factor in [0,1]
    // over the first spin-up flow-through so the domain fills without a pressure shock.
    // The MEAN and the fluctuation are scaled together (the profile shape is preserved).
    void setRamp(T r) { _ramp = (r < 0 ? 0 : (r > 1 ? 1 : r)); }
    static T smoothstep(T x) { if (x<=0) return 0; if (x>=1) return 1; return x*x*(3-2*x); }

    bool operator()(T output[], const T input[]) override {
        double ux, uy, uz;
        // input[] = physical (x,y,z) in metres; _t = physical time in seconds
        _inlet.velocity(input[0], input[1], input[2], _t, ux, uy, uz);
        // CONFIRM 1.8: convert physical velocity → lattice velocity for defineU.
        output[0] = _ramp * _conv.getLatticeVelocity(ux);     // CONFIRM 1.8 helper name
        output[1] = _ramp * _conv.getLatticeVelocity(uy);
        output[2] = _ramp * _conv.getLatticeVelocity(uz);
        return true;
    }
private:
    const CONVERTER& _conv;
    abl::ABLInlet     _inlet;   // copy: stateless, cheap, thread-safe
    T                 _t;
    T                 _ramp;
};

} // namespace bridge
#endif // URBAN_FLOW_WITH_OPENLB
