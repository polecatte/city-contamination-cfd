#!/usr/bin/env python3
"""wind_rose.py — wind directions relative to the city as (ORIENT, WIND_DEG) generator cases.

Convention: phi = the direction the wind blows TOWARD, degrees counter-clockwise from the
city's own +x axis (the layout as built, ORIENT 0). In the domain the wind always enters
through x = 0 (and y = 0 when oblique) at WIND_DEG in [0, 45]; the city is turned by
ORIENT, an exact symmetry of the square cell grid (gen_openlb_geom.cpp):
    k = 0..3 : rotate 90 k deg counter-clockwise
    k = 4..7 : mirror x, then rotate 90 (k-4) deg
The wind as seen by the city is the domain wind mapped back through that transform. Every
phi has exactly one case with WIND_DEG in [0, 45) (or two at the 45 deg diagonals), so a
16-direction rose needs inflow angles 0, 22.5 and 45 deg only.

    python3 tools/wind_rose.py 16          # the cases for a uniform 16-direction rose
    python3 tools/wind_rose.py --case 67.5 # one direction: prints "ORIENT WIND_DEG"
"""
import math, sys


def _rot(v, k):
    x, y = v
    for _ in range(k % 4):
        x, y = -y, x
    return (x, y)


def city_transform(v, orient):
    """Linear part of the generator's ORIENT transform (city frame -> domain frame)."""
    x, y = v
    if orient >= 4:
        x = -x
    return _rot((x, y), orient & 3)


def city_inverse(v, orient):
    """Domain frame -> city frame."""
    x, y = _rot(v, -(orient & 3))
    if orient >= 4:
        x = -x
    return (x, y)


def direction_of(orient, wind_deg):
    """Wind direction phi in the city frame for a generator case."""
    w = (math.cos(math.radians(wind_deg)), math.sin(math.radians(wind_deg)))
    x, y = city_inverse(w, orient)
    return math.degrees(math.atan2(y, x)) % 360.0


def case_for_direction(phi):
    """(orient, wind_deg) with wind_deg in [0, 45] whose wind, seen from the city, blows toward phi.
    The domain wind is the city-frame wind pushed through the transform; take the orientation
    that lands it in the [0, 45] deg wedge (the lowest k at the wedge's edges)."""
    v = (math.cos(math.radians(phi)), math.sin(math.radians(phi)))
    for k in range(8):
        x, y = city_transform(v, k)
        th = math.degrees(math.atan2(y, x))
        if -1e-9 <= th <= 45.0 + 1e-9:
            return k, min(45.0, max(0.0, th))
    raise ValueError(phi)


def uniform_rose(n):
    """[(phi, orient, wind_deg, weight)] for n equally spaced directions."""
    return [(360.0 * i / n, *case_for_direction(360.0 * i / n), 1.0 / n) for i in range(n)]


def _selftest():
    for i in range(720):
        phi = 0.5 * i
        k, th = case_for_direction(phi)
        assert 0.0 <= th <= 45.0, (phi, th)
        back = direction_of(k, th)
        assert abs((back - phi + 180) % 360 - 180) < 1e-9, (phi, k, th, back)
    for k in range(8):                 # inverse really inverts the transform
        for v in ((1.0, 0.0), (0.3, -0.8)):
            w = city_inverse(city_transform(v, k), k)
            assert abs(w[0] - v[0]) < 1e-12 and abs(w[1] - v[1]) < 1e-12
    return True


if __name__ == "__main__":
    _selftest()
    if len(sys.argv) > 2 and sys.argv[1] == "--case":     # one direction -> "ORIENT WIND_DEG"
        k, th = case_for_direction(float(sys.argv[2]))
        print(k, f"{th:g}"); sys.exit(0)
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    print(f"# uniform {n}-direction rose: phi (toward, deg CCW from the city's +x) -> ORIENT, WIND_DEG")
    for phi, k, th, wt in uniform_rose(n):
        print(f"{phi:7.2f}  ORIENT={k}  WIND_DEG={th:g}  weight={wt:.4f}")
