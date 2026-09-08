"""anon_exposure.py — a source-anonymous urban exposure functional.

Idea (built from scratch):
  The objective makes NO assumption about where a release happens. A release is
  treated as equally possible anywhere in the fabric (a uniform source field q),
  and the only thing that distinguishes good from bad cities is (a) how the fabric
  PROPAGATES a release and (b) WHERE PEOPLE ARE. Propagation is fully specified by
  two local morphology fields — built density d(x) and characteristic block size
  s(x) — through physically-motivated closures. People are specified by an
  inhabitance map I(x). The objective is the population-weighted, source-averaged
  steady concentration:

      E = < I , C >        with   A[d,s] C = q ,   q = anonymous (uniform) source

  where A is a steady advection-diffusion-removal operator whose coefficients are
  functions of d and s only. E is a pure coupling coefficient (I and q each sum to
  1), so it compares cities at equal population and equal total release — a clean,
  anonymous "how exposed are these people to a release that could be anywhere."

Propagation closures (the whole physical content):
  mixing diffusivity   K = K_bg + aK * U * Lmix,   Lmix = s*(1 + bK*d)
      -> bigger blocks and denser fabric drive stronger mechanical mixing/dilution
  removal rate         lam = aL * (d/s) * U
      -> denser, finer-grained fabric has more surface area per volume -> faster
         loss of airborne material to surfaces (deposition/capture)
  Wind anonymity: average the steady solution over a wind rose.
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

H = 10.0                     # cell size (m)
NX = NY = 80
def K(i, j): return i*NY + j

# closure constants
K_BG, AK, BK, AL = 2.0, 0.15, 2.0, 0.12

def closures(d, s, U):
    Lmix = s*(1.0 + BK*d)
    Kdiff = K_BG + AK*U*Lmix
    lam   = AL*(d/np.maximum(s, 1.0))*U
    return Kdiff, lam

def solve_conc(d, s, U, theta, q):
    """Steady advection-diffusion-removal for one wind direction theta."""
    Ux, Uy = U*np.cos(theta), U*np.sin(theta)
    Kd, lam = closures(d, s, U)
    n = NX*NY
    rows, cols, vals = [], [], []
    b = q.reshape(-1).astype(float).copy()
    def add(r, c, v): rows.append(r); cols.append(c); vals.append(v)
    for i in range(NX):
        for j in range(NY):
            k = K(i, j); diag = lam[i, j]
            # upwind advection (x); inflow ghost C=0, outflow implicit
            if Ux >= 0:
                if i > 0: add(k, K(i-1, j), -Ux/H)
                diag += Ux/H
            else:
                if i < NX-1: add(k, K(i+1, j), Ux/H)
                diag += -Ux/H
            # upwind advection (y)
            if Uy >= 0:
                if j > 0: add(k, K(i, j-1), -Uy/H)
                diag += Uy/H
            else:
                if j < NY-1: add(k, K(i, j+1), Uy/H)
                diag += -Uy/H
            # variable-coefficient diffusion, no-flux at domain edges
            for ii, jj in ((i+1, j), (i-1, j), (i, j+1), (i, j-1)):
                if 0 <= ii < NX and 0 <= jj < NY:
                    Kf = 0.5*(Kd[i, j] + Kd[ii, jj])/(H*H)
                    add(k, K(ii, jj), -Kf); diag += Kf
            add(k, k, diag)
    A = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    return spsolve(A, b).reshape(NX, NY)

def wind_rose(n=8, prevailing=0.0, sharpness=1.6):
    th = np.linspace(0, 2*np.pi, n, endpoint=False)
    w = np.exp(sharpness*np.cos(th - prevailing))
    return th, w/w.sum()

def exposure(d, s, I, U=3.0, rose=None):
    """Source-anonymous, inhabitance-tied exposure functional E = <I, C_rose>."""
    if rose is None: rose = wind_rose()
    th, w = rose
    q = np.ones((NX, NY)); q /= q.sum()          # anonymous (uniform) unit source
    In = I/ I.sum()                               # normalize population
    C = np.zeros((NX, NY))
    for t, wt in zip(th, w):
        C += wt*solve_conc(d, s, U, t, q)
    return float(np.sum(In*C)), C

# ---- morphology generators (fresh, minimal) ----
def field(const):  return np.full((NX, NY), float(const))

def gaussian_map(cx, cy, sig, floor=0.0):
    xs = (np.arange(NX)[:, None]); ys = (np.arange(NY)[None, :])
    r2 = ((xs-cx)**2 + (ys-cy)**2)/(2*sig*sig)
    g = np.exp(-r2)
    return floor + (1-floor)*g

def example_city(seed=0):
    rng = np.random.default_rng(seed)
    c = NX/2
    core = gaussian_map(c, c, NX*0.22)                 # dense core, sparse edge
    d = 0.18 + 0.42*core + 0.05*rng.standard_normal((NX, NY))
    d = np.clip(d, 0.05, 0.75)
    s = 60 - 30*core + 4*rng.standard_normal((NX, NY))  # smaller blocks downtown
    s = np.clip(s, 12, 70)
    I = d*gaussian_map(c, c, NX*0.30, floor=0.15)       # people follow built density, central bias
    return d, s, I
