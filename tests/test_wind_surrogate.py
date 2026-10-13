#!/usr/bin/env python3
"""test_wind_surrogate.py — wind direction as a surrogate input, on a synthetic J with known truth.

    python3 tests/test_wind_surrogate.py        (needs requirements-opt.txt)

What it checks: a GP over (design x, wind direction theta), fed ONE direction per design, can
estimate the rose-averaged objective F(x) = mean over the 16 directions of J(x, theta) better
than taking that one direction's J at face value, and its stated uncertainty is honest. Then a
second direction for the most promising designs only shrinks their uncertainty: the mechanism
by which the optimizer spends extra directions on contenders, not everywhere.

Synthetic truth (fractions of a mean of 1, as a design spread of ~+-10 %):
    J(x, theta) = F(x) + A(x) cos(4 theta) + 0.03 cos(theta - 0.7) + noise(0.5 %)
F(x) is the rose mean (the cos terms average to 0 over the 16 directions); A(x) is a
design-dependent "street-alignment" amplitude, 2-10 %, i.e. as large as the design spread
itself: the hard case. Not the CFD; only the BoTorch machinery is under test.

Kernel: k = k_x + k_x' * k_theta, Matern-5/2 on x (ARD), periodic in theta (period fixed to one
turn). The additive k_x carries the part of J common to all directions; the product carries the
direction-dependent part, whose angular pattern varies smoothly with the design.
Rose mean: BoTorch AppendFeatures (the 16 directions) + ExpectationPosteriorTransform, the same
construction as BoTorch's environmental-variable tutorial (Cakmak et al. 2020).
"""
import math, sys, warnings
import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms import Standardize
from botorch.models.transforms.input import AppendFeatures
from botorch.acquisition.objective import ExpectationPosteriorTransform
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, PeriodicKernel, ScaleKernel
from torch.quasirandom import SobolEngine

warnings.filterwarnings("ignore")
torch.manual_seed(0)
dt = torch.double
D, N_DIR, N_DESIGN, NOISE = 3, 16, 60, 0.005
DIRS = torch.arange(N_DIR, dtype=dt) / N_DIR                  # theta as a fraction of a turn


def F_true(x):            # rose mean: smooth, minimum inside the cube
    return 1 + 0.10 * torch.cos(math.pi * x[:, 0]) * torch.cos(0.5 * math.pi * x[:, 1]) + 0.05 * (x[:, 2] - 0.3) ** 2


def J_true(x, th):        # one wind direction
    A = 0.02 + 0.08 * x[:, 1]                                    # alignment amplitude, design-dependent
    return (F_true(x) + A * torch.cos(2 * math.pi * 4 * th)
            + 0.03 * torch.cos(2 * math.pi * th - 0.7))


def fit(X, Y):
    kx1 = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=D, active_dims=list(range(D))))
    kx2 = MaternKernel(nu=2.5, ard_num_dims=D, active_dims=list(range(D)))
    kth = PeriodicKernel(active_dims=[D])
    kth.period_length = 1.0
    kth.raw_period_length.requires_grad_(False)                 # one turn, fixed
    covar = kx1 + ScaleKernel(kx2 * kth)
    m = SingleTaskGP(X, Y, covar_module=covar, outcome_transform=Standardize(m=1),
                     input_transform=AppendFeatures(feature_set=DIRS.unsqueeze(-1)))
    fit_gpytorch_mll(ExactMarginalLogLikelihood(m.likelihood, m))
    return m


def rose_posterior(m, x):
    p = m.posterior(x, posterior_transform=ExpectationPosteriorTransform(n_w=N_DIR))
    return p.mean.squeeze(-1).detach(), p.variance.clamp_min(0).sqrt().squeeze(-1).detach()


def main():
    # round 1: N_DESIGN designs, ONE random direction each
    Xd = SobolEngine(D, scramble=True, seed=1).draw(N_DESIGN).to(dt)
    th = DIRS[torch.randint(0, N_DIR, (N_DESIGN,))]
    Y = (J_true(Xd, th) + NOISE * torch.randn(N_DESIGN, dtype=dt)).unsqueeze(-1)
    X = torch.cat([Xd, th.unsqueeze(-1)], -1)
    m = fit(X, Y)

    Fd = F_true(Xd)
    mu, sd = rose_posterior(m, Xd)
    err_naive = (Y.squeeze(-1) - Fd).abs()              # take the one direction at face value
    err_gp = (mu - Fd).abs()
    cover = ((mu - Fd).abs() <= 2 * sd).double().mean().item()
    spread = (Fd.max() - Fd.min()).item()
    print(f"[wind] {N_DESIGN} designs x 1 direction; design spread of F = {100*spread:.1f} %")
    print(f"[wind] error in F, one direction taken at face value: mean {100*err_naive.mean():.2f} %, max {100*err_naive.max():.2f} %")
    print(f"[wind] error in F, GP rose posterior:                 mean {100*err_gp.mean():.2f} %, max {100*err_gp.max():.2f} %")
    print(f"[wind] GP 2-sigma interval covers the true F for {100*cover:.0f} % of designs (mean sigma {100*sd.mean():.2f} %)")

    # held-out designs: never run at any direction
    Xt = SobolEngine(D, scramble=True, seed=7).draw(200).to(dt)
    mt, st = rose_posterior(m, Xt)
    Ft = F_true(Xt)
    rank_corr = torch.corrcoef(torch.stack([mt.argsort().argsort().double(), Ft.argsort().argsort().double()]))[0, 1].item()
    print(f"[wind] 200 unseen designs: error mean {100*(mt-Ft).abs().mean():.2f} %, rank correlation with true F {rank_corr:.2f}")

    # round 2: a second direction for the 10 most promising designs only (lowest lower bound).
    # A fixed rule here (45 deg from the first); the driver lets the acquisition choose.
    top = (mu - 2 * sd).argsort()[:10]
    used = th[top]
    th2 = (used + 0.125) % 1.0                                    # 45 deg away: opposite phase of cos(4 theta)
    Y2 = (J_true(Xd[top], th2) + NOISE * torch.randn(10, dtype=dt)).unsqueeze(-1)
    m2 = fit(torch.cat([X, torch.cat([Xd[top], th2.unsqueeze(-1)], -1)]), torch.cat([Y, Y2]))
    mu2, sd2 = rose_posterior(m2, Xd)
    print(f"[wind] +1 direction for the 10 most promising designs: their sigma {100*sd[top].mean():.2f} % -> {100*sd2[top].mean():.2f} %, "
          f"error {100*err_gp[top].mean():.2f} % -> {100*(mu2[top]-Fd[top]).abs().mean():.2f} % "
          f"(other designs' sigma {100*sd[~torch.isin(torch.arange(N_DESIGN), top)].mean():.2f} % -> "
          f"{100*sd2[~torch.isin(torch.arange(N_DESIGN), top)].mean():.2f} %)")

    ok = (err_gp.mean() < 0.5 * err_naive.mean()) and cover >= 0.8 and sd2[top].mean() < sd[top].mean()
    print(f"[wind] {'PASS' if ok else 'FAIL'}: GP error < half the face-value error, coverage >= 80 %, "
          f"extra direction shrinks the contenders' uncertainty")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
