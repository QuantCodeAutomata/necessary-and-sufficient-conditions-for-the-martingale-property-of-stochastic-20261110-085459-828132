"""1-D PDE solver for the loss-of-mass function of the 3/2 model (Example 6.3),
used by exp_11 to illustrate Appendix B / Theorem B.1 / Proposition B.1 of
Bayraktar & Xing without solving the full 2-D valuation PDE.

Dimensional reduction (Appendix B, Proposition B.1): the valuation PDE for
u(t,s,y) = E[S_T | S_t=s, Y_t=y] with payoff phi(s,y) = s is, after the time
reversal tau = T - t,

    d/dtau u = L u,     u(0,s,y) = s,

where L is the (s,y) generator of the 3/2 model. Because phi is affine in s
and the model's s-dependence enters L only through terms proportional to s
and s*d/ds (no y-only source term, no s-independent term), the ansatz

    u(tau,s,y) = s * (1 - m(tau,y))

solves this 2-D PDE exactly whenever m solves the 1-D "loss of mass" PDE

    d/dtau m = L~ m,     m(0,y) = 0,

with L~ the Girsanov-shifted generator acting only on y (the same shift that
turns E[S_T] into the survival probability Qtilde_y(tau_D > T) of Lemma 2.1).
Substituting u = s*(1-m) into d/dtau u = L u and using L(s*f(y)) = s*(L~ f)(y)
for this model (verified by direct computation: the s*d/ds term reproduces s
itself, and the mixed/y-only terms act on f(y) exactly as L~) reduces the 2-D
boundary value problem to the 1-D problem above, so there is no need to ever
assemble or solve the full 2-D (s,y) system. S is a true martingale iff
m(tau,y) = 0 for every tau, y (no loss of mass); a strict local martingale
loses mass, m(tau,y) > 0 for some tau > 0, matching Theorem B.1.

For the 3/2 model (Example 6.3), the shifted generator is

    L~ m = 0.5*eta^2*y*d2m/dy2 + (alpha + rho*eta - kappa*y)*dm/dy.

Boundary treatment (implementation_notes, Appendix B): the truncated upper
boundary y_max is a purely artificial numerical cutoff, so it always uses a
"free" (zero discrete second derivative / linear extrapolation) constraint
m[0] - 2*m[1] + m[2] = 0 (applied to the last three grid points at y_max),
instead of imposing a Dirichlet or Neumann value; this keeps the linear
system well-posed without asserting any particular value there.

The lower edge y_min approximates the degenerate point y=0, where the paper
states no boundary condition needs to be imposed. This module implements
that literally with the same free/extrapolation row *whenever the boundary
is of entrance (inaccessible) type*, i.e. whenever is_martingale_3_2 holds
(alpha+rho*eta >= eta^2/2, Fichera drift pointing into the domain, Sec. 3.1):
in that regime the free row is exactly the right numerical no-op, because a
linear, homogeneous, finite-difference boundary relation combined with the
homogeneous generator (L~ annihilates constants) and the homogeneous initial
condition m(0,.)=0 keeps the discrete m identically 0 for all tau (verified
analytically and numerically) -- correctly reproducing "no loss of mass" for
a true martingale, with no boundary value ever asserted.

When the boundary is instead of exit (accessible) type, i.e. is_martingale_3_2
is False, the shifted process reaches y=0 with positive probability and, once
there, has left D=(0, infinity) for good (the exit time tau_D of Lemma 2.1,
already used via the eps-threshold in model_one_factor_vol/exp_1/exp_2): this
is not an arbitrary numerical choice but the direct probabilistic meaning of
"exit", and it forces m(tau, y_min) = 1 for tau > 0 (Dirichlet). No *free*
discretization can ever reproduce this (any consistent finite-difference
derivative approximation, by construction, annihilates constants, so a purely
homogeneous scheme can never move away from the homogeneous initial data),
so a Dirichlet row is used here precisely in, and only in, the regime where
the paper's own Fichera-drift criterion (src/criteria.is_martingale_3_2,
already used throughout the Monte Carlo experiments) says the boundary is
genuinely active.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

from src.criteria import is_martingale_3_2
from src.mc_core import mc_mean_ci
from src.model_one_factor_vol import simulate_factor_and_exponential


def build_shifted_generator_matrix(
    y_grid: np.ndarray, alpha: float, kappa: float, eta: float, rho: float,
) -> sparse.csr_matrix:
    """Tridiagonal matrix of the shifted generator L~ on a (possibly non-uniform) y_grid.

    Interior rows use the standard 3-point non-uniform central-difference
    weights for f'(y_i) and f''(y_i) given h_minus = y_i - y_{i-1} and
    h_plus = y_{i+1} - y_i:

        f'(y_i)  ~= c_minus*f_{i-1} + c_0*f_i + c_plus*f_{i+1}
        f''(y_i) ~= d_minus*f_{i-1} + d_0*f_i + d_plus*f_{i+1}

    with c_minus = -h_plus/(h_minus*(h_minus+h_plus)),
         c_0     = (h_plus-h_minus)/(h_minus*h_plus),
         c_plus  = h_minus/(h_plus*(h_minus+h_plus)),
         d_minus = 2/(h_minus*(h_minus+h_plus)),
         d_0     = -2/(h_minus*h_plus),
         d_plus  = 2/(h_plus*(h_minus+h_plus)).

    The first and last rows are instead a free-boundary constraint
    m[0]-2*m[1]+m[2]=0 (resp. the last three points), see module docstring.
    """
    y = np.asarray(y_grid, dtype=float)
    n = y.size
    assert n >= 3, "need at least 3 grid points to form a free-boundary constraint"

    rows, cols, vals = [], [], []
    for i in range(1, n - 1):
        h_minus = y[i] - y[i - 1]
        h_plus = y[i + 1] - y[i]
        c_minus = -h_plus / (h_minus * (h_minus + h_plus))
        c_0 = (h_plus - h_minus) / (h_minus * h_plus)
        c_plus = h_minus / (h_plus * (h_minus + h_plus))
        d_minus = 2.0 / (h_minus * (h_minus + h_plus))
        d_0 = -2.0 / (h_minus * h_plus)
        d_plus = 2.0 / (h_plus * (h_minus + h_plus))

        diff_coeff = 0.5 * eta ** 2 * y[i]
        drift_coeff = alpha + rho * eta - kappa * y[i]

        rows += [i, i, i]
        cols += [i - 1, i, i + 1]
        vals += [
            diff_coeff * d_minus + drift_coeff * c_minus,
            diff_coeff * d_0 + drift_coeff * c_0,
            diff_coeff * d_plus + drift_coeff * c_plus,
        ]

    # Free (linear-extrapolation) boundary rows: no condition is imposed at
    # y=0 or at the truncated upper edge, per the paper's implementation notes.
    rows += [0, 0, 0]
    cols += [0, 1, 2]
    vals += [1.0, -2.0, 1.0]
    rows += [n - 1, n - 1, n - 1]
    cols += [n - 3, n - 2, n - 1]
    vals += [1.0, -2.0, 1.0]

    return sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))


def solve_loss_of_mass(
    y_grid: np.ndarray, alpha: float, kappa: float, eta: float, rho: float,
    T: float, n_time_steps: int,
) -> np.ndarray:
    """Crank-Nicolson solution of d/dtau m = L~ m, m(0,.) = 0, on [0, T].

    Returns an array of shape (n_time_steps+1, len(y_grid)) with the solution
    at every time layer (row 0 is the initial condition m(0,.) = 0).

    The upper edge y_max always uses the free/linear-extrapolation row of
    build_shifted_generator_matrix (artificial truncation boundary). The
    lower edge y_min uses the same free row when is_martingale_3_2 holds
    (entrance/inaccessible boundary, no condition needed or imposed), and a
    Dirichlet row m(., y_min) = 1 when it does not (exit/accessible
    boundary, where "exit" forces certain loss of mass) -- see the module
    docstring for why a free row alone cannot distinguish the two regimes.
    Both boundary rows are re-applied at every step by overwriting the
    corresponding rows of the Crank-Nicolson system (both the implicit
    matrix and the right-hand side). Since the implicit matrix does not
    depend on tau, its LU factorization is computed once and reused for
    every time step.
    """
    y = np.asarray(y_grid, dtype=float)
    n = y.size
    dt = T / n_time_steps
    lower_boundary_accessible = not is_martingale_3_2(alpha, eta, rho)

    l_tilde = build_shifted_generator_matrix(y, alpha, kappa, eta, rho)
    identity = sparse.eye(n, format="csr")

    lhs = (identity - 0.5 * dt * l_tilde).tolil()
    if lower_boundary_accessible:
        lhs[0, :] = 0.0
        lhs[0, 0] = 1.0
    else:
        lhs[0, :] = l_tilde[0, :]
    lhs[-1, :] = l_tilde[-1, :]
    lu = splu(lhs.tocsc())

    rhs_mat = identity + 0.5 * dt * l_tilde

    m = np.zeros((n_time_steps + 1, n))
    for k in range(n_time_steps):
        rhs = rhs_mat @ m[k]
        rhs[0] = 1.0 if lower_boundary_accessible else 0.0
        rhs[-1] = 0.0
        m[k + 1] = lu.solve(rhs)
    return m


def monte_carlo_loss_of_mass(
    y0: float, alpha: float, kappa: float, eta: float, rho: float,
    T_values: list[float], n_paths: int, n_steps: int, rng: np.random.Generator,
) -> dict[float, tuple[float, float]]:
    """Independent Monte Carlo cross-check of the loss of mass 1 - E_y0[S_T].

    Simulates the original (unshifted) 3/2 model factor and stochastic
    exponential exactly as exp_1/exp_2 do (a=0, b=1 in the one-factor
    power-volatility family of src/model_one_factor_vol.py), and reports, for
    each T in T_values, the pair (mean loss of mass, CI half-width) where the
    loss of mass is estimated as 1 - mean(S_T).
    """
    out: dict[float, tuple[float, float]] = {}
    for T in T_values:
        s_t = simulate_factor_and_exponential(
            alpha, kappa, eta, rho, 0.0, 1.0, y0, T, n_steps, n_paths, rng,
        )
        e_s, ci = mc_mean_ci(s_t)
        out[T] = (1.0 - e_s, ci)
    return out
