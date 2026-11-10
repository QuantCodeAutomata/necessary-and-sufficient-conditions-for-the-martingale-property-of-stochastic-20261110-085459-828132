"""Weighted-distance / inverse-distance singularity model (Proposition 5.2,
Example 5.2).

U_t = u + W_t is an m-dimensional standard Brownian motion (m >= 2, so U
never hits the origin a.s. under P) and Z_t is a k-dimensional
Ornstein-Uhlenbeck tangential factor, dZ = -lambda*Z dt + H dV. The
stochastic-exponential integrand is theta(u, z) = -c(z) * u / |u|^2, an
m-vector, driven against W: dS/S = theta . dW.

Simplification (paper's own methodology step "C = 0 (independent for
simplicity)"): the correlation matrix C between W (driving U) and V (driving
Z) is set to 0, so W and V are independent Brownian motions. Under the
Girsanov shift to Qtilde that removes theta, this independence means Z's own
dynamics are unchanged (no covariation between V and theta.dW), while U
acquires the drift -c(Z)*Utilde/|Utilde|^2:
    dUtilde = -c(Z) * Utilde / |Utilde|^2 dt + dWtilde,
    dZ = -lambda*Z dt + H dV   (unshifted).

This repository's config fixes k = EXP8_K = 1 (a single scalar OU factor), so
Z is simulated and passed to c_fn as a (size,)-shaped array throughout.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from src.mc_core import accumulate_exponential, chunk_sizes, ou_step

CFunc = Callable[[np.ndarray], np.ndarray]


def simulate_exponential(
    m: int, k: int, lam: float, h: float, u0: tuple[float, ...], z0: float,
    c_fn: CFunc, T: float, n_steps: int, n_paths: int,
    rng: np.random.Generator, chunk_size: int = 20_000,
) -> np.ndarray:
    """Monte Carlo sample of S_T under P. Returns an (n_paths,) array.

    U is simulated as an explicit stepwise Euler loop (exact for a Brownian
    motion, but kept stepwise so the same dW increments drive both U and the
    stochastic exponential, consistent with the rest of the repository).
    """
    assert k == 1, "this model implementation assumes a single scalar OU factor (EXP8_K == 1)"
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    u0_arr = np.asarray(u0, dtype=float)
    out = np.empty(n_paths)
    start = 0
    for size in chunk_sizes(n_paths, chunk_size):
        u = np.tile(u0_arr, (size, 1))
        z = np.full(size, z0)
        log_s = np.zeros(size)
        for _ in range(n_steps):
            dw = rng.standard_normal((size, m)) * sqrt_dt
            dv = rng.standard_normal(size) * sqrt_dt
            norm_sq = np.maximum(np.sum(u ** 2, axis=-1), 1e-300)
            theta = -c_fn(z)[:, None] * u / norm_sq[:, None]
            log_s = accumulate_exponential(log_s, theta, dw, dt)
            u = u + dw
            z = ou_step(z, lam, 0.0, h, dt, dv)
        out[start:start + size] = np.exp(log_s)
        start += size
    return out


def simulate_shifted_survival(
    m: int, k: int, lam: float, h: float, u0: tuple[float, ...], z0: float,
    c_fn: CFunc, T: float, n_steps: int, n_paths: int,
    rng: np.random.Generator, eps: float = 1e-10, chunk_size: int = 20_000,
) -> np.ndarray:
    """Exit indicator of Utilde from D = R^m \\ {0} under Qtilde.

    dUtilde = -c(Z) * Utilde / |Utilde|^2 dt + dWtilde, Z simulated with its
    original (unshifted) OU parameters since C = 0. |Utilde| is floored at
    eps before dividing; the exit indicator is |Utilde| <= eps at any step.
    Returns a boolean (n_paths,) array.
    """
    assert k == 1, "this model implementation assumes a single scalar OU factor (EXP8_K == 1)"
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    u0_arr = np.asarray(u0, dtype=float)
    out = np.empty(n_paths, dtype=bool)
    start = 0
    for size in chunk_sizes(n_paths, chunk_size):
        u = np.tile(u0_arr, (size, 1))
        z = np.full(size, z0)
        exited = np.zeros(size, dtype=bool)
        for _ in range(n_steps):
            dw = rng.standard_normal((size, m)) * sqrt_dt
            dv = rng.standard_normal(size) * sqrt_dt
            norm = np.maximum(np.sqrt(np.sum(u ** 2, axis=-1)), eps)
            drift = -c_fn(z)[:, None] * u / (norm ** 2)[:, None]
            u = u + drift * dt + dw
            z = ou_step(z, lam, 0.0, h, dt, dv)
            exited |= np.sqrt(np.sum(u ** 2, axis=-1)) <= eps
        out[start:start + size] = exited
        start += size
    return out


def euclidean_projection_integral(c_bar: float, r0: float = 0.1) -> float:
    """Naive Euclidean-projection diagnostic integral for Example 5.2.

    Per Example 5.2: A_pi = 4, B_pi = 6, C_pi = -8*c_bar, so
    beta = (B_pi - A_pi + C_pi) / A_pi = 0.5 - 2*c_bar (constant, since A_pi,
    B_pi, C_pi do not depend on r here). The diagnostic integral is
        J = int_0^{r0} exp(int_u^{r0} beta/s ds) du
          = int_0^{r0} (r0/u)**beta du = int_0^{r0} (u/r0)**(-beta) du,
    which converges (is finite) iff -beta > -1, i.e. beta < 1 (always true
    here since beta <= -3/10 < 1). In closed form,
        J = r0 * int_0^1 x**(-beta) dx = r0 / (1 - beta).

    A finite J means the naive Euclidean-projection test is inconclusive: it
    cannot rule out the boundary being attainable, unlike the sharp
    weighted-distance criterion is_martingale_weighted_distance (Proposition
    5.2), which gives a definite martingale verdict for Example 5.2
    (sup c = 2/5 < (m-2)/2 = 1/2 for m = 3).
    """
    beta = 0.5 - 2 * c_bar
    assert beta < 1.0, "the diagnostic integral diverges for beta >= 1"
    return r0 / (1 - beta)
