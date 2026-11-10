"""Absorbed scalar diffusion with correlated OU modulation (Theorem 6.3).

X_t = x + W0_{t up to tau0}, a Brownian motion absorbed at 0
(tau0 = inf{t: x + W0_t <= 0}); Z_t is a scalar OU factor driven by a
Brownian motion correlated with W0 through rho1. The stochastic exponential

    S_t = exp(int_0^t c(Z_s)/X_s dW0_s - 1/2 int_0^t c(Z_s)^2/X_s^2 ds)   for t < tau0
    S_t = 0                                                               for t >= tau0

is simulated under P by an Euler scheme; the density is killed explicitly at
absorption rather than relying on numerical decay of the running integral
(the paper shows int c^2/X^2 ds -> infinity on {tau0 < infinity}, so the true
limit of S_t as t -> tau0 is 0).
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from src.mc_core import accumulate_exponential, chunk_sizes, ou_step

CFunc = Callable[[np.ndarray], np.ndarray]


def simulate_density(
    x0: float, z0: float, lam: float, eta_z: float, rho1: float,
    c_fn: CFunc, T: float, n_steps: int, n_paths: int,
    rng: np.random.Generator, x_floor: float = 1e-10, chunk_size: int = 20_000,
) -> np.ndarray:
    """Monte Carlo sample of S_T under P. Returns (n_paths,) with S_T=0 on absorbed paths."""
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    sqrt_one_minus_rho1_sq = np.sqrt(max(1.0 - rho1 ** 2, 0.0))
    out = np.empty(n_paths)
    start = 0
    for size in chunk_sizes(n_paths, chunk_size):
        x = np.full(size, x0)
        z = np.full(size, z0)
        log_s = np.zeros(size)
        absorbed = np.zeros(size, dtype=bool)
        for _ in range(n_steps):
            e0 = rng.standard_normal(size)
            e1 = rng.standard_normal(size)
            dw0 = e0 * sqrt_dt
            dw1 = rho1 * dw0 + sqrt_one_minus_rho1_sq * e1 * sqrt_dt

            theta = c_fn(z) / np.maximum(x, x_floor)
            log_s = accumulate_exponential(log_s, theta, dw0, dt)

            x = x + dw0
            absorbed = absorbed | (x <= 0)
            x = np.maximum(x, x_floor)

            z = ou_step(z, lam, 0.0, eta_z, dt, dw1)

        s = np.exp(log_s)
        s[absorbed] = 0.0
        out[start:start + size] = s
        start += size
    return out


def c_function(a: float, b: float) -> CFunc:
    """Build c(z) = a + b/(1+z^2) for the EXP9_CASES parametrization.

    inf_z c(z) = a, since b/(1+z^2) ranges over (0, b] and its infimum over
    z in R is 0 as |z| -> infinity (reached for b=0 the function is constant
    a, consistent with both the plain-constant and the strict-local cases).
    """
    return lambda z, a=a, b=b: a + b / (1.0 + z ** 2)
