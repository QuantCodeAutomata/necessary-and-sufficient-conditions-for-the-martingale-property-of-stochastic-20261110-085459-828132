"""Two correlated square-root (CIR) factors with a boundary singularity at
Y1 = 0 modulated by Y2 (Theorem 6.2, Example 6.6, used by exp_5).

Factors: dY1 = (alpha1 - kappa1*Y1) dt + eta1*sqrt(Y1) dZ1,
         dY2 = (alpha2 - kappa2*Y2) dt + eta2*sqrt(Y2) dZ2,
with d<Z1,Z2> = r12 dt (correlated factor Brownian motions). Asset driver:
B = rho1*Z1 + sqrt(1-rho1^2)*Bperp (no direct Z2 term, Example 6.6 sets the
asset's correlation with Z2 to zero), Bperp independent of (Z1, Z2).

Stochastic-exponential integrand: ell(y) = u*sqrt(y1) + v*g(y2)/sqrt(y1),
theta(y) = ell(y) * (rho1, sqrt(1-rho1^2)) against W = (Z1, Bperp); its
projection onto Z2 is zero because the asset does not load on Z2 directly.
The Girsanov shift therefore adds rho1*eta1*ell(y) to Y1's drift (exactly as
in the one- and two-factor models of model_one_factor_vol.py /
model_multi_factor.py) while leaving Y2's drift unchanged: Y2 is an
autonomous CIR process under both P and Q-tilde, correlated with (shifted)
Y1 only through the r12 correlation of the driving Brownian motions.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from src.mc_core import accumulate_exponential, chunk_sizes, cir_step


def _correlated_z_increments(
    size: int, r12: float, sqrt_dt: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Independent standard-normal draws combined into (dZ1, dZ2, dBperp) with
    Corr(dZ1, dZ2) = r12 and dBperp independent of both."""
    e1 = rng.standard_normal(size)
    e2 = rng.standard_normal(size)
    e3 = rng.standard_normal(size)
    dz1 = e1 * sqrt_dt
    dz2 = (r12 * e1 + np.sqrt(max(1 - r12 ** 2, 0.0)) * e2) * sqrt_dt
    dbp = e3 * sqrt_dt
    return dz1, dz2, dbp


def simulate_factor_and_exponential(
    alpha1: float, kappa1: float, eta1: float,
    alpha2: float, kappa2: float, eta2: float,
    rho1: float, r12: float, u: float, v: float, g: Callable[[np.ndarray], np.ndarray],
    y0: tuple[float, float], T: float, n_steps: int, n_paths: int,
    rng: np.random.Generator, chunk_size: int = 20_000,
) -> np.ndarray:
    """Monte Carlo sample of S_T under P. Returns an (n_paths,) array."""
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    out = np.empty(n_paths)
    start = 0
    for size in chunk_sizes(n_paths, chunk_size):
        y1 = np.full(size, y0[0])
        y2 = np.full(size, y0[1])
        log_s = np.zeros(size)
        for _ in range(n_steps):
            dz1, dz2, dbp = _correlated_z_increments(size, r12, sqrt_dt, rng)
            y1_pos = np.maximum(y1, 0.0)
            ell = u * np.sqrt(y1_pos) + v * g(y2) / np.sqrt(np.maximum(y1_pos, 1e-300))
            theta = np.stack([rho1 * ell, np.sqrt(max(1 - rho1 ** 2, 0.0)) * ell], axis=-1)
            dw = np.stack([dz1, dbp], axis=-1)
            log_s = accumulate_exponential(log_s, theta, dw, dt)
            y1 = cir_step(y1, alpha1, kappa1, eta1, dt, dz1)
            y2 = cir_step(y2, alpha2, kappa2, eta2, dt, dz2)
        out[start:start + size] = np.exp(log_s)
        start += size
    return out


def simulate_shifted_survival(
    alpha1: float, kappa1: float, eta1: float,
    alpha2: float, kappa2: float, eta2: float,
    rho1: float, r12: float, u: float, v: float, g: Callable[[np.ndarray], np.ndarray],
    y0: tuple[float, float], T: float, n_steps: int, n_paths: int,
    rng: np.random.Generator, eps: float = 1e-10, chunk_size: int = 20_000,
) -> np.ndarray:
    """Exit indicator of (Ytilde1, Y2) from D = (0, infinity)^2 under Q-tilde.

    dYtilde1 = (alpha1 + rho1*eta1*v*g(Y2) - (kappa1 - rho1*eta1*u)*Ytilde1) dt
               + eta1*sqrt(Ytilde1) dZ1tilde,
    dY2 unchanged (autonomous CIR; methodology step 5), with Corr(dZ1tilde,dZ2) = r12.
    """
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    kappa1_tilde = kappa1 - rho1 * eta1 * u
    out = np.empty(n_paths, dtype=bool)
    start = 0
    for size in chunk_sizes(n_paths, chunk_size):
        y1 = np.full(size, y0[0])
        y2 = np.full(size, y0[1])
        exited = np.zeros(size, dtype=bool)
        for _ in range(n_steps):
            dz1, dz2, _ = _correlated_z_increments(size, r12, sqrt_dt, rng)
            alpha1_tilde = alpha1 + rho1 * eta1 * v * g(y2)
            y1 = cir_step(y1, alpha1_tilde, kappa1_tilde, eta1, dt, dz1)
            y2 = cir_step(y2, alpha2, kappa2, eta2, dt, dz2)
            exited |= (y1 <= eps) | (y2 <= eps)
        out[start:start + size] = exited
        start += size
    return out
