"""Two-factor inverse-CIR / 4-2 models (Examples 6.4 and 6.5, used by exp_3
and the multi-factor part of exp_4).

Factors: dY1 = (alpha1 - kappa1*Y1) dt + eta1*sqrt(Y1) dZ1,
         dY2 = (alpha2 - kappa2*Y2) dt + eta2*sqrt(Y2) dZ2, Z1 perp Z2.
Asset:   dS = S*(a*sqrt(Y1) + b*g(Y2)/sqrt(Y1)) dB, B = rho*Z1 + sqrt(1-rho^2)*Bperp.

a = 0 recovers the two-factor inverse-CIR model of Example 6.4; a > 0, b > 0
is the multi-factor 4/2 extension of Example 6.5. The second factor Y2 is
autonomous under both P and the Girsanov shift (its drift does not depend on
Y1), so it is simulated once and reused for both the exponential and the
shifted-survival computation.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from src.mc_core import accumulate_exponential, chunk_sizes, cir_step


def simulate_factor_and_exponential(
    alpha1: float, kappa1: float, eta1: float,
    alpha2: float, kappa2: float, eta2: float,
    rho: float, a: float, b: float, g: Callable[[np.ndarray], np.ndarray],
    y0: tuple[float, float], T: float, n_steps: int, n_paths: int,
    rng: np.random.Generator, chunk_size: int = 20_000,
) -> np.ndarray:
    """Monte Carlo sample of S_T under P for the two-factor model. Returns (n_paths,)."""
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    out = np.empty(n_paths)
    start = 0
    for size in chunk_sizes(n_paths, chunk_size):
        y1 = np.full(size, y0[0])
        y2 = np.full(size, y0[1])
        log_s = np.zeros(size)
        for _ in range(n_steps):
            dz1 = rng.standard_normal(size) * sqrt_dt
            dz2 = rng.standard_normal(size) * sqrt_dt
            dbp = rng.standard_normal(size) * sqrt_dt
            y1_pos = np.maximum(y1, 0.0)
            theta_scalar = a * np.sqrt(y1_pos) + b * g(y2) / np.sqrt(np.maximum(y1_pos, 1e-300))
            theta = np.stack([rho * theta_scalar, np.sqrt(max(1 - rho ** 2, 0.0)) * theta_scalar], axis=-1)
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
    rho: float, a: float, b: float, g: Callable[[np.ndarray], np.ndarray],
    y0: tuple[float, float], T: float, n_steps: int, n_paths: int,
    rng: np.random.Generator, eps: float = 1e-10, chunk_size: int = 20_000,
) -> np.ndarray:
    """Exit indicator of (Ytilde1, Y2) from D=(0,inf)^2 under the shifted measure.

    dYtilde1 = (alpha1 + rho*eta1*b*g(Y2) - (kappa1-rho*eta1*a)*Ytilde1) dt + eta1*sqrt(Ytilde1) dZ1tilde,
    Y2 unchanged (autonomous): methodology step 5 (exp_3) / step 4 (exp_4).
    Returns a boolean (n_paths,) array: True if either factor exited by T.
    """
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    kappa1_tilde = kappa1 - rho * eta1 * a
    out = np.empty(n_paths, dtype=bool)
    start = 0
    for size in chunk_sizes(n_paths, chunk_size):
        y1 = np.full(size, y0[0])
        y2 = np.full(size, y0[1])
        exited = np.zeros(size, dtype=bool)
        for _ in range(n_steps):
            dz1 = rng.standard_normal(size) * sqrt_dt
            dz2 = rng.standard_normal(size) * sqrt_dt
            alpha1_tilde = alpha1 + rho * eta1 * b * g(y2)
            y1 = cir_step(y1, alpha1_tilde, kappa1_tilde, eta1, dt, dz1)
            y2 = cir_step(y2, alpha2, kappa2, eta2, dt, dz2)
            exited |= (y1 <= eps) | (y2 <= eps)
        out[start:start + size] = exited
        start += size
    return out
