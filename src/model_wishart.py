"""2x2 Wishart process boundary accessibility (Proposition 7.2).

Simulates Sigma = [[Sigma11, Sigma12], [Sigma12, Sigma22]] with Q = I, M = 0
and isotropic Omega = omega*I (omega = sqrt(omega_sq)). Under these choices
the matrix SDE dSigma = (Omega Omega^T + M Sigma + Sigma M^T) dt +
sqrt(Sigma) dW Q + Q^T dW^T sqrt(Sigma) reduces to the three coupled scalar
SDEs (exp_10 implementation notes):

    dSigma11 = omega^2 dt + 2*sqrt(Sigma11) dW11
    dSigma22 = omega^2 dt + 2*sqrt(Sigma22) dW22
    dSigma12 =              sqrt(Sigma11) dW12 + sqrt(Sigma22) dW21

where W11, W12, W21, W22 are four independent standard Brownian motions (the
entries of the driving 2x2 matrix Brownian motion W). Discretized by
full-truncation Euler-Maruyama (same convention as mc_core.cir_step):
Sigma11, Sigma22 use max(., 0) under the square root and are floored at 0
after each step, so paths are allowed to actually reach the boundary instead
of being reflected away from it.
"""
from __future__ import annotations

import numpy as np

from src.mc_core import chunk_sizes


def lambda_min_2x2(sigma11: np.ndarray, sigma22: np.ndarray, sigma12: np.ndarray) -> np.ndarray:
    """Smaller eigenvalue of [[sigma11, sigma12], [sigma12, sigma22]], closed form."""
    disc = np.sqrt((sigma11 - sigma22) ** 2 + 4.0 * sigma12 ** 2)
    return (sigma11 + sigma22 - disc) / 2.0


def simulate_min_eigenvalue_paths(
    n: int, omega_sq: float, sigma0: tuple[tuple[float, float], tuple[float, float]],
    T: float, n_steps: int, n_paths: int, rng: np.random.Generator, chunk_size: int = 20_000,
) -> np.ndarray:
    """Running minimum over [0, T] of lambda_min(Sigma_t), one value per path.

    sigma0 is the (nested-tuple) initial Sigma_0; omega = sqrt(omega_sq). Only
    the n=2 case has the closed-form lambda_min used here (Proposition 7.2's
    worked example).
    """
    assert n == 2, "closed-form lambda_min simulation is implemented for n=2 only"
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    sigma11_0, sigma12_0 = float(sigma0[0][0]), float(sigma0[0][1])
    sigma22_0 = float(sigma0[1][1])

    out = np.empty(n_paths)
    start = 0
    for size in chunk_sizes(n_paths, chunk_size):
        sigma11 = np.full(size, sigma11_0)
        sigma22 = np.full(size, sigma22_0)
        sigma12 = np.full(size, sigma12_0)
        running_min = lambda_min_2x2(sigma11, sigma22, sigma12).copy()
        for _ in range(n_steps):
            dw11 = rng.standard_normal(size) * sqrt_dt
            dw12 = rng.standard_normal(size) * sqrt_dt
            dw21 = rng.standard_normal(size) * sqrt_dt
            dw22 = rng.standard_normal(size) * sqrt_dt
            sqrt_s11 = np.sqrt(np.maximum(sigma11, 0.0))
            sqrt_s22 = np.sqrt(np.maximum(sigma22, 0.0))
            sigma11 = np.maximum(sigma11 + omega_sq * dt + 2.0 * sqrt_s11 * dw11, 0.0)
            sigma22 = np.maximum(sigma22 + omega_sq * dt + 2.0 * sqrt_s22 * dw22, 0.0)
            sigma12 = sigma12 + sqrt_s11 * dw12 + sqrt_s22 * dw21
            running_min = np.minimum(running_min, lambda_min_2x2(sigma11, sigma22, sigma12))
        out[start:start + size] = running_min
        start += size
    return out


def boundary_hit_probability(min_eig_paths: np.ndarray, eps: float) -> float:
    """Fraction of paths whose running-minimum lambda_min fell to/below eps."""
    return float(np.mean(min_eig_paths <= eps))
