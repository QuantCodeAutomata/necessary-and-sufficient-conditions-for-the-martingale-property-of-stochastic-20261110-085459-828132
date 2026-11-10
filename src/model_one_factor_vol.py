"""One-factor power-volatility models: the 3/2 model (Example 6.3, used by
exp_1 and exp_2) and its 4/2 generalization (Example 6.5, used by exp_4).

Asset: dS = S*(a*sqrt(Y) + b/sqrt(Y)) dB, B = rho*Z + sqrt(1-rho^2)*Bperp.
Factor: dY = (alpha - kappa*Y) dt + eta*sqrt(Y) dZ on D = (0, infinity).

The stochastic-exponential integrand is theta(y) = (a*sqrt(y)+b/sqrt(y)) *
(rho, sqrt(1-rho^2)) against the two-dimensional driver W = (Z, Bperp), so
sigma.theta(y) = rho*eta*(a*y+b) is affine even though the model (a,b both
nonzero) is not (methodology of exp_4). The Girsanov-shifted factor has drift
mu_tilde(y) = alpha + rho*eta*b - (kappa - rho*eta*a)*y.

a = 0, b = 1 recovers the 3/2 model of Example 6.3 (exp_1, exp_2).
"""
from __future__ import annotations

import numpy as np

from src.mc_core import accumulate_exponential, chunk_sizes, cir_step


def simulate_factor_and_exponential(
    alpha: float, kappa: float, eta: float, rho: float, a: float, b: float,
    y0: float, T: float, n_steps: int, n_paths: int, rng: np.random.Generator,
    chunk_size: int = 20_000,
) -> np.ndarray:
    """Monte Carlo sample of S_T under the original measure P.

    Simulates the CIR factor Y with full-truncation Euler-Maruyama and
    accumulates log S_t = int theta.dW - 1/2 int |theta|^2 ds pathwise.
    Returns an (n_paths,) array of S_T samples.
    """
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    out = np.empty(n_paths)
    start = 0
    for size in chunk_sizes(n_paths, chunk_size):
        y = np.full(size, y0)
        log_s = np.zeros(size)
        for _ in range(n_steps):
            dz = rng.standard_normal(size) * sqrt_dt
            dbp = rng.standard_normal(size) * sqrt_dt
            y_pos = np.maximum(y, 0.0)
            theta_scalar = a * np.sqrt(y_pos) + b / np.sqrt(np.maximum(y_pos, 1e-300))
            theta = np.stack([rho * theta_scalar, np.sqrt(max(1 - rho ** 2, 0.0)) * theta_scalar], axis=-1)
            dw = np.stack([dz, dbp], axis=-1)
            log_s = accumulate_exponential(log_s, theta, dw, dt)
            y = cir_step(y, alpha, kappa, eta, dt, dz)
        out[start:start + size] = np.exp(log_s)
        start += size
    return out


def simulate_shifted_survival(
    alpha: float, kappa: float, eta: float, rho: float, a: float, b: float,
    y0: float, T: float, n_steps: int, n_paths: int, rng: np.random.Generator,
    eps: float = 1e-10, chunk_size: int = 20_000,
) -> np.ndarray:
    """Monte Carlo survival indicator for the Girsanov-shifted factor Ytilde.

    dYtilde = (alpha + rho*eta*b - (kappa - rho*eta*a)*Ytilde) dt + eta*sqrt(Ytilde) dZtilde.
    Returns a boolean (n_paths,) array: True if Ytilde ever fell at or below eps
    before T (an exit from D = (0, infinity)), used to estimate
    Qtilde_y(tau_D <= T) = 1 - Qtilde_y(tau_D > T).
    """
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    alpha_tilde = alpha + rho * eta * b
    kappa_tilde = kappa - rho * eta * a
    out = np.empty(n_paths, dtype=bool)
    start = 0
    for size in chunk_sizes(n_paths, chunk_size):
        y = np.full(size, y0)
        exited = np.zeros(size, dtype=bool)
        for _ in range(n_steps):
            dz = rng.standard_normal(size) * sqrt_dt
            y = cir_step(y, alpha_tilde, kappa_tilde, eta, dt, dz)
            exited |= y <= eps
        out[start:start + size] = exited
        start += size
    return out
