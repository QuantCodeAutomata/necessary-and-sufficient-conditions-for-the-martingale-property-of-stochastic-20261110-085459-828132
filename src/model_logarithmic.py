"""Critical CIR factor coupled to an independent-driver OU factor through a
logarithmic singularity (Theorem 3.2, Corollary 3.2, used by exp_6).

Factors: dX = (1/2 - kappa*X) dt + sqrt(X) dW0 (critical CIR, 2*alpha = eta^2
with eta = 1, i.e. alpha = 1/2), dZ = -lambda*Z dt + eta_z dW1, with
Corr(dW0, dW1) = rho1. The stochastic-exponential integrand loads only on
W0: theta0(x, z) = -c(z) * h(x) / sqrt(x), c(z) = a + b*z^2/(1+z^2) (sup c =
a + b, used by src.criteria.is_martingale_coupled_log /
is_uniformly_integrable_coupled_log).

Analytic note (not verified numerically here): Theorem 3.2 shows condition
(T) at the boundary x = 0 fails for every c > 0 in this family, yet S remains
a true martingale on every finite horizon whenever sup c <= 1/2; this is the
point of the paper's necessary-and-sufficient refinement over condition (T).
"""
from __future__ import annotations

import math

import numpy as np

from src.mc_core import accumulate_exponential, chunk_sizes, cir_step, ou_step

# h(x) interpolation boundary points. The paper only specifies the two
# endpoint values (1/log(1/x) below exp(-2), 0 above exp(-1)); the linear
# interpolation on the intermediate range is a numerical choice made here to
# obtain a continuous, easily evaluated h, not a claim about the paper.
_H_LOW = math.exp(-2.0)
_H_HIGH = math.exp(-1.0)
_H_LOW_VAL = 1.0 / math.log(1.0 / _H_LOW)  # = 0.5, matches the log branch at x = exp(-2)
_H_HIGH_VAL = 0.0


def h_function(x: np.ndarray, floor: float = 0.0) -> np.ndarray:
    """h(x) = 1/log(1/x) for x <= exp(-2), 0 for x >= exp(-1), linear in between.

    x is floored from below at `floor` before evaluating the log, to avoid
    log(0) or division by zero for degenerate inputs.
    """
    x = np.maximum(np.asarray(x, dtype=float), floor)
    x_safe = np.maximum(x, 1e-300)
    with np.errstate(divide="ignore"):
        # log(1/x_safe) can be 0 (x_safe == 1) or the result discarded by
        # np.where below; the division-by-zero warning is harmless here.
        log_branch = 1.0 / np.log(1.0 / x_safe)
    interp = _H_LOW_VAL + (_H_HIGH_VAL - _H_LOW_VAL) * (x - _H_LOW) / (_H_HIGH - _H_LOW)
    return np.where(x <= _H_LOW, log_branch, np.where(x >= _H_HIGH, _H_HIGH_VAL, interp))


def c_function(z: np.ndarray, a: float, b: float) -> np.ndarray:
    """c(z) = a + b*z^2/(1+z^2); sup c = a + b (attained as |z| -> infinity)."""
    z2 = np.asarray(z, dtype=float) ** 2
    return a + b * z2 / (1.0 + z2)


def _correlated_w_increments(
    size: int, rho1: float, sqrt_dt: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Independent standard-normal draws combined into (dW0, dW1) with Corr(dW0, dW1) = rho1."""
    e0 = rng.standard_normal(size)
    e1 = rng.standard_normal(size)
    dw0 = e0 * sqrt_dt
    dw1 = (rho1 * e0 + math.sqrt(max(1.0 - rho1 ** 2, 0.0)) * e1) * sqrt_dt
    return dw0, dw1


def simulate_coupled_log_exponential(
    a: float, b: float,
    kappa: float, lam: float, eta_z: float, rho1: float,
    x0: float, z0: float, floor: float,
    T: float, n_steps: int, n_paths: int,
    rng: np.random.Generator, chunk_size: int = 20_000,
) -> np.ndarray:
    """Monte Carlo sample of S_T under P. Returns an (n_paths,) array.

    No absorption/kill switch is needed: X is the critical CIR process
    itself (the boundary singularity lives at X = 0 inside the integrand),
    so S_T is simply read off at T after running the Euler loop to maturity.
    """
    dt = T / n_steps
    sqrt_dt = math.sqrt(dt)
    out = np.empty(n_paths)
    start = 0
    for size in chunk_sizes(n_paths, chunk_size):
        x = np.full(size, x0)
        z = np.full(size, z0)
        log_s = np.zeros(size)
        for _ in range(n_steps):
            dw0, dw1 = _correlated_w_increments(size, rho1, sqrt_dt, rng)
            x_safe = np.maximum(x, floor)
            theta0 = -c_function(z, a, b) * h_function(x_safe, floor) / np.sqrt(x_safe)
            log_s = accumulate_exponential(log_s, theta0, dw0, dt)
            x = cir_step(x, 0.5, kappa, 1.0, dt, dw0, floor=floor)
            z = ou_step(z, lam, 0.0, eta_z, dt, dw1)
        out[start:start + size] = np.exp(log_s)
        start += size
    return out
