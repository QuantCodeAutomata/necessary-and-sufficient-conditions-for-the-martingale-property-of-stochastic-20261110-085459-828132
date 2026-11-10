"""Shared numerical primitives for the Monte Carlo experiments.

All experiments verify necessary-and-sufficient martingale criteria of
Bayraktar & Xing, "Necessary and sufficient conditions for the martingale
property of stochastic exponentials driven by multi-dimensional diffusions",
by simulating a factor diffusion Y under the original measure P and its
Girsanov-shifted counterpart under Q-tilde, and comparing the Monte Carlo
estimate of E_y[S_T] with the survival probability Q-tilde_y(tau_D > T)
(Lemma 2.1 / eq. (2.5)).

Every model-specific module in src/ builds its simulation loop out of the
primitives below (correlated normal draws, full-truncation Euler-Maruyama
steps for CIR/OU factors, running accumulation of the stochastic exponential,
and Wald confidence intervals via scipy.stats). Paths are never stored as a
full (n_paths, n_steps, dim) history: only the current state and the running
accumulators are kept, so memory is O(n_paths x dim) regardless of the number
of time steps, and path batches are processed in chunks to bound peak memory.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def make_rng(seed: int = 0) -> np.random.Generator:
    """Construct a NumPy random Generator with an explicit seed for reproducibility."""
    return np.random.default_rng(seed)


def cholesky_corr(corr: np.ndarray, jitter: float = 1e-12) -> np.ndarray:
    """Cholesky factor L (corr = L L^T) of a correlation matrix.

    Validates positive semi-definiteness (smallest eigenvalue >= -1e-8) before
    factorizing, as required whenever the paper specifies a correlation matrix
    for several driving Brownian motions (e.g. Theorem 6.2, Example 6.6).
    """
    corr = np.asarray(corr, dtype=float)
    eigvals = np.linalg.eigvalsh(corr)
    assert eigvals.min() > -1e-8, f"correlation matrix is not PSD: min eigenvalue {eigvals.min():.3e}"
    return np.linalg.cholesky(corr + jitter * np.eye(corr.shape[0]))


def correlated_normals(L: np.ndarray, n_paths: int, rng: np.random.Generator) -> np.ndarray:
    """One batch of standard-normal draws correlated through Cholesky factor L.

    Returns an (n_paths, m) array of jointly standard-normal increments with
    correlation L @ L.T; multiply by sqrt(dt) at the call site to obtain
    Brownian increments.
    """
    m = L.shape[0]
    z = rng.standard_normal((n_paths, m))
    return z @ L.T


def cir_step(y: np.ndarray, alpha: float, kappa: float, eta: float, dt: float,
             dz: np.ndarray, floor: float = 0.0) -> np.ndarray:
    """One full-truncation Euler-Maruyama step of dY = (alpha - kappa Y) dt + eta sqrt(Y) dZ.

    Full truncation (Lord, Koekkoek & Van Dyk 2010): the square-root
    coefficient uses max(Y, 0) and the updated state is floored, which keeps
    the scheme well defined for CIR-type processes with 2*alpha close to
    eta**2 (Feller boundary case used throughout the paper's examples).
    """
    y_pos = np.maximum(y, 0.0)
    y_next = y + (alpha - kappa * y) * dt + eta * np.sqrt(y_pos) * dz
    return np.maximum(y_next, floor)


def ou_step(z: np.ndarray, speed: float, level: float, eta: float, dt: float,
            dw: np.ndarray) -> np.ndarray:
    """One Euler step of dZ = speed * (level - Z) dt + eta dW (Ornstein-Uhlenbeck)."""
    return z + speed * (level - z) * dt + eta * dw


def accumulate_exponential(log_s: np.ndarray, theta: np.ndarray, dw: np.ndarray, dt: float) -> np.ndarray:
    """Update the running log of a stochastic exponential S = E(int theta . dW).

    Implements the Ito formula log S_t = int theta.dW - 1/2 int |theta|^2 ds
    pathwise, one Euler step at a time: theta and dw share shape (n_paths,) for
    a scalar driver or (n_paths, m) for an m-dimensional driver.
    """
    increment = np.sum(theta * dw, axis=-1) if theta.ndim > 1 else theta * dw
    quad = np.sum(theta ** 2, axis=-1) if theta.ndim > 1 else theta ** 2
    return log_s + increment - 0.5 * quad * dt


def mc_mean_ci(samples: np.ndarray, confidence: float = 0.95) -> tuple[float, float]:
    """Sample mean and the half-width of a normal-approximation confidence interval."""
    samples = np.asarray(samples, dtype=float)
    n = samples.size
    assert n > 1, "need at least two Monte Carlo samples for a confidence interval"
    mean = samples.mean()
    se = samples.std(ddof=1) / np.sqrt(n)
    z = stats.norm.ppf(0.5 + confidence / 2)
    return float(mean), float(z * se)


def chunk_sizes(n_paths: int, chunk_size: int) -> list[int]:
    """Split n_paths into chunks of at most chunk_size, to bound peak memory."""
    assert n_paths > 0 and chunk_size > 0
    full, remainder = divmod(n_paths, chunk_size)
    sizes = [chunk_size] * full
    if remainder:
        sizes.append(remainder)
    return sizes
