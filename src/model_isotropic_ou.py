"""Isotropic multi-factor power-volatility model on D = R^d (Example 6.1),
used by exp_7.

Factor: a d-dimensional Ornstein-Uhlenbeck process dY = -kappa*Y dt + eta*dZ,
Z a standard d-dimensional Brownian motion (d >= 2, so Y never hits 0).
Asset driver: B = rho*<Y/|Y|, dZ> + sqrt(1-rho^2)*dBperp, Bperp an
independent scalar Brownian motion. Stochastic exponential: dS/S = |Y|^p dB,
i.e. theta(y) = (rho*|y|^(p-1)*y, sqrt(1-rho^2)*|y|^p) against the
(d+1)-dimensional driver W = (Z, Bperp): the first d components load on dZ,
the last scalar component loads on dBperp.

Example 6.1's criterion (src/criteria.is_martingale_power_volatility):
S is a true martingale iff rho <= 0 or p <= 1.
"""
from __future__ import annotations

import functools
from typing import Callable

import numpy as np
import scipy.integrate as integrate

from src.mc_core import accumulate_exponential, chunk_sizes


def simulate_path_and_exponential(
    kappa: float, eta: float, rho: float, p: float,
    y0: tuple[float, ...], T: float, n_steps: int, n_paths: int,
    rng: np.random.Generator, chunk_size: int = 20_000,
) -> np.ndarray:
    """Monte Carlo sample of S_T under the original measure P.

    Returns an (n_paths,) array. Euler-Maruyama throughout, consistent with
    the rest of the repository (no exact-OU shortcut for the factor).
    """
    d = len(y0)
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    y0_arr = np.asarray(y0, dtype=float)
    out = np.empty(n_paths)
    start = 0
    for size in chunk_sizes(n_paths, chunk_size):
        y = np.broadcast_to(y0_arr, (size, d)).copy()
        log_s = np.zeros(size)
        for _ in range(n_steps):
            dz = rng.standard_normal((size, d)) * sqrt_dt
            dbperp = rng.standard_normal(size) * sqrt_dt

            r = np.maximum(np.linalg.norm(y, axis=1), 1e-300)
            theta_z = rho * (r ** (p - 1))[:, None] * y
            theta_bperp = np.sqrt(max(1 - rho ** 2, 0.0)) * r ** p

            log_s = accumulate_exponential(log_s, theta_z, dz, dt)
            log_s = log_s + (theta_bperp * dbperp - 0.5 * theta_bperp ** 2 * dt)

            y = y - kappa * y * dt + eta * dz
        out[start:start + size] = np.exp(log_s)
        start += size
    return out


def radial_scale_functions(
    d: int, kappa: float, eta: float, rho: float, p: float,
) -> tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]:
    """Scale-function coefficients A(r), B(r) for r = |y|^2/2 under the
    Girsanov-shifted process with drift mu_tilde(y) = (-kappa + rho*eta*|y|^(p-1))*y
    and unchanged diffusion eta*I_d (paper Example 6.1, radial test functions
    for the Feller test at infinity; custom derivation via Ito's formula on
    r = |y|^2/2, not copied from any library):

    A(r) = 2*eta^2*r
    B(r) = d/(2*r) - 2*kappa/eta^2 + 2*rho*(2*r)^((p-1)/2)/eta
    """

    def A(r: np.ndarray) -> np.ndarray:
        return 2.0 * eta ** 2 * r

    def B(r: np.ndarray) -> np.ndarray:
        return d / (2.0 * r) - 2.0 * kappa / eta ** 2 + 2.0 * rho * (2.0 * r) ** ((p - 1) / 2.0) / eta

    return A, B


def feller_test_v_infinity(
    d: int, kappa: float, eta: float, rho: float, p: float,
    r0: float = 1.0, r_values: tuple[float, ...] = (10.0, 1e2, 1e3, 1e4, 1e5),
) -> dict:
    """Finite-truncation numerical proxy for the Feller test at infinity
    (paper Example 6.1 / Proposition 4.2): this does NOT compute the exact
    infinite integral v(infinity), it only evaluates v(R) for a grid of finite
    R and inspects whether the sequence keeps growing (diverging -> true
    martingale, no explosion) or appears to plateau (finite limit -> strict
    local martingale / explosion).

    v(R) = integral_{r0}^{R} (1/C(rho_)) * [integral_{r0}^{rho_} C(s)/A(s) ds] d(rho_)
    where C(r) = exp(integral_{r0}^{r} B(u) du).

    Returns {R: v(R) for R in r_values} plus a boolean "diverging".
    """
    A, B = radial_scale_functions(d, kappa, eta, rho, p)
    # For these config values (small eta relative to kappa), B(r) carries a
    # large negative constant offset -2*kappa/eta^2, so C(r) = exp(int B) can
    # collapse (or blow up) double-exponentially fast; h(rho_) = inner/C(rho_)
    # then genuinely exceeds the float64 range for R well below 1e5. Clipping
    # the exponent before it is ever evaluated keeps every quad evaluation
    # finite (no inf/nan from QUADPACK's internal extrapolation), at the cost
    # of saturating (rather than exactly representing) truly astronomical
    # true values -- acceptable since this function is a finite-truncation
    # proxy for an idealized infinite integral, not an exact computation.
    _EXP_CLIP = 700.0
    k1 = (p + 1.0) / 2.0  # exponent + 1 of the (2r)^((p-1)/2) term in B

    def log_c(r: float) -> float:
        """log_c(r) = integral_{r0}^r B(u) du (log_c(r0) = 0 by construction).

        B is elementary (power/reciprocal/constant terms), so its
        antiderivative is closed-form; using it instead of a nested
        scipy.integrate.quad call removes one whole level of adaptive
        quadrature (log_c would otherwise be re-integrated from scratch at
        every sample point of h's and v's quad calls, which is both slow and
        numerically noisy). Only *differences* of log_c values are ever
        passed to exp (see h below), which keeps every exponentiated
        quantity bounded instead of forming inf/inf = nan.
        """
        log_term = (d / 2.0) * np.log(r / r0)
        linear_term = -(2.0 * kappa / eta ** 2) * (r - r0)
        power_term = (rho / (eta * k1)) * ((2.0 * r) ** k1 - (2.0 * r0) ** k1)
        return log_term + linear_term + power_term

    @functools.lru_cache(maxsize=None)
    def h(rho_: float) -> float:
        """h(rho_) = inner(rho_) / C(rho_), i.e. the outer integrand of v(R)."""
        log_c_rho = log_c(rho_)

        def integrand(s: float) -> float:
            return np.exp(min(log_c(s) - log_c_rho, _EXP_CLIP)) / A(s)

        with np.errstate(over="ignore", invalid="ignore"):
            val, _ = integrate.quad(integrand, r0, rho_, limit=100)
        return val

    results: dict[float, float] = {}
    with np.errstate(over="ignore", invalid="ignore"):
        for R in r_values:
            v_r, _ = integrate.quad(h, r0, R, limit=100)
            # h, and hence v(R), is a monotone-non-decreasing integral of a
            # non-negative integrand; any nan/inf here is a float64-overflow
            # artifact of a genuinely huge finite quantity, not an
            # indeterminate value, so it is reported as a large sentinel.
            if not np.isfinite(v_r):
                v_r = 1e300
            results[R] = v_r

    sorted_r = sorted(r_values)
    r_last, r_second_last = sorted_r[-1], sorted_r[-2]
    v_last, v_second_last = results[r_last], results[r_second_last]
    # Numerical proxy: still growing roughly linearly in R (slope not
    # collapsing to ~0) over the last two grid points -> treat as diverging.
    slope = (v_last - v_second_last) / (r_last - r_second_last)
    still_growing = slope * (r_last - r_second_last) > 0.1 * max(abs(v_last), 1e-12)
    # Saturation override: once v(R) has grown far beyond any plausible
    # finite Feller-test limit for this proxy's clip, the exponent-clipping
    # above can make a truly divergent sequence plateau numerically; treat
    # such extreme magnitudes as diverging regardless of the local slope.
    saturated = v_last > 1e80
    diverging = bool(still_growing or saturated)

    out: dict = {R: results[R] for R in r_values}
    out["diverging"] = diverging
    return out
