"""Unit tests for the shared Monte Carlo primitives in src/mc_core.py."""
from __future__ import annotations

import numpy as np
import pytest

from src.mc_core import (
    accumulate_exponential,
    cholesky_corr,
    chunk_sizes,
    cir_step,
    correlated_normals,
    make_rng,
    mc_mean_ci,
    ou_step,
)


def test_make_rng_reproducible() -> None:
    rng1 = make_rng(42)
    rng2 = make_rng(42)
    assert np.array_equal(rng1.standard_normal(10), rng2.standard_normal(10))


def test_cholesky_corr_reconstructs_correlation_matrix() -> None:
    corr = np.array([[1.0, -0.5, -0.3], [-0.5, 1.0, 0.6], [-0.3, 0.6, 1.0]])
    L = cholesky_corr(corr)
    reconstructed = L @ L.T
    assert np.allclose(reconstructed, corr, atol=1e-6)


def test_cholesky_corr_rejects_non_psd_matrix() -> None:
    bad = np.array([[1.0, 0.99, -0.99], [0.99, 1.0, 0.99], [-0.99, 0.99, 1.0]])
    with pytest.raises(AssertionError):
        cholesky_corr(bad)


def test_correlated_normals_recovers_target_correlation() -> None:
    corr = np.array([[1.0, 0.6], [0.6, 1.0]])
    L = cholesky_corr(corr)
    rng = make_rng(0)
    draws = correlated_normals(L, 200_000, rng)
    empirical = np.corrcoef(draws.T)
    assert abs(empirical[0, 1] - 0.6) < 0.02


def test_cir_step_full_truncation_never_negative() -> None:
    rng = make_rng(1)
    y = np.full(1000, 0.001)
    # Large negative shocks would push a naive Euler step below zero.
    dz = -np.abs(rng.standard_normal(1000)) * 10.0
    y_next = cir_step(y, alpha=0.1, kappa=1.0, eta=1.0, dt=0.01, dz=dz)
    assert np.all(y_next >= 0.0)


def test_cir_step_matches_closed_form_drift_on_average() -> None:
    # With dz = 0 the step is deterministic: Y + (alpha - kappa*Y)*dt.
    y = np.array([1.0, 2.0, 0.5])
    y_next = cir_step(y, alpha=1.0, kappa=1.0, eta=1.0, dt=0.1, dz=np.zeros(3))
    expected = y + (1.0 - 1.0 * y) * 0.1
    assert np.allclose(y_next, expected)


def test_ou_step_mean_reverts_deterministically() -> None:
    z = np.array([2.0])
    z_next = ou_step(z, speed=1.0, level=0.0, eta=0.0, dt=0.1, dw=np.array([0.0]))
    assert z_next[0] < z[0]


def test_accumulate_exponential_scalar_matches_vector() -> None:
    log_s0 = np.zeros(5)
    theta_scalar = np.full(5, 0.3)
    dw_scalar = np.full(5, 0.1)
    out_scalar = accumulate_exponential(log_s0, theta_scalar, dw_scalar, dt=0.01)

    theta_vec = np.stack([theta_scalar, np.zeros(5)], axis=-1)
    dw_vec = np.stack([dw_scalar, np.zeros(5)], axis=-1)
    out_vec = accumulate_exponential(log_s0, theta_vec, dw_vec, dt=0.01)
    assert np.allclose(out_scalar, out_vec)


def test_accumulate_exponential_zero_integrand_keeps_s_at_one() -> None:
    log_s0 = np.zeros(10)
    out = accumulate_exponential(log_s0, np.zeros(10), np.ones(10), dt=0.5)
    assert np.allclose(np.exp(out), 1.0)


def test_mc_mean_ci_shrinks_with_sample_size() -> None:
    rng = make_rng(2)
    small = rng.standard_normal(100) + 1.0
    large = rng.standard_normal(10_000) + 1.0
    _, half_small = mc_mean_ci(small)
    _, half_large = mc_mean_ci(large)
    assert half_large < half_small


def test_mc_mean_ci_requires_at_least_two_samples() -> None:
    with pytest.raises(AssertionError):
        mc_mean_ci(np.array([1.0]))


def test_chunk_sizes_sum_to_total() -> None:
    sizes = chunk_sizes(105, 20)
    assert sum(sizes) == 105
    assert all(s <= 20 for s in sizes)
    assert sizes == [20, 20, 20, 20, 20, 5]


def test_chunk_sizes_exact_multiple() -> None:
    assert chunk_sizes(60, 20) == [20, 20, 20]


def test_chunk_sizes_rejects_nonpositive_input() -> None:
    with pytest.raises(AssertionError):
        chunk_sizes(0, 10)
    with pytest.raises(AssertionError):
        chunk_sizes(10, 0)
