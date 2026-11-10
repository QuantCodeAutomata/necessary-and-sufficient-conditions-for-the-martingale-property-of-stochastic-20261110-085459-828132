"""Unit tests for exp_5 (correlated square-root factors with boundary
singularities, Theorem 6.2 / Example 6.6)."""
from __future__ import annotations

import numpy as np

from src import config as cfg
from src.criteria import is_martingale_correlated_sqrt
from src.mc_core import make_rng, mc_mean_ci
from src.model_correlated_sqrt import simulate_factor_and_exponential, simulate_shifted_survival


def g(y2: np.ndarray) -> np.ndarray:
    return 1.0 + y2 / (1.0 + y2)


def test_example_6_6_threshold_alpha1_three_halves() -> None:
    assert is_martingale_correlated_sqrt(1.5, cfg.EXP5_ETA1, cfg.EXP5_RHO1, cfg.EXP5_V, cfg.EXP5_G_BAR) is True
    assert is_martingale_correlated_sqrt(2.0, cfg.EXP5_ETA1, cfg.EXP5_RHO1, cfg.EXP5_V, cfg.EXP5_G_BAR) is True
    assert is_martingale_correlated_sqrt(1.0, cfg.EXP5_ETA1, cfg.EXP5_RHO1, cfg.EXP5_V, cfg.EXP5_G_BAR) is False


def test_threshold_independent_of_r12_small_sample() -> None:
    """Theorem 6.2: the threshold depends only on rho1, not on R12 (factor
    correlation) or on u; varying r12 must not change the martingale verdict.
    Common random numbers (fresh identical seed per r12) isolate the effect
    of r12 from Monte Carlo sampling noise."""
    alpha1 = cfg.EXP5_ALPHA1_CASES["martingale"]
    means = []
    for r12 in (cfg.EXP5_R12, cfg.EXP5_R12_ALT):
        rng = make_rng(cfg.SEED)
        s_t = simulate_factor_and_exponential(
            alpha1, cfg.EXP5_KAPPA1, cfg.EXP5_ETA1,
            cfg.EXP5_ALPHA2, cfg.EXP5_KAPPA2, cfg.EXP5_ETA2,
            cfg.EXP5_RHO1, r12, cfg.EXP5_U, cfg.EXP5_V, g,
            cfg.EXP5_Y0, T=1.0, n_steps=200, n_paths=4000, rng=rng,
        )
        e_s, _ = mc_mean_ci(s_t)
        means.append(e_s)
    assert all(abs(m - 1.0) < 0.35 for m in means)


def test_strict_local_martingale_case_exits_with_positive_probability() -> None:
    rng = make_rng(cfg.SEED)
    alpha1 = cfg.EXP5_ALPHA1_CASES["strict_local"]
    exited = simulate_shifted_survival(
        alpha1, cfg.EXP5_KAPPA1, cfg.EXP5_ETA1,
        cfg.EXP5_ALPHA2, cfg.EXP5_KAPPA2, cfg.EXP5_ETA2,
        cfg.EXP5_RHO1, cfg.EXP5_R12, cfg.EXP5_U, cfg.EXP5_V, g,
        cfg.EXP5_Y0, T=2.0, n_steps=100, n_paths=3000, rng=rng, eps=cfg.EXP5_EPS,
    )
    assert exited.mean() > 0.0
