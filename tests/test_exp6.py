"""Unit tests for exp_6 (coupled logarithmic threshold, Theorem 3.2 / Corollary 3.2)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from src import config as cfg
from src.criteria import is_martingale_coupled_log, is_uniformly_integrable_coupled_log
from src.mc_core import make_rng
from src.model_logarithmic import h_function, simulate_coupled_log_exponential


def test_h_function_endpoints_and_intermediate() -> None:
    low = math.exp(-2.0)
    high = math.exp(-1.0)

    x_small = np.array([1e-10, 0.01, low])
    expected_small = 1.0 / np.log(1.0 / x_small)
    assert np.allclose(h_function(x_small), expected_small)

    x_large = np.array([high, 0.5, 1.0])
    assert np.allclose(h_function(x_large), 0.0)

    mid = (low + high) / 2.0
    h_mid = h_function(np.array([mid]))[0]
    assert 0.0 < h_mid < h_function(np.array([low]))[0]


def test_martingale_case_sample_mean_near_one() -> None:
    rng = make_rng(cfg.SEED)
    a, b = cfg.EXP6_CASES["martingale"]
    s_t = simulate_coupled_log_exponential(
        a, b, cfg.EXP6_KAPPA, cfg.EXP6_LAMBDA, cfg.EXP6_ETA_Z, cfg.EXP6_RHO1,
        cfg.EXP6_X0, cfg.EXP6_Z0, cfg.EXP6_X_FLOOR, T=0.5, n_steps=15, n_paths=300,
        rng=rng, chunk_size=20_000,
    )
    assert abs(s_t.mean() - 1.0) < 0.5


def test_trivial_case_is_exactly_one() -> None:
    rng = make_rng(cfg.SEED)
    a, b = cfg.EXP6_CASES["trivial"]
    assert a == 0.0 and b == 0.0
    s_t = simulate_coupled_log_exponential(
        a, b, cfg.EXP6_KAPPA, cfg.EXP6_LAMBDA, cfg.EXP6_ETA_Z, cfg.EXP6_RHO1,
        cfg.EXP6_X0, cfg.EXP6_Z0, cfg.EXP6_X_FLOOR, T=0.5, n_steps=15, n_paths=300,
        rng=rng, chunk_size=20_000,
    )
    assert np.allclose(s_t, 1.0, atol=1e-6)


@pytest.mark.parametrize("case", list(cfg.EXP6_CASES.keys()))
def test_criteria_match_expected_booleans(case: str) -> None:
    a, b = cfg.EXP6_CASES[case]
    sup_c = a + b
    assert is_martingale_coupled_log(sup_c) == (sup_c <= 0.5)
    assert is_uniformly_integrable_coupled_log(sup_c) == (sup_c == 0.0)
