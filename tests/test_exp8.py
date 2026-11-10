"""Fast correctness tests for exp_8 (weighted-distance criterion, Prop. 5.2
and Example 5.2). Uses tiny Monte Carlo sizes so the whole module runs in a
few seconds; it is not meant to be a statistically tight check.
"""
from __future__ import annotations

import math

import numpy as np

from src import config as cfg
from src.criteria import is_martingale_weighted_distance
from src.mc_core import make_rng, mc_mean_ci
from src.model_weighted_distance import (
    euclidean_projection_integral,
    simulate_exponential,
    simulate_shifted_survival,
)

N_PATHS = 300
N_STEPS = 100
T = 0.5


def test_criterion_matches_config_cases() -> None:
    assert is_martingale_weighted_distance(cfg.EXP8_C_CASES["martingale"], cfg.EXP8_M) is True
    assert is_martingale_weighted_distance(cfg.EXP8_C_CASES["critical"], cfg.EXP8_M) is True
    assert is_martingale_weighted_distance(cfg.EXP8_C_CASES["strict_local"], cfg.EXP8_M) is False

    sup_c_example52 = cfg.EXP8_EXAMPLE52_A + cfg.EXP8_EXAMPLE52_B
    assert math.isclose(sup_c_example52, 0.4)
    assert is_martingale_weighted_distance(sup_c_example52, cfg.EXP8_M) is True


def test_martingale_case_mean_near_one() -> None:
    rng = make_rng(cfg.SEED)
    c = cfg.EXP8_C_CASES["martingale"]
    c_fn = lambda z: np.full_like(z, c)
    s_t = simulate_exponential(
        cfg.EXP8_M, cfg.EXP8_K, cfg.EXP8_LAMBDA, cfg.EXP8_H, cfg.EXP8_U0, cfg.EXP8_Z0,
        c_fn, T, N_STEPS, N_PATHS, rng,
    )
    mean, _ = mc_mean_ci(s_t, cfg.CONFIDENCE)
    assert abs(mean - 1.0) < 0.5


def test_euclidean_projection_integral_closed_form() -> None:
    sup_c = 0.4
    j = euclidean_projection_integral(sup_c)
    assert math.isfinite(j)
    assert j > 0.0

    beta = 0.5 - 2 * sup_c
    r0 = 0.1
    expected = r0 / (1 - beta)
    assert math.isclose(j, expected, rel_tol=1e-12)
    assert math.isclose(beta, -0.3)


def test_simulate_shifted_survival_shape_and_dtype() -> None:
    rng = make_rng(cfg.SEED)
    c = cfg.EXP8_C_CASES["martingale"]
    c_fn = lambda z: np.full_like(z, c)
    exited = simulate_shifted_survival(
        cfg.EXP8_M, cfg.EXP8_K, cfg.EXP8_LAMBDA, cfg.EXP8_H, cfg.EXP8_U0, cfg.EXP8_Z0,
        c_fn, T, N_STEPS, N_PATHS, rng,
    )
    assert exited.shape == (N_PATHS,)
    assert exited.dtype == bool
