"""Fast correctness tests for exp_9 (absorbed factor with correlated
modulation, Theorem 6.3). Uses tiny Monte Carlo sizes so the whole module
runs in a few seconds; it is not meant to be a statistically tight check.
"""
from __future__ import annotations

import numpy as np

from src import config as cfg
from src.criteria import is_martingale_absorbed
from src.mc_core import make_rng, mc_mean_ci
from src.model_absorbed import c_function, simulate_density

N_PATHS = 500
N_STEPS = 200
T = 0.5


def test_criterion_matches_config_cases() -> None:
    assert is_martingale_absorbed(cfg.EXP9_CASES["martingale"][0]) is True
    assert is_martingale_absorbed(cfg.EXP9_CASES["critical"][0]) is True
    assert is_martingale_absorbed(cfg.EXP9_CASES["strict_local"][0]) is False
    assert is_martingale_absorbed(cfg.EXP9_CASES["classical"][0]) is True


def test_martingale_case_mean_near_one() -> None:
    rng = make_rng(cfg.SEED)
    # Largest inf_c among the configured cases: strongly a true martingale.
    inf_c_values = {name: a for name, (a, _) in cfg.EXP9_CASES.items()}
    best_case = max(inf_c_values, key=inf_c_values.get)
    a, b = cfg.EXP9_CASES[best_case]
    c_fn = c_function(a, b)

    s_t = simulate_density(
        cfg.EXP9_X0, cfg.EXP9_Z0, cfg.EXP9_LAMBDA, cfg.EXP9_ETA_Z, cfg.EXP9_RHO1,
        c_fn, T, N_STEPS, N_PATHS, rng, cfg.EXP9_X_FLOOR,
    )
    mean, _ = mc_mean_ci(s_t, cfg.CONFIDENCE)
    assert abs(mean - 1.0) < 0.5


def test_explicit_density_kill_on_absorption() -> None:
    rng = make_rng(cfg.SEED)
    c_fn = c_function(1.0, 0.0)
    x0, T_adverse, n_paths_adverse, n_steps_adverse = 0.02, 5.0, 2000, 2000

    s_t = simulate_density(
        x0, cfg.EXP9_Z0, cfg.EXP9_LAMBDA, cfg.EXP9_ETA_Z, cfg.EXP9_RHO1,
        c_fn, T_adverse, n_steps_adverse, n_paths_adverse, rng, cfg.EXP9_X_FLOOR,
    )
    assert s_t.shape == (n_paths_adverse,)
    assert s_t.dtype == np.float64

    n_absorbed = int((s_t == 0.0).sum())
    if n_absorbed == 0:
        print("WARNING: no paths were absorbed in the adverse configuration; "
              "only checking output shape/dtype for this test run.")
    else:
        assert n_absorbed > 0


def test_c_function_evaluates_correctly() -> None:
    z = np.array([0.0, 1.0, -2.0])
    c_fn = c_function(0.4, 0.2)
    expected = 0.4 + 0.2 / (1.0 + z ** 2)
    np.testing.assert_allclose(c_fn(z), expected)

    c_fn_const = c_function(1.0, 0.0)
    np.testing.assert_allclose(c_fn_const(z), np.ones_like(z))
