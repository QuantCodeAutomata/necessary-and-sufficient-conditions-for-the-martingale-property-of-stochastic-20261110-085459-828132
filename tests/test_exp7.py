"""Fast correctness tests for exp_7 (isotropic power-volatility model,
Example 6.1). Uses tiny Monte Carlo sizes so the whole module runs in a few
seconds; it is not meant to be a statistically tight check.
"""
from __future__ import annotations

import math

from src import config as cfg
from src.criteria import is_martingale_power_volatility
from src.mc_core import make_rng, mc_mean_ci
from src.model_isotropic_ou import (
    feller_test_v_infinity,
    radial_scale_functions,
    simulate_path_and_exponential,
)

N_PATHS = 300
N_STEPS = 100
T = 0.5


def test_criterion_matches_config_cases() -> None:
    assert is_martingale_power_volatility(*cfg.EXP7_CASES["neg_rho"]) is True
    assert is_martingale_power_volatility(*cfg.EXP7_CASES["p_le_1"]) is True
    assert is_martingale_power_volatility(*cfg.EXP7_CASES["strict_local"]) is False
    assert is_martingale_power_volatility(*cfg.EXP7_CASES["boundary_p1"]) is True


def test_martingale_case_mean_near_one() -> None:
    rng = make_rng(cfg.SEED)
    for case in ("p_le_1", "neg_rho"):
        rho, p = cfg.EXP7_CASES[case]
        s_t = simulate_path_and_exponential(
            cfg.EXP7_KAPPA, cfg.EXP7_ETA, rho, p, cfg.EXP7_Y0, T, N_STEPS, N_PATHS, rng,
        )
        mean, _ = mc_mean_ci(s_t, cfg.CONFIDENCE)
        assert abs(mean - 1.0) < 0.6


def test_radial_scale_functions_well_defined() -> None:
    rho, p = cfg.EXP7_CASES["strict_local"]
    A, B = radial_scale_functions(cfg.EXP7_D, cfg.EXP7_KAPPA, cfg.EXP7_ETA, rho, p)
    r = 1.0
    assert A(r) > 0.0
    assert math.isfinite(B(r))


def test_feller_test_v_infinity_runs_and_has_expected_keys() -> None:
    rho, p = cfg.EXP7_CASES["strict_local"]
    r_values = (10.0, 1e2, 1e3)
    result = feller_test_v_infinity(cfg.EXP7_D, cfg.EXP7_KAPPA, cfg.EXP7_ETA, rho, p, r_values=r_values)
    for R in r_values:
        assert R in result
    assert "diverging" in result
    assert isinstance(result["diverging"], bool)
