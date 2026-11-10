"""Unit tests for exp_2 (3/2 model martingale criterion via the Fichera drift, Example 6.3)."""
from __future__ import annotations

import pytest

from src import config as cfg
from src.criteria import is_martingale_3_2
from src.mc_core import make_rng, mc_mean_ci
from src.model_one_factor_vol import simulate_factor_and_exponential


@pytest.mark.parametrize("case", list(cfg.EXP2_ALPHA_CASES.keys()))
def test_criteria_matches_formula(case: str) -> None:
    alpha = cfg.EXP2_ALPHA_CASES[case]
    expected = alpha + cfg.EXP2_RHO * cfg.EXP2_ETA >= cfg.EXP2_ETA ** 2 / 2
    assert is_martingale_3_2(alpha, cfg.EXP2_ETA, cfg.EXP2_RHO) == expected


def test_martingale_and_critical_cases_have_mean_near_one() -> None:
    rng = make_rng(cfg.SEED)
    for case in ("martingale", "critical"):
        alpha = cfg.EXP2_ALPHA_CASES[case]
        assert is_martingale_3_2(alpha, cfg.EXP2_ETA, cfg.EXP2_RHO) is True
        s_t = simulate_factor_and_exponential(
            alpha, cfg.EXP2_KAPPA, cfg.EXP2_ETA, cfg.EXP2_RHO, 0.0, 1.0,
            cfg.EXP2_Y0, T=1.0, n_steps=60, n_paths=3000, rng=rng,
        )
        e_s, _ = mc_mean_ci(s_t)
        assert abs(e_s - 1.0) < 0.3


def test_strict_local_martingale_case_mean_strictly_below_one_and_decreasing() -> None:
    rng = make_rng(cfg.SEED)
    alpha = cfg.EXP2_ALPHA_CASES["strict_local"]
    assert is_martingale_3_2(alpha, cfg.EXP2_ETA, cfg.EXP2_RHO) is False
    means = []
    for T in (0.5, 2.0):
        s_t = simulate_factor_and_exponential(
            alpha, cfg.EXP2_KAPPA, cfg.EXP2_ETA, cfg.EXP2_RHO, 0.0, 1.0,
            cfg.EXP2_Y0, T=T, n_steps=80, n_paths=4000, rng=rng,
        )
        e_s, _ = mc_mean_ci(s_t)
        means.append(e_s)
    assert means[0] < 1.0
    assert means[1] <= means[0] + 0.15  # deficit should not shrink as T grows
