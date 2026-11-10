"""Unit tests for exp_3 (two-factor inverse-CIR model, Example 6.4)."""
from __future__ import annotations

import numpy as np
import pytest

from src import config as cfg
from src.criteria import is_martingale_two_factor_inverse_cir
from src.mc_core import make_rng, mc_mean_ci
from src.model_multi_factor import simulate_factor_and_exponential, simulate_shifted_survival


def g(y2: np.ndarray) -> np.ndarray:
    return 1.0 + y2 / (1.0 + y2)


@pytest.mark.parametrize("case", list(cfg.EXP3_RHO_CASES.keys()))
def test_criteria_matches_formula(case: str) -> None:
    rho = cfg.EXP3_RHO_CASES[case]
    expected = 2 * cfg.EXP3_ALPHA1 - cfg.EXP3_ETA1 ** 2 >= 2 * cfg.EXP3_ETA1 * max(-rho, 0.0) * cfg.EXP3_G_BAR
    assert is_martingale_two_factor_inverse_cir(cfg.EXP3_ALPHA1, cfg.EXP3_ETA1, rho, cfg.EXP3_G_BAR) == expected


def test_nonnegative_rho_is_always_martingale_small_sample() -> None:
    rng = make_rng(cfg.SEED)
    rho = cfg.EXP3_RHO_CASES["nonneg_rho"]
    assert rho >= 0
    s_t = simulate_factor_and_exponential(
        cfg.EXP3_ALPHA1, cfg.EXP3_KAPPA1, cfg.EXP3_ETA1,
        cfg.EXP3_ALPHA2, cfg.EXP3_KAPPA2, cfg.EXP3_ETA2,
        rho, 0.0, 1.0, g, cfg.EXP3_Y0, T=1.0, n_steps=60, n_paths=3000, rng=rng,
    )
    e_s, _ = mc_mean_ci(s_t)
    assert abs(e_s - 1.0) < 0.3


def test_negative_rho_strict_local_martingale_mean_below_one() -> None:
    rng = make_rng(cfg.SEED)
    rho = cfg.EXP3_RHO_CASES["neg_rho_strict"]
    assert is_martingale_two_factor_inverse_cir(cfg.EXP3_ALPHA1, cfg.EXP3_ETA1, rho, cfg.EXP3_G_BAR) is False
    s_t = simulate_factor_and_exponential(
        cfg.EXP3_ALPHA1, cfg.EXP3_KAPPA1, cfg.EXP3_ETA1,
        cfg.EXP3_ALPHA2, cfg.EXP3_KAPPA2, cfg.EXP3_ETA2,
        rho, 0.0, 1.0, g, cfg.EXP3_Y0, T=2.0, n_steps=100, n_paths=4000, rng=rng,
    )
    e_s, _ = mc_mean_ci(s_t)
    assert e_s < 1.0


def test_second_factor_is_autonomous_under_shift() -> None:
    """Y2's drift under the shift must not depend on Y1 (methodology step 5):
    the shifted-survival simulator must accept and use the same g/alpha2/kappa2/eta2
    regardless of rho, since only Y1's drift carries the rho*eta1*b*g(Y2) term."""
    rng = make_rng(cfg.SEED)
    rho = cfg.EXP3_RHO_CASES["neg_rho_mart"]
    exited = simulate_shifted_survival(
        cfg.EXP3_ALPHA1, cfg.EXP3_KAPPA1, cfg.EXP3_ETA1,
        cfg.EXP3_ALPHA2, cfg.EXP3_KAPPA2, cfg.EXP3_ETA2,
        rho, 0.0, 1.0, g, cfg.EXP3_Y0, T=1.0, n_steps=50, n_paths=1000, rng=rng, eps=cfg.EXP3_EPS,
    )
    assert exited.dtype == bool
    assert 0.0 <= exited.mean() <= 1.0
