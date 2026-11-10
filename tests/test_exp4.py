"""Unit tests for exp_4 (4/2 model and multi-factor 4/2 criterion, Example 6.5)."""
from __future__ import annotations

from src import config as cfg
from src.criteria import is_martingale_4_2, is_martingale_multi_factor_4_2
from src.mc_core import make_rng, mc_mean_ci
from src.model_one_factor_vol import simulate_factor_and_exponential as sim_one_factor
from src.model_multi_factor import simulate_factor_and_exponential as sim_multi_factor
from exp.exp3 import g_func


def test_heston_weight_a_does_not_change_martingale_verdict() -> None:
    for a in cfg.EXP4_A_GRID:
        assert is_martingale_4_2(cfg.EXP4_ALPHA, cfg.EXP4_ETA, cfg.EXP4_RHO, cfg.EXP4_B_FIXED) is False


def test_varying_a_leaves_martingale_verdict_unchanged_small_sample() -> None:
    """Corollary 3.1: a bounded perturbation of the integrand (the Heston
    weight a) leaves the Fichera drift, hence the martingale/strict-local-
    martingale verdict, unchanged. At cfg.EXP4_ALPHA/EXP4_B_FIXED the
    threshold alpha+rho*eta*b=eta^2/2 fails for every a (the condition does
    not involve a), so every a must stay in the strict local martingale
    regime: the analytic verdict is identical, and the simulated E[S_T]
    stays strictly below 1 for every a (the exact deficit is not invariant,
    since kappa_tilde = kappa - rho*eta*a does depend on a)."""
    verdicts = [is_martingale_4_2(cfg.EXP4_ALPHA, cfg.EXP4_ETA, cfg.EXP4_RHO, cfg.EXP4_B_FIXED)
                for _ in (0.0, 1.0, 2.0)]
    assert len(set(verdicts)) == 1 and verdicts[0] is False
    for a in (0.0, 1.0, 2.0):
        rng = make_rng(cfg.SEED)
        s_t = sim_one_factor(cfg.EXP4_ALPHA, cfg.EXP4_KAPPA, cfg.EXP4_ETA, cfg.EXP4_RHO,
                              a, cfg.EXP4_B_FIXED, cfg.EXP4_V0, T=1.0, n_steps=200, n_paths=4000, rng=rng)
        e_s, _ = mc_mean_ci(s_t)
        assert e_s < 0.95  # comfortably below 1, well outside Monte Carlo noise at this sample size


def test_varying_a_leaves_e_s_t_at_one_in_martingale_regime() -> None:
    """When alpha+rho*eta*b >= eta^2/2 holds, S is a true martingale for
    every a (Corollary 3.1), so E[S_T] = 1 exactly regardless of a."""
    rho, b = 0.5, 0.6  # alpha + rho*eta*b = 0.4+0.3=0.7 >= eta^2/2=0.5
    assert is_martingale_4_2(cfg.EXP4_ALPHA, cfg.EXP4_ETA, rho, b) is True
    for a in (0.0, 1.0, 2.0):
        rng = make_rng(cfg.SEED)
        s_t = sim_one_factor(cfg.EXP4_ALPHA, cfg.EXP4_KAPPA, cfg.EXP4_ETA, rho,
                              a, b, cfg.EXP4_V0, T=1.0, n_steps=200, n_paths=4000, rng=rng)
        e_s, ci = mc_mean_ci(s_t)
        assert abs(e_s - 1.0) < 4 * ci + 0.05


def test_b_threshold_crosses_at_eta_sq_half_minus_alpha_over_rho_eta() -> None:
    # Example 6.5 (one-factor 4/2): alpha + rho*eta*b >= eta^2/2. With rho*eta < 0,
    # the martingale region is b <= threshold_b (dividing by a negative number flips
    # the inequality), so b below the threshold is the martingale side.
    threshold_b = (cfg.EXP4_ETA ** 2 / 2 - cfg.EXP4_ALPHA_VARY_B) / (cfg.EXP4_RHO * cfg.EXP4_ETA)
    below = is_martingale_4_2(cfg.EXP4_ALPHA_VARY_B, cfg.EXP4_ETA, cfg.EXP4_RHO, threshold_b - 0.1)
    above = is_martingale_4_2(cfg.EXP4_ALPHA_VARY_B, cfg.EXP4_ETA, cfg.EXP4_RHO, threshold_b + 0.1)
    assert below is True
    assert above is False


def test_multi_factor_matches_two_factor_inverse_cir_criterion() -> None:
    for case, rho in cfg.EXP4_MULTI_RHO_CASES.items():
        predicted = is_martingale_multi_factor_4_2(cfg.EXP3_ALPHA1, cfg.EXP3_ETA1, rho,
                                                    cfg.EXP4_MULTI_B, cfg.EXP3_G_BAR)
        rng = make_rng(cfg.SEED)
        s_t = sim_multi_factor(cfg.EXP3_ALPHA1, cfg.EXP3_KAPPA1, cfg.EXP3_ETA1,
                                cfg.EXP3_ALPHA2, cfg.EXP3_KAPPA2, cfg.EXP3_ETA2,
                                rho, 0.0, cfg.EXP4_MULTI_B, g_func, cfg.EXP3_Y0,
                                T=1.0, n_steps=60, n_paths=3000, rng=rng)
        e_s, _ = mc_mean_ci(s_t)
        if predicted:
            assert abs(e_s - 1.0) < 0.3
        else:
            assert e_s < 1.1
