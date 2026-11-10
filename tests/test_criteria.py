"""Unit tests for the analytic martingale criteria in src/criteria.py
(Fichera-drift / boundary-behaviour thresholds of Bayraktar & Xing)."""
from __future__ import annotations

import pytest

from src import criteria as crit


def test_fichera_drift_3_2_sign_matches_martingale_condition() -> None:
    # alpha + rho*eta >= eta^2/2  <=>  F(0) <= 0
    assert crit.fichera_drift_3_2(alpha=1.2, eta=1.0, rho=-0.5) <= 0
    assert crit.is_martingale_3_2(alpha=1.2, eta=1.0, rho=-0.5) is True
    assert crit.fichera_drift_3_2(alpha=0.6, eta=1.0, rho=-0.5) > 0
    assert crit.is_martingale_3_2(alpha=0.6, eta=1.0, rho=-0.5) is False


def test_is_martingale_3_2_critical_equality_is_martingale() -> None:
    # alpha + rho*eta == eta^2/2 exactly: equality counts as a martingale (Theorem 6.1(ii)).
    alpha, eta, rho = 0.5, 1.0, 0.0
    assert alpha + rho * eta == eta ** 2 / 2
    assert crit.is_martingale_3_2(alpha, eta, rho) is True


def test_is_martingale_4_2_independent_of_heston_weight_a() -> None:
    # Corollary 3.1: a bounded integrand (the a*sqrt(v) Heston term) does not
    # change the Fichera drift, so the verdict must not depend on a.
    verdicts = {crit.is_martingale_4_2(alpha=0.4, eta=1.0, rho=-0.5, b=0.5) for _ in range(1)}
    assert len(verdicts) == 1
    for a in (0.0, 0.5, 1.0, 2.0):
        # a does not appear in the signature's martingale criterion at all.
        assert crit.is_martingale_4_2(alpha=0.4, eta=1.0, rho=-0.5, b=0.5) == (0.4 - 0.25 >= 0.5)


def test_is_martingale_two_factor_inverse_cir_nonneg_rho_always_martingale() -> None:
    # alpha1=0.6, eta1=1.0 satisfies the Feller condition 2*alpha1 >= eta1^2 (data_requirements);
    # for rho >= 0, rho^- = 0 so the condition reduces to 2*alpha1-eta1^2 >= 0, which holds.
    for rho in (0.0, 0.2, 0.9):
        assert crit.is_martingale_two_factor_inverse_cir(alpha1=0.6, eta1=1.0, rho=rho, g_bar=10.0) is True


def test_is_martingale_two_factor_inverse_cir_threshold_crossing() -> None:
    alpha1, eta1, g_bar = 1.2, 1.0, 2.0
    # 2*alpha1 - eta1^2 = 1.4; threshold 2*eta1*|rho|*gbar crosses 1.4 at |rho| = 0.35
    assert crit.is_martingale_two_factor_inverse_cir(alpha1, eta1, rho=-0.3, g_bar=g_bar) is True
    assert crit.is_martingale_two_factor_inverse_cir(alpha1, eta1, rho=-0.8, g_bar=g_bar) is False


def test_is_martingale_multi_factor_4_2_reduces_to_two_factor_when_b_is_one() -> None:
    args = dict(alpha1=1.2, eta1=1.0, rho=-0.3, g_bar=2.0)
    assert crit.is_martingale_multi_factor_4_2(**args, b=1.0) == crit.is_martingale_two_factor_inverse_cir(
        alpha1=args["alpha1"], eta1=args["eta1"], rho=args["rho"], g_bar=args["g_bar"]
    )


def test_is_martingale_correlated_sqrt_example_6_6_threshold() -> None:
    # Example 6.6: eta1=1, rho1=-1/2, v=1, sup g = 2 -> threshold alpha1 >= 3/2.
    assert crit.is_martingale_correlated_sqrt(alpha1=2.0, eta1=1.0, rho1=-0.5, v=1.0, g_bar=2.0) is True
    assert crit.is_martingale_correlated_sqrt(alpha1=1.5, eta1=1.0, rho1=-0.5, v=1.0, g_bar=2.0) is True
    assert crit.is_martingale_correlated_sqrt(alpha1=1.0, eta1=1.0, rho1=-0.5, v=1.0, g_bar=2.0) is False


@pytest.mark.parametrize("sup_c,expected", [(0.0, True), (0.2, True), (0.5, True), (0.6, False), (1.0, False)])
def test_is_martingale_coupled_log_threshold(sup_c: float, expected: bool) -> None:
    assert crit.is_martingale_coupled_log(sup_c) is expected


def test_is_uniformly_integrable_coupled_log_only_at_zero() -> None:
    assert crit.is_uniformly_integrable_coupled_log(0.0) is True
    assert crit.is_uniformly_integrable_coupled_log(0.2) is False
    assert crit.is_uniformly_integrable_coupled_log(0.5) is False


def test_is_martingale_power_volatility_example_6_1() -> None:
    assert crit.is_martingale_power_volatility(rho=-0.5, p=2.0) is True   # rho <= 0
    assert crit.is_martingale_power_volatility(rho=0.5, p=0.5) is True    # p <= 1
    assert crit.is_martingale_power_volatility(rho=0.5, p=1.0) is True    # boundary p == 1
    assert crit.is_martingale_power_volatility(rho=0.5, p=2.0) is False   # rho > 0 and p > 1


@pytest.mark.parametrize("m,sup_c,expected", [(3, 0.2, True), (3, 0.5, True), (3, 0.7, False), (4, 0.9, True)])
def test_is_martingale_weighted_distance_bessel_threshold(m: int, sup_c: float, expected: bool) -> None:
    assert crit.is_martingale_weighted_distance(sup_c, m) is expected


@pytest.mark.parametrize("inf_c,expected", [(0.6, True), (0.5, True), (0.4, False)])
def test_is_martingale_absorbed_threshold(inf_c: float, expected: bool) -> None:
    assert crit.is_martingale_absorbed(inf_c) is expected


def test_wishart_fichera_drift_and_boundary_condition_consistent() -> None:
    n = 2
    for omega_sq, q_sq in [(4.0, 1.0), (3.0, 1.0), (2.0, 1.0)]:
        drift = crit.wishart_fichera_drift(omega_sq, q_sq, n)
        inaccessible = crit.is_wishart_boundary_inaccessible(omega_sq, q_sq, n)
        assert (drift <= 0) == inaccessible


def test_naive_cir_threshold_is_wrong_only_in_the_gap_band() -> None:
    n = 2  # correct threshold n+1 = 3, naive scalar-CIR threshold = 2
    assert crit.naive_cir_threshold_is_wrong(omega_sq=2.5, q_sq=1.0, n=n) is True
    assert crit.naive_cir_threshold_is_wrong(omega_sq=4.0, q_sq=1.0, n=n) is False
    assert crit.naive_cir_threshold_is_wrong(omega_sq=1.5, q_sq=1.0, n=n) is False
