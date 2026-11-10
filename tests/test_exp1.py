"""Unit tests for exp_1 (loss-of-mass identity E_y[S_T] = Qtilde_y(tau_D > T), Lemma 2.1)."""
from __future__ import annotations

from src import config as cfg
from src.mc_core import make_rng, mc_mean_ci
from src.model_one_factor_vol import simulate_factor_and_exponential, simulate_shifted_survival


def _small_sample(alpha: float, T: float, n_paths: int = 400, n_steps: int = 20):
    rng = make_rng(cfg.SEED)
    s_t = simulate_factor_and_exponential(
        alpha, cfg.EXP1_KAPPA, cfg.EXP1_ETA, cfg.EXP1_RHO, 0.0, 1.0,
        cfg.EXP1_Y0, T, n_steps, n_paths, rng,
    )
    exited = simulate_shifted_survival(
        alpha, cfg.EXP1_KAPPA, cfg.EXP1_ETA, cfg.EXP1_RHO, 0.0, 1.0,
        cfg.EXP1_Y0, T, n_steps, n_paths, rng, cfg.EXP1_EPS,
    )
    return s_t, exited


def test_s_t_is_nonnegative_and_finite() -> None:
    s_t, _ = _small_sample(cfg.EXP1_ALPHA_MARTINGALE, T=0.5)
    assert (s_t >= 0.0).all()
    assert (s_t < 100.0).all()


def test_martingale_case_mean_close_to_one() -> None:
    assert cfg.EXP1_ALPHA_MARTINGALE + cfg.EXP1_RHO * cfg.EXP1_ETA >= cfg.EXP1_ETA ** 2 / 2
    s_t, exited = _small_sample(cfg.EXP1_ALPHA_MARTINGALE, T=0.5, n_paths=3000, n_steps=60)
    e_s, _ = mc_mean_ci(s_t)
    assert abs(e_s - 1.0) < 0.3
    assert exited.mean() < 0.3


def test_strict_local_martingale_case_mean_below_one() -> None:
    assert cfg.EXP1_ALPHA_STRICT + cfg.EXP1_RHO * cfg.EXP1_ETA < cfg.EXP1_ETA ** 2 / 2
    s_t, exited = _small_sample(cfg.EXP1_ALPHA_STRICT, T=2.0, n_paths=3000, n_steps=120)
    e_s, _ = mc_mean_ci(s_t)
    # Strict local martingale: expected value should be bounded away from one.
    assert e_s < 1.0
    assert exited.mean() > 0.0


def test_loss_of_mass_matches_exit_probability_order_of_magnitude() -> None:
    # Lemma 2.1: E_y[S_T] = Qtilde_y(tau_D > T), so 1 - E[S_T] ~ exit probability.
    s_t, exited = _small_sample(cfg.EXP1_ALPHA_STRICT, T=2.0, n_paths=4000, n_steps=120)
    e_s, half = mc_mean_ci(s_t)
    survival = 1.0 - exited.mean()
    # Loose tolerance appropriate for a small-sample regression test (large MC noise).
    assert abs(e_s - survival) < 0.25
