"""Tests for exp_11: PDE uniqueness / loss-of-mass (Appendix B, Prop. B.1).

Uses small grids (21 y-points, 20 time steps, T=0.5) so the whole module
runs in well under a minute.
"""
from __future__ import annotations

import numpy as np

from src import config as cfg
from src.pde_solver import build_shifted_generator_matrix, monte_carlo_loss_of_mass, solve_loss_of_mass

Y_GRID = np.linspace(cfg.EXP11_Y_GRID_RUN[0], cfg.EXP11_Y_GRID_RUN[1], 21)
N_TIME_STEPS = 20
T = 0.5


def test_martingale_case_has_negligible_loss_of_mass() -> None:
    m = solve_loss_of_mass(
        Y_GRID, cfg.EXP11_ALPHA_MARTINGALE, cfg.EXP11_KAPPA, cfg.EXP11_ETA, cfg.EXP11_RHO, T, N_TIME_STEPS,
    )
    assert np.max(np.abs(m[-1])) < 0.15


def test_strict_local_case_has_positive_loss_of_mass() -> None:
    m = solve_loss_of_mass(
        Y_GRID, cfg.EXP11_ALPHA_STRICT, cfg.EXP11_KAPPA, cfg.EXP11_ETA, cfg.EXP11_RHO, T, N_TIME_STEPS,
    )
    assert np.max(m[-1]) > 0.02


def test_initial_condition_is_zero() -> None:
    m = solve_loss_of_mass(
        Y_GRID, cfg.EXP11_ALPHA_STRICT, cfg.EXP11_KAPPA, cfg.EXP11_ETA, cfg.EXP11_RHO, T, N_TIME_STEPS,
    )
    assert np.all(m[0] == 0.0)


def test_monte_carlo_loss_of_mass_returns_finite_values() -> None:
    rng = np.random.default_rng(0)
    result = monte_carlo_loss_of_mass(
        1.0, cfg.EXP11_ALPHA_STRICT, cfg.EXP11_KAPPA, cfg.EXP11_ETA, cfg.EXP11_RHO,
        [T], n_paths=500, n_steps=200, rng=rng,
    )
    assert set(result.keys()) == {T}
    mean, ci = result[T]
    assert np.isfinite(mean)
    assert np.isfinite(ci)


def test_generator_matrix_shape() -> None:
    mat = build_shifted_generator_matrix(Y_GRID, cfg.EXP11_ALPHA_MARTINGALE, cfg.EXP11_KAPPA, cfg.EXP11_ETA, cfg.EXP11_RHO)
    assert mat.shape == (len(Y_GRID), len(Y_GRID))
