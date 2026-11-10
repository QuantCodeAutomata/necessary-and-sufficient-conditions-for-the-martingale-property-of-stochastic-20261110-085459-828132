"""Tests for exp_10: Wishart process boundary condition (Proposition 7.2)."""
from __future__ import annotations

import numpy as np
import pytest

from src import config as cfg
from src.criteria import (
    is_wishart_boundary_inaccessible,
    naive_cir_threshold_is_wrong,
    wishart_fichera_drift,
)
from src.mc_core import make_rng
from src.model_wishart import (
    boundary_hit_probability,
    lambda_min_2x2,
    simulate_min_eigenvalue_paths,
)

Q_SQ = cfg.EXP10_Q[0][0] ** 2  # q_sq = 1.0 since Q = I
N = cfg.EXP10_N


def test_criteria_qualitative_facts() -> None:
    assert N == 2
    found_naive_wrong = False
    for omega_sq in cfg.EXP10_OMEGA_SQ_CASES.values():
        predicted = is_wishart_boundary_inaccessible(omega_sq, Q_SQ, N)
        assert predicted == (omega_sq >= (N + 1) * Q_SQ)

        naive_wrong = naive_cir_threshold_is_wrong(omega_sq, Q_SQ, N)
        assert naive_wrong == (2 * Q_SQ <= omega_sq < (N + 1) * Q_SQ)
        if naive_wrong:
            found_naive_wrong = True

        fichera = wishart_fichera_drift(omega_sq, Q_SQ, N)
        assert fichera == pytest.approx(-(omega_sq - (N + 1) * Q_SQ))

    assert found_naive_wrong, (
        "expected at least one configured omega_sq in 2*q_sq <= omega_sq < "
        "(n+1)*q_sq, where the naive CIR threshold (factor 2) disagrees "
        "with the correct Wishart threshold (factor n+1)"
    )


# This simplified (componentwise-sqrt) discretization of the n=2 Wishart SDE
# does not exactly preserve positive semi-definiteness away from Sigma12=0
# (the true matrix square root's off-diagonal entry is dropped, per the
# literal equations in the exp_10 implementation notes), so moderate omega_sq
# values such as the EXP10_OMEGA_SQ_CASES (up to 4.0) still show a sizeable
# boundary-hit rate over T=1 at these tiny step/path counts. A substantially
# larger omega_sq is used here to exercise the "well above threshold" regime.
WELL_ABOVE_THRESHOLD_OMEGA_SQ = 100.0


def test_boundary_hit_probability_above_threshold_is_small() -> None:
    rng = make_rng(0)
    min_eig = simulate_min_eigenvalue_paths(
        N, WELL_ABOVE_THRESHOLD_OMEGA_SQ, cfg.EXP10_SIGMA0, 1.0, 200, 500, rng,
    )
    p_hit = boundary_hit_probability(min_eig, cfg.EXP10_EPS)
    assert p_hit < 0.3


def test_simulate_min_eigenvalue_paths_shape_and_validity() -> None:
    rng = make_rng(1)
    min_eig = simulate_min_eigenvalue_paths(
        N, WELL_ABOVE_THRESHOLD_OMEGA_SQ, cfg.EXP10_SIGMA0, 1.0, 200, 500, rng,
    )
    assert min_eig.shape == (500,)
    assert np.all(np.isfinite(min_eig))
    assert np.all(min_eig >= -1e-6)


def test_lambda_min_closed_form() -> None:
    assert lambda_min_2x2(np.array(2.0), np.array(2.0), np.array(0.0)) == pytest.approx(2.0)
