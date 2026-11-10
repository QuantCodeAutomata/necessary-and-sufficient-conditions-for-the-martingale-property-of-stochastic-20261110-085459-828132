"""Analytic martingale criteria derived from the Fichera-drift / boundary
analysis of Bayraktar & Xing. Pure functions of the model coefficients, used
to classify each parameter case before (and as a check on) the Monte Carlo
verification carried out in the exp/ scripts.

# Custom implementation (paper Sec. 3.1, Eq. (3.2): Fichera drift
# F_b(x) = <b(x) - 1/2 div a(x), nu(x)>, and Sec. 6, Examples 6.3-6.6).
"""
from __future__ import annotations


def fichera_drift_3_2(alpha: float, eta: float, rho: float) -> float:
    """Fichera drift at y=0 for the 3/2 model (Example 6.3): F(0) = -(alpha+rho*eta-eta^2/2)."""
    return -(alpha + rho * eta - eta ** 2 / 2)


def is_martingale_3_2(alpha: float, eta: float, rho: float) -> bool:
    """3/2 model: S is a true martingale iff alpha + rho*eta >= eta^2/2 (Example 6.3)."""
    return alpha + rho * eta >= eta ** 2 / 2


def is_martingale_4_2(alpha: float, eta: float, rho: float, b: float) -> bool:
    """One-factor 4/2 model: S is a true martingale iff alpha + rho*eta*b >= eta^2/2 (Example 6.5).

    Independent of the Heston weight a, consistent with Corollary 3.1: a
    bounded integrand (the a*sqrt(v) term) does not change the Fichera drift.
    """
    return alpha + rho * eta * b >= eta ** 2 / 2


def is_martingale_two_factor_inverse_cir(alpha1: float, eta1: float, rho: float, g_bar: float) -> bool:
    """Two-factor inverse-CIR model (Example 6.4):
    martingale iff 2*alpha1 - eta1^2 >= 2*eta1*rho^- *sup(g), rho^- = max(-rho, 0).
    """
    rho_minus = max(-rho, 0.0)
    return 2 * alpha1 - eta1 ** 2 >= 2 * eta1 * rho_minus * g_bar


def is_martingale_multi_factor_4_2(alpha1: float, eta1: float, rho: float, b: float, g_bar: float) -> bool:
    """Multi-factor 4/2 model (Example 6.5): martingale iff 2*alpha1-eta1^2 >= 2*eta1*(rho*b)^- *sup(g)."""
    rho_b_minus = max(-rho * b, 0.0)
    return 2 * alpha1 - eta1 ** 2 >= 2 * eta1 * rho_b_minus * g_bar


def is_martingale_correlated_sqrt(alpha1: float, eta1: float, rho1: float, v: float, g_bar: float) -> bool:
    """Correlated square-root factors (Theorem 6.2): martingale iff
    2*alpha1-eta1^2 >= 2*eta1*(rho1*v)^- *sup(g); independent of R_ij, rho_i (i>=2).
    """
    rho1_v_minus = max(-rho1 * v, 0.0)
    return 2 * alpha1 - eta1 ** 2 >= 2 * eta1 * rho1_v_minus * g_bar


def is_martingale_coupled_log(sup_c: float) -> bool:
    """Coupled logarithmic threshold (Theorem 3.2): martingale on every finite horizon iff sup c <= 1/2."""
    return sup_c <= 0.5


def is_uniformly_integrable_coupled_log(sup_c: float) -> bool:
    """Corollary 3.2: S is uniformly integrable on [0, infinity) iff c == 0 (sup_c == 0 here)."""
    return sup_c == 0.0


def is_martingale_power_volatility(rho: float, p: float) -> bool:
    """Isotropic multi-factor power-volatility model (Example 6.1): martingale iff rho <= 0 or p <= 1."""
    return rho <= 0.0 or p <= 1.0


def is_martingale_weighted_distance(sup_c: float, m: int) -> bool:
    """Weighted-distance / inverse-distance singularity model (Proposition 5.2):
    martingale iff sup c <= (m-2)/2.
    """
    return sup_c <= (m - 2) / 2


def is_martingale_absorbed(inf_c: float) -> bool:
    """Absorbed factor with correlated modulation (Theorem 6.3): martingale iff inf c >= 1/2."""
    return inf_c >= 0.5


def wishart_fichera_drift(omega_sq: float, q_sq: float, n: int) -> float:
    """Fichera drift at a rank-(n-1) Wishart boundary point (Proposition 7.2):
    F = -(omega^2 - (n+1) q^2) for isotropic Omega = omega*I, Q = sqrt(q^2)*I.
    """
    return -(omega_sq - (n + 1) * q_sq)


def is_wishart_boundary_inaccessible(omega_sq: float, q_sq: float, n: int) -> bool:
    """Wishart boundary condition Omega Omega^T >= (n+1) Q^T Q (Proposition 7.2), isotropic case."""
    return omega_sq >= (n + 1) * q_sq


def naive_cir_threshold_is_wrong(omega_sq: float, q_sq: float, n: int) -> bool:
    """True when the naive scalar-CIR threshold (factor 2, ignoring curvature) and the
    correct Wishart threshold (factor n+1) disagree, i.e. 2*q_sq <= omega_sq < (n+1)*q_sq.
    """
    return 2 * q_sq <= omega_sq < (n + 1) * q_sq
