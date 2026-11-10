"""Module-level configuration for every experiment.

All model coefficients (drifts, diffusion parameters, correlations) are taken
from the paper's examples and theorems (Bayraktar & Xing), cited section by
section below. The paper states these coefficients symbolically but does not
fix a Monte Carlo sample size or time-discretization step for any of its
numerical verifications; where a variable's comment says "not specified in
the paper", its value is our own choice. Each experiment keeps a *_PAPER
variant (a sample size and step count fine enough to resolve the paper's
examples) and a *_RUN variant (the configuration executed by default, to fit
the available computation time). Pass --full on any exp/exp*.py script to use
the *_PAPER configuration instead.

Running paths are never stored in full (see src/mc_core.py); peak memory for
one simulation call is O(n_paths * state_dim), so PATHS_CHUNK merely bounds
the batch size processed at once.
"""

SEED = 20261110  # fixed seed for reproducibility of every experiment, not specified by the paper
CONFIDENCE = 0.95  # confidence level for Monte Carlo intervals, not specified by the paper
PATHS_CHUNK = 20_000  # path-batch size used to bound peak memory per simulation call

# --------------------------------------------------------------------------
# exp_1: Loss-of-mass identity E_y[S_T] = Qtilde_y(tau_D > T), Lemma 2.1,
# one-dimensional CIR / 3-2 model of Example 6.3.
# --------------------------------------------------------------------------
EXP1_Y0 = 1.0  # initial value of the CIR factor Y_0 (Sec. 2.1 example), not specified -> set to the long-run scale
EXP1_KAPPA = 1.0  # mean-reversion speed kappa, not specified in the paper -> fixed to 1
EXP1_ETA = 1.0  # volatility of volatility eta, not specified in the paper -> fixed to 1
EXP1_RHO = -0.5  # asset/factor correlation rho, not specified in the paper -> representative value in (-1,1)
# alpha chosen so that 2*alpha >= eta^2 (Feller condition, Sec 2.1) in both the
# martingale case (alpha + rho*eta >= eta^2/2) and the strict local
# martingale case (alpha + rho*eta < eta^2/2), per Example 6.3 (Sec. 6).
EXP1_ALPHA_MARTINGALE = 1.2
EXP1_ALPHA_STRICT = 0.6
EXP1_T_GRID = (0.25, 0.5, 1.0, 2.0, 5.0)  # finite horizons tested, methodology step 8
EXP1_N_STEPS_PAPER = 10_000  # steps per unit time, implementation_notes: dt = T/10000
EXP1_N_STEPS_RUN = 2_000
EXP1_N_PATHS_PAPER = 100_000  # implementation_notes: N = 100,000 paths
EXP1_N_PATHS_RUN = 60_000
EXP1_EPS = 1e-10  # exit threshold for the shifted CIR factor, implementation_notes

# --------------------------------------------------------------------------
# exp_2: 3/2 model martingale criterion via the Fichera drift, Example 6.3.
# --------------------------------------------------------------------------
EXP2_Y0 = 1.0
EXP2_KAPPA = 1.0
EXP2_ETA = 1.0
EXP2_RHO = -0.5
# Three cases spanning the threshold alpha + rho*eta = eta^2/2 = 0.5 (eta=1):
EXP2_ALPHA_CASES = {
    "martingale": 1.2,     # alpha + rho*eta = 0.7 > 0.5
    "critical": 1.0,       # alpha + rho*eta = 0.5 = eta^2/2 (equality is a martingale)
    "strict_local": 0.6,   # alpha + rho*eta = 0.1 < 0.5
}
EXP2_T_GRID = (0.1, 0.5, 1.0, 2.0, 5.0)  # methodology step 5
EXP2_N_STEPS_PAPER = 20_000  # implementation_notes: dt = T/20000
EXP2_N_STEPS_RUN = 3_000
EXP2_N_PATHS_PAPER = 100_000
EXP2_N_PATHS_RUN = 60_000
EXP2_EPS = 1e-10

# --------------------------------------------------------------------------
# exp_3: Two-factor inverse-CIR model, Example 6.4.
# --------------------------------------------------------------------------
EXP3_Y0 = (1.0, 1.0)  # (Y1_0, Y2_0)
EXP3_ALPHA1, EXP3_KAPPA1, EXP3_ETA1 = 1.2, 1.0, 1.0
EXP3_ALPHA2, EXP3_KAPPA2, EXP3_ETA2 = 1.0, 1.0, 1.0
# g(y2) = 1 + y2/(1+y2) has sup g = 2 on [0, infinity); g == 1 is the single-factor limit.
EXP3_G_BAR = 2.0
EXP3_RHO_CASES = {
    "nonneg_rho": 0.5,     # rho >= 0: always a martingale
    "neg_rho_mart": -0.3,  # rho < 0, 2*alpha1-eta1^2=1.4 >= 2*eta1*|rho|*gbar=1.2 -> martingale
    "neg_rho_strict": -0.8,  # 2*alpha1-eta1^2=1.4 < 2*eta1*|rho|*gbar=3.2 -> strict local martingale
}
EXP3_T_GRID = (0.5, 1.0, 2.0, 5.0)
EXP3_N_STEPS_PAPER = 20_000
EXP3_N_STEPS_RUN = 3_000
EXP3_N_PATHS_PAPER = 100_000
EXP3_N_PATHS_RUN = 60_000
EXP3_EPS = 1e-10

# --------------------------------------------------------------------------
# exp_4: 4/2 model (one factor) and multi-factor 4/2 model, Example 6.5.
# --------------------------------------------------------------------------
EXP4_V0 = 1.0
EXP4_ALPHA, EXP4_KAPPA, EXP4_ETA = 0.4, 1.0, 1.0
EXP4_RHO = -0.5
EXP4_B_FIXED = 0.5  # alpha + rho*eta*b = 0.4-0.25 = 0.15 < eta^2/2 = 0.5 -> strict local martingale for every a
EXP4_A_GRID = (0.0, 0.5, 1.0, 2.0)  # Heston weight a, should not affect martingality (Corollary 3.1)
EXP4_B_GRID = (0.1, 0.3, 0.5, 0.7, 0.9, 1.1)  # crossing alpha+rho*eta*b = eta^2/2 at b = 0.8
# alpha used only for the vary_b sweep so that the crossing in EXP4_B_GRID actually falls
# inside the grid (Example 6.5 methodology step 6: "vary b to cross the threshold"); with
# EXP4_ALPHA=0.4 and EXP4_RHO<0 the threshold b*=-0.2 is never reached by b>=0.
EXP4_ALPHA_VARY_B = 0.9
EXP4_T_GRID = (0.5, 1.0, 2.0, 5.0)
EXP4_N_STEPS_PAPER = 20_000
EXP4_N_STEPS_RUN = 3_000
EXP4_N_PATHS_PAPER = 100_000
EXP4_N_PATHS_RUN = 20_000
EXP4_EPS = 1e-10
# Multi-factor extension (same factor parameters as exp_3) with a = 0 (inverse-CIR limit)
EXP4_MULTI_B = 1.0
EXP4_MULTI_RHO_CASES = EXP3_RHO_CASES

# --------------------------------------------------------------------------
# exp_5: Correlated square-root factors, Theorem 6.2 and Example 6.6.
# --------------------------------------------------------------------------
EXP5_ETA1 = EXP5_ETA2 = 1.0
EXP5_KAPPA1 = EXP5_KAPPA2 = 1.0
EXP5_ALPHA2 = 1.0
EXP5_RHO1 = -0.5  # d<B, Z1> ; B = -1/2 Z1 + sqrt(3)/2 Bperp
EXP5_R12 = 3.0 / 5.0  # d<Z1,Z2>, Example 6.6
EXP5_R12_ALT = 0.0  # alternate factor correlation used to check the threshold is R12-independent (methodology step 9)
EXP5_U = 0.0  # Heston-type weight u in ell(y)=u*sqrt(y1)+v*g(y2)/sqrt(y1); u=0 is the inverse-CIR case of Example 6.6
EXP5_U_ALT = 1.0  # u>0 (4/2-type) case used to check martingality is still u-independent of the threshold (Corollary 3.1 analogue)
EXP5_V = 1.0
EXP5_G_BAR = 2.0  # g(z) = 1 + z/(1+z)
EXP5_ALPHA1_CASES = {
    "martingale": 2.0,     # alpha1 = 2 >= 3/2 threshold
    "critical": 1.5,       # alpha1 = 3/2 (equality, martingale)
    "strict_local": 1.0,   # alpha1 = 1 < 3/2
}
EXP5_Y0 = (1.0, 1.0)
EXP5_T_GRID = (0.5, 1.0, 2.0, 5.0)
EXP5_N_STEPS_PAPER = 20_000
EXP5_N_STEPS_RUN = 3_000
EXP5_N_PATHS_PAPER = 100_000
EXP5_N_PATHS_RUN = 20_000
EXP5_EPS = 1e-10

# --------------------------------------------------------------------------
# exp_6: Coupled logarithmic threshold, Theorem 3.2 and Corollary 3.2.
# --------------------------------------------------------------------------
EXP6_X0 = 1.0
EXP6_KAPPA = 1.0  # critical CIR mean reversion kappa in dX=(1/2-kappa X)dt+sqrt(X)dZ0
EXP6_Z0 = 0.0
EXP6_LAMBDA = 1.0
EXP6_ETA_Z = 1.0
EXP6_RHO1 = -0.5  # correlation between W0 and W1
EXP6_CASES = {
    # c(z) = a + b*z^2/(1+z^2); sup c = a + b
    "martingale": (0.2, 0.0),       # sup c = 0.2 < 1/2
    "critical": (0.5, 0.0),         # sup c = 1/2 (equality, still a martingale)
    "strict_local": (0.3, 0.3),     # sup c = 0.6 > 1/2
    "trivial": (0.0, 0.0),          # c == 0, S == 1
}
EXP6_T_GRID = (0.5, 1.0, 2.0, 5.0)
EXP6_N_STEPS_PER_UNIT_PAPER = 50_000  # implementation_notes: dt = T/50000 (per unit time near the singular boundary)
EXP6_N_STEPS_PER_UNIT_RUN = 2_000
EXP6_N_PATHS_PAPER = 100_000
EXP6_N_PATHS_RUN = 10_000
EXP6_X_FLOOR = 1e-15
EXP6_LOG_CAP = None  # computed at runtime as log(1/EXP6_X_FLOOR)
# Separate (coarser) discretization for the long-horizon a.s.-convergence check
# (Corollary 3.2): resolving S_t -> 0 only needs a qualitative decay, not the
# fine boundary resolution used for the finite-horizon martingale test.
EXP6_LONG_T_GRID_PAPER = (50.0, 100.0)
EXP6_LONG_T_GRID_RUN = (20.0, 50.0)
EXP6_LONG_N_STEPS_PER_UNIT_PAPER = 2_000
EXP6_LONG_N_STEPS_PER_UNIT_RUN = 200
EXP6_LONG_N_PATHS_PAPER = 20_000
EXP6_LONG_N_PATHS_RUN = 5_000

# --------------------------------------------------------------------------
# exp_7: Isotropic multi-factor power-volatility model, Example 6.1.
# --------------------------------------------------------------------------
EXP7_D = 3  # number of OU factors (d >= 2 so Y never hits 0)
EXP7_KAPPA = 1.0
EXP7_ETA = 0.3
EXP7_Y0 = (1.0, 0.0, 0.0)
EXP7_CASES = {
    "neg_rho": (-0.5, 2.0),      # rho <= 0 -> always a martingale
    "p_le_1": (0.5, 0.5),        # p <= 1 -> always a martingale
    "strict_local": (0.5, 2.0),  # rho > 0 and p > 1 -> strict local martingale
    "boundary_p1": (0.5, 1.0),   # p == 1 boundary case -> martingale
}
EXP7_T_GRID = (0.5, 1.0, 2.0, 5.0, 10.0)
EXP7_N_STEPS_PAPER = 10_000  # implementation_notes: dt = T/10000 (fixed step count for every T)
EXP7_N_STEPS_RUN = 1_500
EXP7_N_PATHS_PAPER = 200_000  # implementation_notes: N = 200,000 paths
EXP7_N_PATHS_RUN = 20_000

# --------------------------------------------------------------------------
# exp_8: Weighted-distance criterion for inverse-distance singularities,
# Proposition 5.2 and Example 5.2.
# --------------------------------------------------------------------------
EXP8_M = 3  # ambient dimension of U (m >= 2 so U never hits 0)
EXP8_K = 1  # number of OU tangential factors
EXP8_U0 = (1.0, 0.0, 0.0)
EXP8_Z0 = 0.0
EXP8_LAMBDA = 1.0
EXP8_H = 1.0
EXP8_C_CASES = {
    "martingale": 0.2,       # sup c = 0.2 < (m-2)/2 = 0.5
    "critical": 0.5,         # sup c = 0.5 (equality, martingale)
    "strict_local": 0.7,     # sup c = 0.7 > 0.5
}
# Example 5.2: c(z) = 1/5 + (1/5) z^2/(1+z^2), sup c = 2/5 < 1/2, M = diag(2,1,1)
EXP8_EXAMPLE52_A, EXP8_EXAMPLE52_B = 0.2, 0.2
EXP8_EXAMPLE52_M_DIAG = (2.0, 1.0, 1.0)
EXP8_T_GRID = (0.5, 1.0, 2.0, 5.0)
EXP8_N_STEPS_PAPER = 10_000  # implementation_notes: dt = T/10000 (fixed step count for every T)
EXP8_N_STEPS_RUN = 1_500
EXP8_N_PATHS_PAPER = 200_000
EXP8_N_PATHS_RUN = 20_000

# --------------------------------------------------------------------------
# exp_9: Absorbed factor with correlated modulation, Theorem 6.3.
# --------------------------------------------------------------------------
EXP9_X0 = 1.0
EXP9_Z0 = 0.0
EXP9_LAMBDA = 1.0
EXP9_ETA_Z = 1.0
EXP9_RHO1 = 0.5
EXP9_CASES = {
    # c(z) = a + b/(1+z^2)
    "martingale": (0.6, 0.0),       # inf c = 0.6 > 1/2
    "critical": (0.5, 0.0),         # inf c = 0.5 (equality, martingale)
    "strict_local": (0.4, 0.2),     # inf c = 0.4 < 1/2, even though c(0)=0.6 > 1/2
    "classical": (1.0, 0.0),        # c == 1: S_t = X_t / x, shifted factor is a 3D Bessel process
}
EXP9_T_GRID = (0.5, 1.0, 2.0, 5.0)
EXP9_N_STEPS_PAPER = 20_000  # implementation_notes: fine steps near absorption
EXP9_N_STEPS_RUN = 4_000
EXP9_N_PATHS_PAPER = 200_000
EXP9_N_PATHS_RUN = 20_000
EXP9_X_FLOOR = 1e-10

# --------------------------------------------------------------------------
# exp_10: Wishart process boundary condition, Proposition 7.2.
# --------------------------------------------------------------------------
EXP10_N = 2  # matrix dimension
EXP10_Q = ((1.0, 0.0), (0.0, 1.0))  # Q = I
EXP10_M = ((0.0, 0.0), (0.0, 0.0))  # M = 0
EXP10_SIGMA0 = ((1.0, 0.0), (0.0, 1.0))  # Sigma_0 = I
EXP10_OMEGA_SQ_CASES = {
    "above_threshold": 4.0,   # omega^2 = 4 > n+1 = 3
    "critical": 3.0,          # omega^2 = 3 = n+1
    "below_threshold": 2.0,   # omega^2 = 2 < 3 (but > naive CIR threshold 2 is also violated at equality)
}
EXP10_T_GRID = (1.0, 5.0, 10.0)
EXP10_N_STEPS_PAPER = 10_000  # implementation_notes: dt = T/10000 (fixed step count for every T)
EXP10_N_STEPS_RUN = 1_500
EXP10_N_PATHS_PAPER = 100_000
EXP10_N_PATHS_RUN = 20_000
EXP10_EPS = 0.01  # boundary threshold on lambda_min, implementation_notes

# --------------------------------------------------------------------------
# exp_11: PDE uniqueness and the valuation equation, Theorem B.1 / Prop. B.1.
# --------------------------------------------------------------------------
EXP11_ALPHA_MARTINGALE = EXP2_ALPHA_CASES["martingale"]
EXP11_ALPHA_STRICT = EXP2_ALPHA_CASES["strict_local"]
EXP11_KAPPA = EXP2_KAPPA
EXP11_ETA = EXP2_ETA
EXP11_RHO = EXP2_RHO
EXP11_S_GRID_PAPER = (0.0, 10.0, 201)  # (s_min, s_max, n_points) implementation_notes: 200x200 grid
EXP11_Y_GRID_PAPER = (0.01, 5.0, 201)
EXP11_S_GRID_RUN = (0.0, 10.0, 81)
EXP11_Y_GRID_RUN = (0.01, 5.0, 81)
EXP11_T_HORIZON = 2.0
EXP11_N_TIME_STEPS_PAPER = 200  # implementation_notes: dt = T/100 (doubled for Crank-Nicolson stability)
EXP11_N_TIME_STEPS_RUN = 80
EXP11_MC_N_PATHS = 20_000  # Monte Carlo cross-check sample size (not specified by the paper)
EXP11_MC_N_STEPS = 2_000
