"""exp_5: Correlated square-root factors with boundary singularities
(Theorem 6.2, Example 6.6).

Verifies 2*alpha1 - eta1^2 >= 2*eta1*(rho1*v)^- *sup(g) for two correlated
CIR factors Y1, Y2 with d<Z1,Z2> = r12, g(y2) = 1 + y2/(1+y2) (sup g = 2).
Example 6.6 parameters give rho1 = -1/2, v = 1, so the threshold is
alpha1 >= 3/2. Also checks that the factor correlation r12 and the Heston
weight u do not change the martingale/strict-local-martingale classification
(methodology steps 7 and 9).

Run: python -m exp.exp5 [--full]
"""
from __future__ import annotations

import argparse
import csv
import json

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import config as cfg
from src.criteria import is_martingale_correlated_sqrt
from src.mc_core import make_rng, mc_mean_ci
from src.model_correlated_sqrt import simulate_factor_and_exponential, simulate_shifted_survival

RESULTS_DIR = "results"


def g_func(y2: np.ndarray) -> np.ndarray:
    """g(y2) = 1 + y2/(1+y2), bounded, sup g = 2 (Example 6.6)."""
    return 1.0 + y2 / (1.0 + y2)


def run(full: bool = False) -> tuple[list[dict], list[dict], list[dict]]:
    n_paths = cfg.EXP5_N_PATHS_PAPER if full else cfg.EXP5_N_PATHS_RUN
    n_steps = cfg.EXP5_N_STEPS_PAPER if full else cfg.EXP5_N_STEPS_RUN
    rng = make_rng(cfg.SEED)

    def simulate(alpha1, r12, u, T):
        s_t = simulate_factor_and_exponential(
            alpha1, cfg.EXP5_KAPPA1, cfg.EXP5_ETA1,
            cfg.EXP5_ALPHA2, cfg.EXP5_KAPPA2, cfg.EXP5_ETA2,
            cfg.EXP5_RHO1, r12, u, cfg.EXP5_V, g_func, cfg.EXP5_Y0,
            T, n_steps, n_paths, rng,
        )
        e_s, ci = mc_mean_ci(s_t, cfg.CONFIDENCE)
        exited = simulate_shifted_survival(
            alpha1, cfg.EXP5_KAPPA1, cfg.EXP5_ETA1,
            cfg.EXP5_ALPHA2, cfg.EXP5_KAPPA2, cfg.EXP5_ETA2,
            cfg.EXP5_RHO1, r12, u, cfg.EXP5_V, g_func, cfg.EXP5_Y0,
            T, n_steps, n_paths, rng, cfg.EXP5_EPS,
        )
        survival = 1.0 - exited.mean()
        return e_s, ci, survival

    # Main table: Example 6.6 cases across the T grid.
    rows = []
    for case, alpha1 in cfg.EXP5_ALPHA1_CASES.items():
        predicted = is_martingale_correlated_sqrt(alpha1, cfg.EXP5_ETA1, cfg.EXP5_RHO1, cfg.EXP5_V, cfg.EXP5_G_BAR)
        for T in cfg.EXP5_T_GRID:
            e_s, ci, survival = simulate(alpha1, cfg.EXP5_R12, cfg.EXP5_U, T)
            rows.append({
                "case": case, "alpha1": alpha1, "predicted_martingale": predicted, "T": T,
                "E[S_T]": e_s, "E[S_T]_ci_halfwidth": ci, "Qtilde(tau_D>T)": survival,
                "n_paths": n_paths, "n_steps": n_steps,
            })
            print(f"{case:12s} alpha1={alpha1:.2f} predicted_mart={predicted!s:5s} T={T:4.1f} "
                  f"E[S_T]={e_s:.4f}+-{ci:.4f} Qtilde={survival:.4f}")

    # Robustness: vary the factor correlation r12 at fixed T; the threshold on
    # alpha1 is independent of R_ij (methodology step 9 / Theorem 6.2).
    T_fixed = cfg.EXP5_T_GRID[len(cfg.EXP5_T_GRID) // 2]
    r12_rows = []
    for case, alpha1 in cfg.EXP5_ALPHA1_CASES.items():
        for r12 in (cfg.EXP5_R12, cfg.EXP5_R12_ALT):
            e_s, ci, survival = simulate(alpha1, r12, cfg.EXP5_U, T_fixed)
            r12_rows.append({"case": case, "alpha1": alpha1, "r12": r12, "T": T_fixed,
                              "E[S_T]": e_s, "E[S_T]_ci_halfwidth": ci})
            print(f"[r12 robustness] {case:12s} r12={r12:+.2f} T={T_fixed:.1f} E[S_T]={e_s:.4f}+-{ci:.4f}")

    # Robustness: vary the Heston weight u at fixed T; martingality should be
    # unaffected (Corollary 3.1-type invariance to bounded integrands).
    u_rows = []
    for case, alpha1 in cfg.EXP5_ALPHA1_CASES.items():
        for u in (cfg.EXP5_U, cfg.EXP5_U_ALT):
            e_s, ci, survival = simulate(alpha1, cfg.EXP5_R12, u, T_fixed)
            u_rows.append({"case": case, "alpha1": alpha1, "u": u, "T": T_fixed,
                            "E[S_T]": e_s, "E[S_T]_ci_halfwidth": ci})
            print(f"[u robustness]   {case:12s} u={u:.2f} T={T_fixed:.1f} E[S_T]={e_s:.4f}+-{ci:.4f}")

    return rows, r12_rows, u_rows


def save(rows: list[dict], r12_rows: list[dict], u_rows: list[dict]) -> None:
    for name, data in (("exp5_correlated_sqrt.csv", rows),
                        ("exp5_r12_robustness.csv", r12_rows),
                        ("exp5_u_robustness.csv", u_rows)):
        path = f"{RESULTS_DIR}/{name}"
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            writer.writeheader()
            writer.writerows(data)
        print(f"saved {path}")

    fig, ax = plt.subplots(figsize=(7, 5))
    for case in sorted({r["case"] for r in rows}):
        sub = [r for r in rows if r["case"] == case]
        ts = [r["T"] for r in sub]
        es = [r["E[S_T]"] for r in sub]
        ax.plot(ts, es, "o-", label=case)
    ax.axhline(1.0, color="gray", linewidth=0.5)
    ax.set_xlabel("T")
    ax.set_ylabel("E[S_T]")
    ax.set_title("exp_5: correlated square-root factors (Theorem 6.2, Example 6.6)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/exp5_correlated_sqrt.png", dpi=150)
    plt.close(fig)
    print(f"saved {RESULTS_DIR}/exp5_correlated_sqrt.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    rows, r12_rows, u_rows = run(full=args.full)
    save(rows, r12_rows, u_rows)
    with open(f"{RESULTS_DIR}/exp5_summary.json", "w") as f:
        json.dump({"full": args.full, "n_rows": len(rows)}, f, indent=2)


if __name__ == "__main__":
    main()
