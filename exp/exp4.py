"""exp_4: 4/2 model and multi-factor 4/2 martingale criterion (Example 6.5).

Part A (one-factor 4/2): verifies alpha + rho*eta*b >= eta^2/2, independent of
the Heston weight a (Corollary 3.1): E[S_T] is computed for several values of
a at fixed b (should be unchanged), then b is varied across the threshold.
Part B (multi-factor 4/2, a=0): verifies 2*alpha1-eta1^2 >= 2*eta1*(rho*b)^-
*sup(g) using the two-factor machinery of src/model_multi_factor.py.

Run: python -m exp.exp4 [--full]
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
from src.criteria import is_martingale_4_2, is_martingale_multi_factor_4_2
from src.mc_core import make_rng, mc_mean_ci
from src.model_one_factor_vol import (
    simulate_factor_and_exponential as sim_one_factor,
    simulate_shifted_survival as sim_one_factor_survival,
)
from src.model_multi_factor import (
    simulate_factor_and_exponential as sim_multi_factor,
    simulate_shifted_survival as sim_multi_factor_survival,
)
from exp.exp3 import g_func

RESULTS_DIR = "results"


def run_one_factor(full: bool) -> list[dict]:
    n_paths = cfg.EXP4_N_PATHS_PAPER if full else cfg.EXP4_N_PATHS_RUN
    n_steps = cfg.EXP4_N_STEPS_PAPER if full else cfg.EXP4_N_STEPS_RUN
    rng = make_rng(cfg.SEED)
    rows = []

    # Vary the Heston weight a at fixed b (strict local martingale case for all a).
    for a in cfg.EXP4_A_GRID:
        predicted = is_martingale_4_2(cfg.EXP4_ALPHA, cfg.EXP4_ETA, cfg.EXP4_RHO, cfg.EXP4_B_FIXED)
        for T in cfg.EXP4_T_GRID:
            s_t = sim_one_factor(cfg.EXP4_ALPHA, cfg.EXP4_KAPPA, cfg.EXP4_ETA, cfg.EXP4_RHO,
                                  a, cfg.EXP4_B_FIXED, cfg.EXP4_V0, T, n_steps, n_paths, rng)
            e_s, ci = mc_mean_ci(s_t, cfg.CONFIDENCE)
            rows.append({"sweep": "vary_a", "a": a, "b": cfg.EXP4_B_FIXED, "T": T,
                         "predicted_martingale": predicted, "E[S_T]": e_s, "ci_halfwidth": ci,
                         "n_paths": n_paths, "n_steps": n_steps})
            print(f"vary_a  a={a:.2f} b={cfg.EXP4_B_FIXED:.2f} T={T:4.1f} E[S_T]={e_s:.4f}+-{ci:.4f}")

    # Vary b at a=0 (pure 3/2 behaviour, inverse-CIR limit) across the threshold.
    # EXP4_ALPHA_VARY_B (not EXP4_ALPHA) is used here so the crossing alpha+rho*eta*b=eta^2/2
    # actually falls inside EXP4_B_GRID, as required to show the martingale -> strict local
    # martingale transition (Example 6.5 methodology step 6).
    for b in cfg.EXP4_B_GRID:
        predicted = is_martingale_4_2(cfg.EXP4_ALPHA_VARY_B, cfg.EXP4_ETA, cfg.EXP4_RHO, b)
        T = cfg.EXP4_T_GRID[-1]
        s_t = sim_one_factor(cfg.EXP4_ALPHA_VARY_B, cfg.EXP4_KAPPA, cfg.EXP4_ETA, cfg.EXP4_RHO,
                              0.0, b, cfg.EXP4_V0, T, n_steps, n_paths, rng)
        e_s, ci = mc_mean_ci(s_t, cfg.CONFIDENCE)
        rows.append({"sweep": "vary_b", "a": 0.0, "b": b, "T": T,
                     "predicted_martingale": predicted, "E[S_T]": e_s, "ci_halfwidth": ci,
                     "n_paths": n_paths, "n_steps": n_steps})
        print(f"vary_b  b={b:.2f} T={T:4.1f} predicted_mart={predicted!s:5s} E[S_T]={e_s:.4f}+-{ci:.4f}")
    return rows


def run_multi_factor(full: bool) -> list[dict]:
    n_paths = cfg.EXP4_N_PATHS_PAPER if full else cfg.EXP4_N_PATHS_RUN
    n_steps = cfg.EXP4_N_STEPS_PAPER if full else cfg.EXP4_N_STEPS_RUN
    rng = make_rng(cfg.SEED)
    rows = []
    for case, rho in cfg.EXP4_MULTI_RHO_CASES.items():
        predicted = is_martingale_multi_factor_4_2(cfg.EXP3_ALPHA1, cfg.EXP3_ETA1, rho,
                                                    cfg.EXP4_MULTI_B, cfg.EXP3_G_BAR)
        for T in cfg.EXP4_T_GRID:
            s_t = sim_multi_factor(cfg.EXP3_ALPHA1, cfg.EXP3_KAPPA1, cfg.EXP3_ETA1,
                                    cfg.EXP3_ALPHA2, cfg.EXP3_KAPPA2, cfg.EXP3_ETA2,
                                    rho, 0.0, cfg.EXP4_MULTI_B, g_func, cfg.EXP3_Y0,
                                    T, n_steps, n_paths, rng)
            e_s, ci = mc_mean_ci(s_t, cfg.CONFIDENCE)
            rows.append({"case": case, "rho": rho, "predicted_martingale": predicted, "T": T,
                         "E[S_T]": e_s, "ci_halfwidth": ci, "n_paths": n_paths, "n_steps": n_steps})
            print(f"multi {case:18s} rho={rho:+.2f} T={T:4.1f} predicted_mart={predicted!s:5s} "
                  f"E[S_T]={e_s:.4f}+-{ci:.4f}")
    return rows


def save(one_factor: list[dict], multi_factor: list[dict]) -> None:
    for name, rows in [("exp4_one_factor_4_2", one_factor), ("exp4_multi_factor_4_2", multi_factor)]:
        path = f"{RESULTS_DIR}/{name}.csv"
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"saved {path}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    vary_a = [r for r in one_factor if r["sweep"] == "vary_a"]
    for a in sorted({r["a"] for r in vary_a}):
        sub = [r for r in vary_a if r["a"] == a]
        axes[0].plot([r["T"] for r in sub], [r["E[S_T]"] for r in sub], "o-", label=f"a={a}")
    axes[0].axhline(1.0, color="gray", linewidth=0.5)
    axes[0].set_xlabel("T"); axes[0].set_ylabel("E[S_T]")
    axes[0].set_title("E[S_T] vs T for varying Heston weight a (Corollary 3.1)")
    axes[0].legend(fontsize=7)

    vary_b = [r for r in one_factor if r["sweep"] == "vary_b"]
    axes[1].plot([r["b"] for r in vary_b], [r["E[S_T]"] for r in vary_b], "o-")
    axes[1].axvline(0.8, color="red", linestyle="--", label="threshold b=(eta^2/2-alpha)/(rho*eta)")
    axes[1].axhline(1.0, color="gray", linewidth=0.5)
    axes[1].set_xlabel("b"); axes[1].set_ylabel("E[S_T]")
    axes[1].set_title("E[S_T] vs b crossing alpha+rho*eta*b=eta^2/2")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/exp4_4_2_model.png", dpi=150)
    plt.close(fig)
    print(f"saved {RESULTS_DIR}/exp4_4_2_model.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    one_factor = run_one_factor(full=args.full)
    multi_factor = run_multi_factor(full=args.full)
    save(one_factor, multi_factor)
    with open(f"{RESULTS_DIR}/exp4_summary.json", "w") as f:
        json.dump({"full": args.full, "n_one_factor": len(one_factor), "n_multi_factor": len(multi_factor)}, f, indent=2)


if __name__ == "__main__":
    main()
