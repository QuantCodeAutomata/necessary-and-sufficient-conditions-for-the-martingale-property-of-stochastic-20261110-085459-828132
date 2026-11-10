"""exp_1: Verification of the loss-of-mass formula E_y[S_T] = Qtilde_y(tau_D > T).

Reproduces Lemma 2.1 (Sec. 2.1) for the one-dimensional CIR / 3-2 model of
Example 6.3 (Sec. 6): the expected value of the stochastic exponential S_T
equals the survival probability of the Girsanov-shifted diffusion. Two
parameter sets are tested: a martingale case (alpha + rho*eta >= eta^2/2) and
a strict local martingale case (alpha + rho*eta < eta^2/2), across a grid of
finite horizons T.

Run: python -m exp.exp1 [--full]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import config as cfg
from src.mc_core import make_rng, mc_mean_ci
from src.model_one_factor_vol import simulate_factor_and_exponential, simulate_shifted_survival

RESULTS_DIR = "results"


def run(full: bool = False) -> list[dict]:
    n_paths = cfg.EXP1_N_PATHS_PAPER if full else cfg.EXP1_N_PATHS_RUN
    n_steps = cfg.EXP1_N_STEPS_PAPER if full else cfg.EXP1_N_STEPS_RUN
    rng = make_rng(cfg.SEED)

    cases = {
        "martingale (alpha+rho*eta=%.2f >= eta^2/2=0.5)" % (cfg.EXP1_ALPHA_MARTINGALE + cfg.EXP1_RHO * cfg.EXP1_ETA): cfg.EXP1_ALPHA_MARTINGALE,
        "strict_local (alpha+rho*eta=%.2f < eta^2/2=0.5)" % (cfg.EXP1_ALPHA_STRICT + cfg.EXP1_RHO * cfg.EXP1_ETA): cfg.EXP1_ALPHA_STRICT,
    }
    rows = []
    for label, alpha in cases.items():
        for T in cfg.EXP1_T_GRID:
            s_t = simulate_factor_and_exponential(
                alpha, cfg.EXP1_KAPPA, cfg.EXP1_ETA, cfg.EXP1_RHO, 0.0, 1.0,
                cfg.EXP1_Y0, T, n_steps, n_paths, rng,
            )
            e_s, e_s_ci = mc_mean_ci(s_t, cfg.CONFIDENCE)
            exited = simulate_shifted_survival(
                alpha, cfg.EXP1_KAPPA, cfg.EXP1_ETA, cfg.EXP1_RHO, 0.0, 1.0,
                cfg.EXP1_Y0, T, n_steps, n_paths, rng, cfg.EXP1_EPS,
            )
            survival = 1.0 - exited.mean()
            survival_se = exited.std(ddof=1) / np.sqrt(n_paths)
            rows.append({
                "case": label,
                "alpha": alpha,
                "T": T,
                "E[S_T]": e_s,
                "E[S_T]_ci_halfwidth": e_s_ci,
                "Qtilde(tau_D>T)": survival,
                "Qtilde_se": survival_se,
                "abs_diff": abs(e_s - survival),
                "n_paths": n_paths,
                "n_steps": n_steps,
            })
            print(f"{label:55s} T={T:5.2f}  E[S_T]={e_s:.4f}+-{e_s_ci:.4f}  "
                  f"Qtilde(tau_D>T)={survival:.4f}+-{1.96*survival_se:.4f}")
    return rows


def save(rows: list[dict]) -> None:
    import csv
    path = f"{RESULTS_DIR}/exp1_loss_of_mass.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {path}")

    fig, ax = plt.subplots(figsize=(7, 5))
    for label in sorted({r["case"] for r in rows}):
        sub = [r for r in rows if r["case"] == label]
        ts = [r["T"] for r in sub]
        es = [r["E[S_T]"] for r in sub]
        qs = [r["Qtilde(tau_D>T)"] for r in sub]
        ax.plot(ts, es, "o-", label=f"E[S_T] ({label})")
        ax.plot(ts, qs, "x--", label=f"Qtilde(tau_D>T) ({label})")
    ax.axhline(1.0, color="gray", linewidth=0.5)
    ax.set_xlabel("T")
    ax.set_ylabel("value")
    ax.set_title("exp_1: E_y[S_T] vs Qtilde_y(tau_D > T) (Lemma 2.1)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/exp1_loss_of_mass.png", dpi=150)
    plt.close(fig)
    print(f"saved {RESULTS_DIR}/exp1_loss_of_mass.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="use the paper-scale Monte Carlo configuration")
    args = parser.parse_args()
    rows = run(full=args.full)
    save(rows)
    with open(f"{RESULTS_DIR}/exp1_summary.json", "w") as f:
        json.dump({"full": args.full, "n_cases": len(rows)}, f, indent=2)


if __name__ == "__main__":
    main()
