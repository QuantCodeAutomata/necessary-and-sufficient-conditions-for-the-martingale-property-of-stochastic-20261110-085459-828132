"""exp_2: 3/2 model martingale criterion via the Fichera drift (Example 6.3).

Verifies the necessary-and-sufficient condition alpha + rho*eta >= eta^2/2
across a martingale case, the critical equality case, and a strict local
martingale case, for a grid of horizons T. Also reports the analytic Fichera
drift F(0) = -(alpha + rho*eta - eta^2/2) for each case.

Run: python -m exp.exp2 [--full]
"""
from __future__ import annotations

import argparse
import csv
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import config as cfg
from src.criteria import fichera_drift_3_2, is_martingale_3_2
from src.mc_core import make_rng, mc_mean_ci
from src.model_one_factor_vol import simulate_factor_and_exponential, simulate_shifted_survival

RESULTS_DIR = "results"


def run(full: bool = False) -> list[dict]:
    n_paths = cfg.EXP2_N_PATHS_PAPER if full else cfg.EXP2_N_PATHS_RUN
    n_steps = cfg.EXP2_N_STEPS_PAPER if full else cfg.EXP2_N_STEPS_RUN
    rng = make_rng(cfg.SEED)

    rows = []
    for case, alpha in cfg.EXP2_ALPHA_CASES.items():
        fichera = fichera_drift_3_2(alpha, cfg.EXP2_ETA, cfg.EXP2_RHO)
        predicted = is_martingale_3_2(alpha, cfg.EXP2_ETA, cfg.EXP2_RHO)
        for T in cfg.EXP2_T_GRID:
            s_t = simulate_factor_and_exponential(
                alpha, cfg.EXP2_KAPPA, cfg.EXP2_ETA, cfg.EXP2_RHO, 0.0, 1.0,
                cfg.EXP2_Y0, T, n_steps, n_paths, rng,
            )
            e_s, ci = mc_mean_ci(s_t, cfg.CONFIDENCE)
            exited = simulate_shifted_survival(
                alpha, cfg.EXP2_KAPPA, cfg.EXP2_ETA, cfg.EXP2_RHO, 0.0, 1.0,
                cfg.EXP2_Y0, T, n_steps, n_paths, rng, cfg.EXP2_EPS,
            )
            survival = 1.0 - exited.mean()
            rows.append({
                "case": case, "alpha": alpha, "fichera_drift_F(0)": fichera,
                "predicted_martingale": predicted, "T": T, "E[S_T]": e_s,
                "E[S_T]_ci_halfwidth": ci, "Qtilde(tau_D>T)": survival,
                "n_paths": n_paths, "n_steps": n_steps,
            })
            print(f"{case:13s} alpha={alpha:.2f} F(0)={fichera:+.3f} predicted_mart={predicted!s:5s} "
                  f"T={T:4.1f} E[S_T]={e_s:.4f}+-{ci:.4f} Qtilde={survival:.4f}")
    return rows


def save(rows: list[dict]) -> None:
    path = f"{RESULTS_DIR}/exp2_3_2_model.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {path}")

    fig, ax = plt.subplots(figsize=(7, 5))
    for case in sorted({r["case"] for r in rows}):
        sub = [r for r in rows if r["case"] == case]
        ax.plot([r["T"] for r in sub], [r["E[S_T]"] for r in sub], "o-", label=case)
    ax.axhline(1.0, color="gray", linewidth=0.5)
    ax.set_xlabel("T")
    ax.set_ylabel("E[S_T]")
    ax.set_title("exp_2: 3/2 model, E[S_T] vs T (Example 6.3)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/exp2_3_2_model.png", dpi=150)
    plt.close(fig)
    print(f"saved {RESULTS_DIR}/exp2_3_2_model.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    rows = run(full=args.full)
    save(rows)
    with open(f"{RESULTS_DIR}/exp2_summary.json", "w") as f:
        json.dump({"full": args.full, "n_rows": len(rows)}, f, indent=2)


if __name__ == "__main__":
    main()
