"""exp_3: Two-factor inverse-CIR model martingale criterion (Example 6.4).

Verifies 2*alpha1 - eta1^2 >= 2*eta1*rho^- *sup(g) with rho^- = max(-rho, 0),
for g(y2) = 1 + y2/(1+y2) (sup g = 2). Cases: rho >= 0 (always a martingale),
rho < 0 with the inequality holding (martingale), and rho < 0 with the
inequality violated (strict local martingale).

Run: python -m exp.exp3 [--full]
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
from src.criteria import is_martingale_two_factor_inverse_cir
from src.mc_core import make_rng, mc_mean_ci
from src.model_multi_factor import simulate_factor_and_exponential, simulate_shifted_survival

RESULTS_DIR = "results"


def g_func(y2: np.ndarray) -> np.ndarray:
    """g(y2) = 1 + y2/(1+y2), bounded, sup g = 2 (methodology step 6)."""
    return 1.0 + y2 / (1.0 + y2)


def run(full: bool = False) -> list[dict]:
    n_paths = cfg.EXP3_N_PATHS_PAPER if full else cfg.EXP3_N_PATHS_RUN
    n_steps = cfg.EXP3_N_STEPS_PAPER if full else cfg.EXP3_N_STEPS_RUN
    rng = make_rng(cfg.SEED)

    rows = []
    for case, rho in cfg.EXP3_RHO_CASES.items():
        predicted = is_martingale_two_factor_inverse_cir(cfg.EXP3_ALPHA1, cfg.EXP3_ETA1, rho, cfg.EXP3_G_BAR)
        for T in cfg.EXP3_T_GRID:
            s_t = simulate_factor_and_exponential(
                cfg.EXP3_ALPHA1, cfg.EXP3_KAPPA1, cfg.EXP3_ETA1,
                cfg.EXP3_ALPHA2, cfg.EXP3_KAPPA2, cfg.EXP3_ETA2,
                rho, 0.0, 1.0, g_func, cfg.EXP3_Y0, T, n_steps, n_paths, rng,
            )
            e_s, ci = mc_mean_ci(s_t, cfg.CONFIDENCE)
            exited = simulate_shifted_survival(
                cfg.EXP3_ALPHA1, cfg.EXP3_KAPPA1, cfg.EXP3_ETA1,
                cfg.EXP3_ALPHA2, cfg.EXP3_KAPPA2, cfg.EXP3_ETA2,
                rho, 0.0, 1.0, g_func, cfg.EXP3_Y0, T, n_steps, n_paths, rng, cfg.EXP3_EPS,
            )
            survival = 1.0 - exited.mean()
            rows.append({
                "case": case, "rho": rho, "predicted_martingale": predicted, "T": T,
                "E[S_T]": e_s, "E[S_T]_ci_halfwidth": ci, "Qtilde(tau_D>T)": survival,
                "n_paths": n_paths, "n_steps": n_steps,
            })
            print(f"{case:18s} rho={rho:+.2f} predicted_mart={predicted!s:5s} T={T:4.1f} "
                  f"E[S_T]={e_s:.4f}+-{ci:.4f} Qtilde={survival:.4f}")
    return rows


def save(rows: list[dict]) -> None:
    path = f"{RESULTS_DIR}/exp3_two_factor_inverse_cir.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
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
    ax.set_title("exp_3: two-factor inverse-CIR model (Example 6.4)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/exp3_two_factor_inverse_cir.png", dpi=150)
    plt.close(fig)
    print(f"saved {RESULTS_DIR}/exp3_two_factor_inverse_cir.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    rows = run(full=args.full)
    save(rows)
    with open(f"{RESULTS_DIR}/exp3_summary.json", "w") as f:
        json.dump({"full": args.full, "n_rows": len(rows)}, f, indent=2)


if __name__ == "__main__":
    main()
