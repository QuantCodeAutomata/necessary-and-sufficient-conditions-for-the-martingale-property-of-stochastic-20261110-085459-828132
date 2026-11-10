"""exp_10: Wishart process boundary condition (Proposition 7.2).

For the n=2 Wishart process with Q = I, M = 0, isotropic Omega = omega*I,
checks that the exact boundary condition Omega Omega^T >= (n+1) Q^T Q
(omega^2 >= 3*q^2 here) predicts whether the lower boundary of the cone of
positive semi-definite matrices is inaccessible, as opposed to the naive
scalar-CIR-style threshold (factor 2 instead of n+1), by Monte Carlo
estimation of P(min_t<=T lambda_min(Sigma_t) <= eps) for each omega^2 case
and horizon T.

Run: python -m exp.exp10 [--full]
"""
from __future__ import annotations

import argparse
import csv
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import config as cfg
from src.criteria import (
    is_wishart_boundary_inaccessible,
    naive_cir_threshold_is_wrong,
    wishart_fichera_drift,
)
from src.mc_core import make_rng
from src.model_wishart import boundary_hit_probability, simulate_min_eigenvalue_paths

RESULTS_DIR = "results"


def run(full: bool = False) -> list[dict]:
    n_paths = cfg.EXP10_N_PATHS_PAPER if full else cfg.EXP10_N_PATHS_RUN
    n_steps = cfg.EXP10_N_STEPS_PAPER if full else cfg.EXP10_N_STEPS_RUN
    rng = make_rng(cfg.SEED)
    q_sq = cfg.EXP10_Q[0][0] ** 2

    rows = []
    for case, omega_sq in cfg.EXP10_OMEGA_SQ_CASES.items():
        predicted = is_wishart_boundary_inaccessible(omega_sq, q_sq, cfg.EXP10_N)
        naive_wrong = naive_cir_threshold_is_wrong(omega_sq, q_sq, cfg.EXP10_N)
        fichera = wishart_fichera_drift(omega_sq, q_sq, cfg.EXP10_N)
        for T in cfg.EXP10_T_GRID:
            min_eig = simulate_min_eigenvalue_paths(
                cfg.EXP10_N, omega_sq, cfg.EXP10_SIGMA0, T, n_steps, n_paths, rng,
            )
            p_hit = boundary_hit_probability(min_eig, cfg.EXP10_EPS)
            rows.append({
                "case": case, "omega_sq": omega_sq, "fichera_drift": fichera,
                "predicted_boundary_inaccessible": predicted,
                "naive_cir_threshold_is_wrong": naive_wrong,
                "T": T, "P(boundary_hit_by_T)": p_hit,
                "n_paths": n_paths, "n_steps": n_steps,
            })
            print(f"{case:16s} omega_sq={omega_sq:.2f} fichera={fichera:+.2f} "
                  f"predicted_inaccessible={predicted!s:5s} naive_wrong={naive_wrong!s:5s} "
                  f"T={T:5.1f} P(hit)={p_hit:.4f}")
    return rows


def save(rows: list[dict]) -> None:
    path = f"{RESULTS_DIR}/exp10_wishart.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {path}")

    fig, ax = plt.subplots(figsize=(7, 5))
    for case in sorted({r["case"] for r in rows}):
        sub = [r for r in rows if r["case"] == case]
        ts = [r["T"] for r in sub]
        ps = [r["P(boundary_hit_by_T)"] for r in sub]
        ax.plot(ts, ps, "o-", label=case)
    ax.set_xlabel("T")
    ax.set_ylabel("P(boundary hit by T)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("exp_10: Wishart boundary accessibility (Proposition 7.2)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/exp10_wishart.png", dpi=150)
    plt.close(fig)
    print(f"saved {RESULTS_DIR}/exp10_wishart.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    rows = run(full=args.full)
    save(rows)
    with open(f"{RESULTS_DIR}/exp10_summary.json", "w") as f:
        json.dump({"full": args.full, "n_rows": len(rows)}, f, indent=2)


if __name__ == "__main__":
    main()
