"""exp_8: Weighted-distance criterion for inverse-distance singularities
(Proposition 5.2, Example 5.2).

Verifies sup c <= (m-2)/2 for the constant-c cases in EXP8_C_CASES and for
Example 5.2 (c(z) = A + B*z^2/(1+z^2), sup c = A + B), comparing the Monte
Carlo estimate of E[S_T] with the Girsanov-shifted survival probability
Qtilde(tau_D > T). Also reports the naive Euclidean-projection diagnostic
integral for Example 5.2, which is finite (inconclusive) even though the
sharp weighted-distance criterion gives a definite martingale verdict.

Run: python -m exp.exp8 [--full]
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
from src.criteria import is_martingale_weighted_distance
from src.mc_core import make_rng, mc_mean_ci
from src.model_weighted_distance import (
    euclidean_projection_integral,
    simulate_exponential,
    simulate_shifted_survival,
)

RESULTS_DIR = "results"


def _simulate_row(case: str, c_fn, sup_c: float, T: float, n_paths: int, n_steps: int,
                   rng: np.random.Generator) -> dict:
    predicted = is_martingale_weighted_distance(sup_c, cfg.EXP8_M)
    s_t = simulate_exponential(
        cfg.EXP8_M, cfg.EXP8_K, cfg.EXP8_LAMBDA, cfg.EXP8_H, cfg.EXP8_U0, cfg.EXP8_Z0,
        c_fn, T, n_steps, n_paths, rng,
    )
    e_s, ci = mc_mean_ci(s_t, cfg.CONFIDENCE)
    exited = simulate_shifted_survival(
        cfg.EXP8_M, cfg.EXP8_K, cfg.EXP8_LAMBDA, cfg.EXP8_H, cfg.EXP8_U0, cfg.EXP8_Z0,
        c_fn, T, n_steps, n_paths, rng,
    )
    survival = 1.0 - exited.mean()
    row = {
        "case": case, "c": sup_c, "predicted_martingale": predicted, "T": T,
        "E[S_T]": e_s, "E[S_T]_ci_halfwidth": ci, "Qtilde(tau_D>T)": survival,
        "n_paths": n_paths, "n_steps": n_steps,
    }
    print(f"{case:18s} c={sup_c:.3f} predicted_mart={predicted!s:5s} T={T:4.1f} "
          f"E[S_T]={e_s:.4f}+-{ci:.4f} Qtilde={survival:.4f}")
    return row


def run(full: bool = False) -> tuple[list[dict], dict]:
    n_paths = cfg.EXP8_N_PATHS_PAPER if full else cfg.EXP8_N_PATHS_RUN
    n_steps = cfg.EXP8_N_STEPS_PAPER if full else cfg.EXP8_N_STEPS_RUN
    rng = make_rng(cfg.SEED)

    rows = []
    for case, c in cfg.EXP8_C_CASES.items():
        c_fn = lambda z, c=c: np.full_like(z, c)
        for T in cfg.EXP8_T_GRID:
            rows.append(_simulate_row(case, c_fn, c, T, n_paths, n_steps, rng))

    # Example 5.2: c(z) = A + B*z^2/(1+z^2), sup c = A + B.
    sup_c = cfg.EXP8_EXAMPLE52_A + cfg.EXP8_EXAMPLE52_B

    def example52_c_fn(z: np.ndarray) -> np.ndarray:
        return cfg.EXP8_EXAMPLE52_A + cfg.EXP8_EXAMPLE52_B * z ** 2 / (1.0 + z ** 2)

    for T in cfg.EXP8_T_GRID:
        rows.append(_simulate_row("example_5_2", example52_c_fn, sup_c, T, n_paths, n_steps, rng))

    naive_j = euclidean_projection_integral(sup_c)
    sharp_predicted = is_martingale_weighted_distance(sup_c, cfg.EXP8_M)
    example52_row = {
        "sup_c": sup_c, "m": cfg.EXP8_M, "naive_integral_J": naive_j,
        "naive_test_inconclusive": True, "sharp_predicted_martingale": sharp_predicted,
    }
    print(f"[Example 5.2] sup_c={sup_c:.3f} naive_integral_J={naive_j:.4f} "
          f"(finite -> inconclusive) sharp_predicted_martingale={sharp_predicted!s}")

    return rows, example52_row


def save(rows: list[dict], example52_row: dict) -> None:
    path = f"{RESULTS_DIR}/exp8_weighted_distance.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {path}")

    proj_path = f"{RESULTS_DIR}/exp8_euclidean_projection.csv"
    with open(proj_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(example52_row.keys()))
        writer.writeheader()
        writer.writerow(example52_row)
    print(f"saved {proj_path}")

    fig, ax = plt.subplots(figsize=(7, 5))
    for case in sorted({r["case"] for r in rows}):
        sub = [r for r in rows if r["case"] == case]
        ts = [r["T"] for r in sub]
        es = [r["E[S_T]"] for r in sub]
        ax.plot(ts, es, "o-", label=case)
    ax.axhline(1.0, color="gray", linewidth=0.5)
    ax.set_xlabel("T")
    ax.set_ylabel("E[S_T]")
    ax.set_title("exp_8: weighted-distance criterion (Proposition 5.2, Example 5.2)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/exp8_weighted_distance.png", dpi=150)
    plt.close(fig)
    print(f"saved {RESULTS_DIR}/exp8_weighted_distance.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    rows, example52_row = run(full=args.full)
    save(rows, example52_row)
    with open(f"{RESULTS_DIR}/exp8_summary.json", "w") as f:
        json.dump({"full": args.full, "n_rows": len(rows)}, f, indent=2)


if __name__ == "__main__":
    main()
