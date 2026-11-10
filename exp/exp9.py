"""exp_9: Absorbed factor with correlated modulation (Theorem 6.3).

X_t = x + W0_{t up to tau0} is a Brownian motion absorbed at 0, Z_t a
correlated OU factor, and c(z) = a + b/(1+z^2) so that inf_z c(z) = a.
Verifies is_martingale_absorbed(inf_c): martingale iff inf_c >= 1/2, including
the "strict_local" case where c(Z0) > 1/2 at the starting point but the global
infimum a=0.4 < 1/2 still makes S a strict local martingale (a favorable
initial value does not save martingality -- only the infimum over the whole
state space matters), and the "classical" case c==1 where the paper shows S_t
= X_t/x exactly and S is a true martingale (E[S_T]=1).

Run: python -m exp.exp9 [--full]
"""
from __future__ import annotations

import argparse
import csv
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import config as cfg
from src.criteria import is_martingale_absorbed
from src.mc_core import make_rng, mc_mean_ci
from src.model_absorbed import c_function, simulate_density

RESULTS_DIR = "results"


def run(full: bool = False) -> list[dict]:
    n_paths = cfg.EXP9_N_PATHS_PAPER if full else cfg.EXP9_N_PATHS_RUN
    n_steps = cfg.EXP9_N_STEPS_PAPER if full else cfg.EXP9_N_STEPS_RUN
    rng = make_rng(cfg.SEED)

    rows = []
    for case, (a, b) in cfg.EXP9_CASES.items():
        inf_c = a
        predicted = is_martingale_absorbed(inf_c)
        c_fn = c_function(a, b)
        for T in cfg.EXP9_T_GRID:
            s_t = simulate_density(
                cfg.EXP9_X0, cfg.EXP9_Z0, cfg.EXP9_LAMBDA, cfg.EXP9_ETA_Z, cfg.EXP9_RHO1,
                c_fn, T, n_steps, n_paths, rng, cfg.EXP9_X_FLOOR,
            )
            e_s, ci = mc_mean_ci(s_t, cfg.CONFIDENCE)
            p_absorbed = float((s_t == 0.0).mean())
            rows.append({
                "case": case, "inf_c": inf_c, "predicted_martingale": predicted, "T": T,
                "E[S_T]": e_s, "E[S_T]_ci_halfwidth": ci, "P(absorbed_by_T)": p_absorbed,
                "n_paths": n_paths, "n_steps": n_steps,
            })
            print(f"{case:14s} inf_c={inf_c:.2f} predicted_mart={predicted!s:5s} T={T:4.1f} "
                  f"E[S_T]={e_s:.4f}+-{ci:.4f} P(absorbed)={p_absorbed:.4f}")

            if case == "classical":
                # The paper states S_t = X_t/x exactly for c == 1, hence S is a
                # true martingale and E[S_T] should be close to 1 (not a hard
                # assertion: just a warning if the Monte Carlo estimate drifts).
                if abs(e_s - 1.0) > ci + 0.05:
                    print(f"  WARNING: classical case E[S_T]={e_s:.4f}+-{ci:.4f} not close to 1 at T={T}")
    return rows


def save(rows: list[dict]) -> None:
    path = f"{RESULTS_DIR}/exp9_absorbed.csv"
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
    ax.set_title("exp_9: absorbed factor with correlated modulation (Theorem 6.3)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/exp9_absorbed.png", dpi=150)
    plt.close(fig)
    print(f"saved {RESULTS_DIR}/exp9_absorbed.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    rows = run(full=args.full)
    save(rows)
    with open(f"{RESULTS_DIR}/exp9_summary.json", "w") as f:
        json.dump({"full": args.full, "n_rows": len(rows)}, f, indent=2)


if __name__ == "__main__":
    main()
