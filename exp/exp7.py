"""exp_7: Isotropic multi-factor power-volatility model (Example 6.1).

Verifies the martingale criterion rho <= 0 or p <= 1 for the stochastic
exponential dS/S = |Y|^p dB, where B = rho*<Y/|Y|, dZ> + sqrt(1-rho^2)*dBperp
and Y is a d-dimensional Ornstein-Uhlenbeck factor (d >= 2). Cases (from
EXP7_CASES): rho <= 0 (always a martingale), p <= 1 (always a martingale),
rho > 0 and p > 1 (strict local martingale), and the p == 1 boundary case
(martingale, equality in the criterion).

In addition to the Monte Carlo mean of S_T, once per case this script runs
feller_test_v_infinity, a finite-truncation numerical proxy for the Feller
test at infinity on the Girsanov-shifted radial process (paper Example 6.1 /
Proposition 4.2): v(R) growing without bound as R -> infinity corresponds to
no explosion (true martingale), while v(R) plateauing at a finite value
corresponds to explosion (strict local martingale).

Run: python -m exp.exp7 [--full]
"""
from __future__ import annotations

import argparse
import csv
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import config as cfg
from src.criteria import is_martingale_power_volatility
from src.mc_core import make_rng, mc_mean_ci
from src.model_isotropic_ou import feller_test_v_infinity, simulate_path_and_exponential

RESULTS_DIR = "results"


def run(full: bool = False) -> tuple[list[dict], list[dict]]:
    n_paths = cfg.EXP7_N_PATHS_PAPER if full else cfg.EXP7_N_PATHS_RUN
    n_steps = cfg.EXP7_N_STEPS_PAPER if full else cfg.EXP7_N_STEPS_RUN
    rng = make_rng(cfg.SEED)

    rows = []
    feller_rows = []
    for case, (rho, p) in cfg.EXP7_CASES.items():
        predicted = is_martingale_power_volatility(rho, p)
        for T in cfg.EXP7_T_GRID:
            s_t = simulate_path_and_exponential(
                cfg.EXP7_KAPPA, cfg.EXP7_ETA, rho, p, cfg.EXP7_Y0, T, n_steps, n_paths, rng,
            )
            e_s, ci = mc_mean_ci(s_t, cfg.CONFIDENCE)
            rows.append({
                "case": case, "rho": rho, "p": p, "predicted_martingale": predicted, "T": T,
                "E[S_T]": e_s, "E[S_T]_ci_halfwidth": ci, "n_paths": n_paths, "n_steps": n_steps,
            })
            print(f"{case:14s} rho={rho:+.2f} p={p:.2f} predicted_mart={predicted!s:5s} T={T:5.1f} "
                  f"E[S_T]={e_s:.4f}+-{ci:.4f}")

        feller = feller_test_v_infinity(cfg.EXP7_D, cfg.EXP7_KAPPA, cfg.EXP7_ETA, rho, p)
        r_values = sorted(k for k in feller if k != "diverging")
        feller_row = {"case": case, "rho": rho, "p": p, "predicted_martingale": predicted}
        for i, R in enumerate(r_values, start=1):
            feller_row[f"v_R{int(R)}"] = feller[R]
        feller_row["diverging"] = feller["diverging"]
        feller_rows.append(feller_row)
        print(f"[feller test]  {case:14s} rho={rho:+.2f} p={p:.2f} "
              f"v(R)={[feller[R] for R in r_values]} diverging={feller['diverging']}")
    return rows, feller_rows


def save(rows: list[dict], feller_rows: list[dict]) -> None:
    path = f"{RESULTS_DIR}/exp7_power_volatility.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {path}")

    feller_path = f"{RESULTS_DIR}/exp7_feller_test.csv"
    with open(feller_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(feller_rows[0].keys()))
        writer.writeheader()
        writer.writerows(feller_rows)
    print(f"saved {feller_path}")

    fig, ax = plt.subplots(figsize=(7, 5))
    for case in sorted({r["case"] for r in rows}):
        sub = [r for r in rows if r["case"] == case]
        ts = [r["T"] for r in sub]
        es = [r["E[S_T]"] for r in sub]
        ax.plot(ts, es, "o-", label=case)
    ax.axhline(1.0, color="gray", linewidth=0.5)
    ax.set_xlabel("T")
    ax.set_ylabel("E[S_T]")
    ax.set_title("exp_7: isotropic power-volatility model (Example 6.1)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/exp7_power_volatility.png", dpi=150)
    plt.close(fig)
    print(f"saved {RESULTS_DIR}/exp7_power_volatility.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    rows, feller_rows = run(full=args.full)
    save(rows, feller_rows)
    with open(f"{RESULTS_DIR}/exp7_summary.json", "w") as f:
        json.dump({"full": args.full, "n_rows": len(rows)}, f, indent=2)


if __name__ == "__main__":
    main()
