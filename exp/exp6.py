"""exp_6: Coupled logarithmic threshold (Theorem 3.2, Corollary 3.2).

Critical CIR factor dX = (1/2 - kappa*X) dt + sqrt(X) dW0 coupled to an OU
factor dZ = -lambda*Z dt + eta_z*dW1 (Corr(dW0, dW1) = rho1), driving a
stochastic exponential with integrand theta0 = -c(Z)*h(X)/sqrt(X),
c(z) = a + b*z^2/(1+z^2). Verifies E[S_T] ~ 1 for every finite T whenever
sup c = a + b <= 1/2 (is_martingale_coupled_log), across the EXP6_CASES
family and the EXP6_T_GRID horizons.

Analytic note: Theorem 3.2's condition (T) at the X = 0 boundary fails for
every c > 0 in this family; that fact is analytic (not numerically checked
here) and is exactly what the paper's refined criterion improves upon, since
S is nonetheless a true martingale whenever sup c <= 1/2.

Also implements the long-horizon illustration of Corollary 3.2: S is a true
martingale (E[S_T] = 1 for every finite T) but not uniformly integrable on
[0, infinity) whenever c is not identically zero, so S_t -> 0 almost surely
as t -> infinity; reported as the fraction of paths with S_T < 0.01 at large
T for the cases that are true martingales with sup c > 0 ("martingale" and
"critical").

Run: python -m exp.exp6 [--full]
"""
from __future__ import annotations

import argparse
import csv
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import config as cfg
from src.criteria import is_martingale_coupled_log, is_uniformly_integrable_coupled_log
from src.mc_core import make_rng, mc_mean_ci
from src.model_logarithmic import simulate_coupled_log_exponential

RESULTS_DIR = "results"


def run(full: bool = False) -> tuple[list[dict], list[dict]]:
    n_paths = cfg.EXP6_N_PATHS_PAPER if full else cfg.EXP6_N_PATHS_RUN
    n_steps_per_unit = cfg.EXP6_N_STEPS_PER_UNIT_PAPER if full else cfg.EXP6_N_STEPS_PER_UNIT_RUN
    rng = make_rng(cfg.SEED)

    # Main table: E[S_T] across cases and the finite-horizon T grid.
    rows = []
    for case, (a, b) in cfg.EXP6_CASES.items():
        sup_c = a + b
        predicted = is_martingale_coupled_log(sup_c)
        predicted_ui = is_uniformly_integrable_coupled_log(sup_c)
        for T in cfg.EXP6_T_GRID:
            n_steps = max(int(n_steps_per_unit * T), 1)
            s_t = simulate_coupled_log_exponential(
                a, b, cfg.EXP6_KAPPA, cfg.EXP6_LAMBDA, cfg.EXP6_ETA_Z, cfg.EXP6_RHO1,
                cfg.EXP6_X0, cfg.EXP6_Z0, cfg.EXP6_X_FLOOR, T, n_steps, n_paths, rng,
                chunk_size=cfg.PATHS_CHUNK,
            )
            e_s, ci = mc_mean_ci(s_t, cfg.CONFIDENCE)
            rows.append({
                "case": case, "a": a, "b": b, "sup_c": sup_c,
                "predicted_martingale": predicted, "predicted_ui": predicted_ui,
                "T": T, "E[S_T]": e_s, "E[S_T]_ci_halfwidth": ci,
                "n_paths": n_paths, "n_steps": n_steps,
            })
            print(f"{case:14s} sup_c={sup_c:+.2f} predicted_mart={predicted!s:5s} "
                  f"T={T:4.1f} E[S_T]={e_s:.4f}+-{ci:.4f} n_paths={n_paths} n_steps={n_steps}")

    # Long-horizon decay check (Corollary 3.2): true martingales with c != 0
    # still satisfy S_t -> 0 a.s. as t -> infinity (not uniformly integrable).
    long_n_paths = cfg.EXP6_LONG_N_PATHS_PAPER if full else cfg.EXP6_LONG_N_PATHS_RUN
    long_n_steps_per_unit = (
        cfg.EXP6_LONG_N_STEPS_PER_UNIT_PAPER if full else cfg.EXP6_LONG_N_STEPS_PER_UNIT_RUN
    )
    long_T_grid = cfg.EXP6_LONG_T_GRID_PAPER if full else cfg.EXP6_LONG_T_GRID_RUN

    long_rows = []
    for case, (a, b) in cfg.EXP6_CASES.items():
        sup_c = a + b
        if not is_martingale_coupled_log(sup_c) or sup_c == 0.0:
            continue  # restrict to true martingales with c != 0 (Corollary 3.2's premise)
        for T in long_T_grid:
            n_steps = max(int(long_n_steps_per_unit * T), 1)
            s_t = simulate_coupled_log_exponential(
                a, b, cfg.EXP6_KAPPA, cfg.EXP6_LAMBDA, cfg.EXP6_ETA_Z, cfg.EXP6_RHO1,
                cfg.EXP6_X0, cfg.EXP6_Z0, cfg.EXP6_X_FLOOR, T, n_steps, long_n_paths, rng,
                chunk_size=cfg.PATHS_CHUNK,
            )
            e_s, ci = mc_mean_ci(s_t, cfg.CONFIDENCE)
            frac_decayed = float((s_t < 0.01).mean())
            long_rows.append({
                "case": case, "sup_c": sup_c, "T": T,
                "E[S_T]": e_s, "E[S_T]_ci_halfwidth": ci,
                "frac_S_T_below_0.01": frac_decayed,
                "n_paths": long_n_paths, "n_steps": n_steps,
            })
            print(f"[long horizon] {case:14s} sup_c={sup_c:+.2f} T={T:5.1f} "
                  f"E[S_T]={e_s:.4f}+-{ci:.4f} frac(S_T<0.01)={frac_decayed:.4f}")

    return rows, long_rows


def save(rows: list[dict], long_rows: list[dict]) -> None:
    path = f"{RESULTS_DIR}/exp6_coupled_log.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {path}")

    long_path = f"{RESULTS_DIR}/exp6_long_horizon_decay.csv"
    with open(long_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(long_rows[0].keys()))
        writer.writeheader()
        writer.writerows(long_rows)
    print(f"saved {long_path}")

    fig, ax = plt.subplots(figsize=(7, 5))
    for case in sorted({r["case"] for r in rows}):
        sub = [r for r in rows if r["case"] == case]
        ts = [r["T"] for r in sub]
        es = [r["E[S_T]"] for r in sub]
        ax.plot(ts, es, "o-", label=case)
    ax.axhline(1.0, color="gray", linewidth=0.5)
    ax.set_xlabel("T")
    ax.set_ylabel("E[S_T]")
    ax.set_title("exp_6: coupled logarithmic threshold (Theorem 3.2)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/exp6_coupled_log.png", dpi=150)
    plt.close(fig)
    print(f"saved {RESULTS_DIR}/exp6_coupled_log.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    rows, long_rows = run(full=args.full)
    save(rows, long_rows)
    with open(f"{RESULTS_DIR}/exp6_summary.json", "w") as f:
        json.dump({"full": args.full, "n_rows": len(rows)}, f, indent=2)


if __name__ == "__main__":
    main()
