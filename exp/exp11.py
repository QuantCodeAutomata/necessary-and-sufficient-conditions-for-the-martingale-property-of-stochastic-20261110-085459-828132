"""exp_11: PDE uniqueness and the valuation equation (Appendix B, Theorem B.1 /
Proposition B.1), 3/2 model of Example 6.3.

Rather than solving the full 2-D (s,y) valuation PDE for u(t,s,y) = E[S_T |
S_t=s, Y_t=y] with payoff phi(s,y) = s, this experiment exploits that phi is
affine in s: u(tau,s,y) = s*(1 - m(tau,y)) solves the 2-D PDE exactly whenever
the "loss of mass" m solves its own 1-D PDE driven by the Girsanov-shifted
generator (see src/pde_solver.py docstring for the full derivation). The PDE
solution m(T, y0) is cross-checked against an independent Monte Carlo
estimate of 1 - E_y0[S_T].

For each of a true-martingale case and a strict-local-martingale case (the
analytic criterion of src/criteria.is_martingale_3_2), this experiment:
  - solves the 1-D loss-of-mass PDE on a y-grid over [0, T_HORIZON];
  - reports m(T, y0) at a representative y0, together with u1 = y0 (the
    payoff itself) and u2 = y0*(1 - m) (the actual forward-price solution),
    whose difference y0*m is the "bubble" term of Theorem B.1;
  - cross-checks m(T, y0) against an independent Monte Carlo estimate of
    1 - E_y0[S_T];
  - saves the full m(T, .) curve for inspection.

Run: python -m exp.exp11 [--full]
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
from src.criteria import is_martingale_3_2
from src.mc_core import make_rng
from src.pde_solver import monte_carlo_loss_of_mass, solve_loss_of_mass

RESULTS_DIR = "results"

CASES = {
    "martingale": cfg.EXP11_ALPHA_MARTINGALE,
    "strict_local": cfg.EXP11_ALPHA_STRICT,
}


def _representative_y0(y_min: float, y_max: float) -> float:
    """Pick y0 = 1.0 if it lies in [y_min, y_max], else the grid midpoint."""
    if y_min <= 1.0 <= y_max:
        return 1.0
    return 0.5 * (y_min + y_max)


def run(full: bool = False) -> tuple[list[dict], dict[str, tuple[np.ndarray, np.ndarray]]]:
    y_min, y_max, n_points = cfg.EXP11_Y_GRID_PAPER if full else cfg.EXP11_Y_GRID_RUN
    n_time_steps = cfg.EXP11_N_TIME_STEPS_PAPER if full else cfg.EXP11_N_TIME_STEPS_RUN
    T = cfg.EXP11_T_HORIZON
    rng = make_rng(cfg.SEED)

    y_grid = np.linspace(y_min, y_max, n_points)
    y0 = _representative_y0(y_min, y_max)

    rows = []
    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for case, alpha in CASES.items():
        predicted = is_martingale_3_2(alpha, cfg.EXP11_ETA, cfg.EXP11_RHO)

        m = solve_loss_of_mass(y_grid, alpha, cfg.EXP11_KAPPA, cfg.EXP11_ETA, cfg.EXP11_RHO, T, n_time_steps)
        m_final = m[-1]
        curves[case] = (y_grid, m_final)

        m_pde = float(np.interp(y0, y_grid, m_final))

        mc_result = monte_carlo_loss_of_mass(
            y0, alpha, cfg.EXP11_KAPPA, cfg.EXP11_ETA, cfg.EXP11_RHO,
            [T], cfg.EXP11_MC_N_PATHS, cfg.EXP11_MC_N_STEPS, rng,
        )
        m_mc, m_mc_ci = mc_result[T]

        u1 = y0
        u2 = y0 * (1.0 - m_pde)
        bubble = y0 * m_pde

        rows.append({
            "case": case, "alpha": alpha, "predicted_martingale": predicted,
            "y0": y0, "T": T, "m_pde": m_pde, "m_mc": m_mc,
            "m_mc_ci_halfwidth": m_mc_ci, "u1": u1, "u2": u2, "bubble": bubble,
            "n_time_steps": n_time_steps, "n_y_points": n_points,
            "mc_n_paths": cfg.EXP11_MC_N_PATHS, "mc_n_steps": cfg.EXP11_MC_N_STEPS,
        })
        print(f"{case:13s} alpha={alpha:.2f} predicted_mart={predicted!s:5s} y0={y0:.3f} T={T:4.1f} "
              f"m_pde={m_pde:.4f} m_mc={m_mc:.4f}+-{m_mc_ci:.4f} u1={u1:.4f} u2={u2:.4f} bubble={bubble:.4f}")
    return rows, curves


def save(rows: list[dict], curves: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    path = f"{RESULTS_DIR}/exp11_pde_uniqueness.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {path}")

    for case, (y_grid, m_final) in curves.items():
        curve_path = f"{RESULTS_DIR}/exp11_loss_of_mass_curve_{case}.csv"
        with open(curve_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["y", "m"])
            writer.writerows(zip(y_grid.tolist(), m_final.tolist()))
        print(f"saved {curve_path}")

    fig, ax = plt.subplots(figsize=(7, 5))
    for case, (y_grid, m_final) in curves.items():
        ax.plot(y_grid, m_final, "-", label=case)
    ax.axhline(0.0, color="gray", linewidth=0.5)
    ax.set_xlabel("y")
    ax.set_ylabel("m(T, y)")
    ax.set_title("exp_11: loss-of-mass / bubble profile m(T, y) (Appendix B, Prop. B.1)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/exp11_pde_uniqueness.png", dpi=150)
    plt.close(fig)
    print(f"saved {RESULTS_DIR}/exp11_pde_uniqueness.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    rows, curves = run(full=args.full)
    save(rows, curves)
    with open(f"{RESULTS_DIR}/exp11_summary.json", "w") as f:
        json.dump({"full": args.full, "n_rows": len(rows)}, f, indent=2)


if __name__ == "__main__":
    main()
