# AGENTS.md

Guide for coding agents and contributors working in this repository.

## What this repository implements

A Monte Carlo and finite-difference replication of selected numerical
verifications from Bayraktar, E. and Xing, H., *"Necessary and Sufficient
Conditions for the Martingale Property of Stochastic Exponentials Driven by
Multi-Dimensional Diffusions."* Each experiment (`exp_1` .. `exp_11`) checks
one analytic martingale criterion the paper derives from the Fichera drift /
boundary-behaviour analysis of an auxiliary diffusion against a Monte Carlo
(and, for `exp_11`, PDE) simulation of the same model. No market data or
external dataset is used anywhere in the repository: every input is a model
coefficient given in the paper's examples and theorems.

## Layout

```
src/        shared simulation primitives, analytic criteria and config
  config.py        every experiment's parameters as module-level UPPER_CASE
                   variables, each with a one-line source note (paper section
                   or "not specified in the paper")
  mc_core.py       RNG helpers, chunked Monte Carlo averaging, confidence
                   intervals (keeps peak memory bounded via PATHS_CHUNK)
  criteria.py      the paper's analytic necessary-and-sufficient conditions
                   (Fichera drift formulas) for every example, independent of
                   the Monte Carlo code
  feller_test.py   the radial Feller/Khasminskii test at infinity (exp_7)
  pde_solver.py    Crank-Nicolson solver for the loss-of-mass PDE (exp_11)
  model_*.py       one Euler-Maruyama (full-truncation) simulator per model
                   family (3/2, two-factor inverse-CIR, 4/2, correlated
                   square-root factors, coupled log, power volatility,
                   weighted distance, absorbed factor, Wishart)
exp/         one script per experiment, `python -m exp.expN [--full]`
tests/       pytest unit tests for src/ and smoke tests for every experiment
results/     CSV tables, PNG figures and RESULTS.md (the paper comparison)
data/        empty package: no experiment loads market or synthetic data
```

## Commands

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m pytest                       # full test suite
python -m pytest -m "not market_data"  # same: no market_data tests exist here

python -m exp.exp1                     # default (time-limited) configuration
python -m exp.exp1 --full              # paper-scale Monte Carlo sample sizes

bash reproduce.sh                      # runs the test suite, then exp_1..exp_11
```

## Conventions

* All experiment parameters live only in `src/config.py` as module-level
  `UPPER_CASE` variables (no classes, no dataclasses, no parameter dicts);
  each has an inline comment stating its meaning, unit and source.
* Every experiment keeps a `*_PAPER` and a `*_RUN` variant of its Monte Carlo
  sample size / step count; `--full` on any `exp/exp*.py` script selects the
  `*_PAPER` values, the default run uses `*_RUN`.
* Shared simulation and criteria code lives only in `src/`; experiment
  scripts in `exp/` call it through the signatures defined there and do not
  reimplement a simulator or a criterion locally.
* Tests run on small, fast Monte Carlo samples (a few hundred to a few
  thousand paths) and check the paper's qualitative properties (signs,
  thresholds, absence of look-ahead, monotonicity in `T`), not the exact
  numbers reported for the full run.
* No API keys are used anywhere in this repository (no market data).
* After rerunning any `exp/exp*.py` script, the files it writes under
  `results/` are overwritten; the numbers and verdicts in
  `results/RESULTS.md` must then be checked against the new files and
  updated by hand — they are not regenerated automatically.
