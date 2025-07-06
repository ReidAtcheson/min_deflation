import argparse
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from util import cg_residuals


def iterations_to_threshold(residuals: list[float], thresh: float = 1e-10) -> int:
    """Return the iteration count needed to drop below ``thresh``."""
    for i, r in enumerate(residuals, start=1):
        if r < thresh:
            return i
    return len(residuals)


def run_experiment(samples: int, eps_exp_low: float, eps_exp_high: float,
                   m_exp_low: float, m_exp_high: float, seed: int | None) -> None:
    rng = np.random.default_rng(seed)
    log_eps = rng.uniform(eps_exp_low, eps_exp_high, size=samples)
    eps = 10.0 ** log_eps
    log_m = rng.uniform(m_exp_low, m_exp_high, size=samples)
    m = np.round(10.0 ** log_m).astype(int)
    # ensure m >= 1
    m = np.clip(m, 1, None)
    k = (rng.random(samples) * m).astype(int)

    targets = []
    features = []
    for e, mm, kk, le, lm in zip(eps, m, k, log_eps, log_m):
        res = cg_residuals(float(e), int(mm), int(kk))
        iters = iterations_to_threshold(res)
        targets.append(iters / mm)
        features.append([le, lm, kk / mm])

    X = np.array(features)
    y = np.array(targets)
    X_design = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X_design, y, rcond=None)

    names = ["intercept", "log_eps", "log_m", "k/m"]
    for name, b in zip(names, beta):
        print(f"{name}: {b}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CG sampling experiment and fit regression model")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--eps-exp-low", type=float, default=-5)
    parser.add_argument("--eps-exp-high", type=float, default=-1)
    parser.add_argument("--m-exp-low", type=float, default=1)
    parser.add_argument("--m-exp-high", type=float, default=5)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    run_experiment(args.samples, args.eps_exp_low, args.eps_exp_high,
                   args.m_exp_low, args.m_exp_high, args.seed)


if __name__ == "__main__":
    main()
