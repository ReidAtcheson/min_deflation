import argparse
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from util import cg_residuals


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot residuals vs processed floats for different deflation parameters"
    )
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--m", type=int, default=100000)
    parser.add_argument("--nnz-per-row", type=int, default=100)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).with_name("effective_residuals.svg"),
    )
    args = parser.parse_args()

    ks = [64, 128, 256, 512]

    plt.figure()

    base_cost = args.m * args.nnz_per_row + 2 * args.m
    for k in ks:
        residuals = cg_residuals(args.eps, args.m, k)
        cost_per_iter = base_cost + 2 * k * args.m
        xs = [(cost_per_iter * (i + 1))/(args.m*args.m*args.nnz_per_row) for i in range(len(residuals))]
        plt.semilogy(xs, residuals, label=f"k={100.0*(k / m)}%")

    plt.xlabel("floats processed")
    plt.ylabel("residual norm")
    plt.legend()
    plt.tight_layout()

    plt.savefig(args.out, format="svg")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
