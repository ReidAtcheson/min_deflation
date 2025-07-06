"""Plot residuals for different deflation parameters using CG."""

import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from util import cg_residuals


def main() -> None:
    eps = 1e-3
    m = 1024
    ks = [0, 1, 4, 16]

    plt.figure()
    for k in ks:
        res = cg_residuals(eps, m, k)
        plt.semilogy(res, label=f"k={k}")
    plt.xlabel("iteration")
    plt.ylabel("residual norm")
    plt.legend()
    plt.tight_layout()

    out_path = Path(__file__).with_name("residuals.svg")
    plt.savefig(out_path, format="svg")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
