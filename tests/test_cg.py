import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from util import cg_residuals


def test_cg_residuals_basic():
    eps = 0.5
    m = 32
    k = 2
    res = cg_residuals(eps, m, k)
    assert res, "No residuals returned"
    assert res[-1] < 1e-14
