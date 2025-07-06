import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from util import prolongation_matrix, coarsen_matrix


def test_prolongation_coarsen_shapes_and_transpose():
    m = 5
    P = prolongation_matrix(m)
    R = coarsen_matrix(m)
    assert P.shape == (m, (m + 1) // 2)
    assert R.shape == (P.shape[1], m)
    # they should be transposes of each other
    assert np.allclose(P.todense().T, R.todense())

    x = np.arange(1, m + 1)
    y = R @ x
    expected = np.array([(x[0] + x[1]) / 2,
                         (x[2] + x[3]) / 2,
                         x[4]])
    assert np.allclose(y, expected)
