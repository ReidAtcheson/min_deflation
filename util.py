import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import cg


def chebyshev_diagonal_spd(n: int, eps: float = 1e-8):
    """Return an SPD diagonal matrix with eigenvalues at mapped Chebyshev points.

    The diagonal entries are Chebyshev points on ``[-1, 1]`` linearly
    mapped to ``[eps, 1]``.  This yields a matrix with condition number
    ``1/eps``.

    Parameters
    ----------
    n : int
        Size of the matrix.
    eps : float, optional
        Lower bound for the eigenvalues. Must satisfy ``0 < eps <= 1``.

    Returns
    -------
    scipy.sparse.csr_matrix
        Sparse diagonal matrix of shape ``(n, n)``.
    """
    if not (0 < eps <= 1):
        raise ValueError("eps must be in (0, 1]")
    if n <= 0:
        raise ValueError("n must be positive")

    if n == 1:
        diag = np.array([1.0])
    else:
        k = np.arange(n)
        # Chebyshev points of the second kind including endpoints
        nodes = np.cos(np.pi * k / (n - 1))
        diag = 0.5 * (1 - eps) * nodes + 0.5 * (1 + eps)
    return diags(diag, offsets=0, format="csr")


def cg_residuals(eps: float, m: int, k: int):
    """Run CG on ``chebyshev_diagonal_spd(m, eps)`` and record residuals.

    Parameters
    ----------
    eps : float
        Lower bound for the eigenvalues used to construct ``A``.
    m : int
        Dimension of ``A`` and the right-hand side ``b``.
    k : int
        Number of leading entries of ``b`` to zero out.

    Returns
    -------
    list[float]
        Euclidean norms of the residual at each iteration of CG.
    """

    A = chebyshev_diagonal_spd(m, eps)
    b = np.ones(m)
    if k:
        b[:k] = 0.0
    b /= np.linalg.norm(b)

    residuals: list[float] = []

    def callback(xk: np.ndarray) -> None:
        r = b - A.dot(xk)
        residuals.append(np.linalg.norm(r))

    cg(A, b, atol=0.0, rtol=1e-14, callback=callback)
    return residuals
