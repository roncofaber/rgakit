"""
nnls.py
-------
Non-negative least squares solver (scipy wrapper).
"""

from scipy.optimize import nnls as _scipy_nnls


def solve_nnls(A, y):
    """
    NNLS with an iteration cap scaled to the problem size, so large active
    sets do not hit scipy's default ``maxiter = 3 * n_features`` and emit
    "Maximum number of iterations reached" warnings.
    """
    x, rnorm = _scipy_nnls(A, y, maxiter=max(1000, 10 * A.shape[1]))
    return x, rnorm


def make_nnls():
    """Return the NNLS solver callable ``(A, y) -> (weights, residual)``."""
    return solve_nnls
