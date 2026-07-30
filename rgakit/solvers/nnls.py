"""
nnls.py
-------
Non-negative least squares solver (scipy wrapper).
"""

from scipy.optimize import nnls


def make_nnls():
    """Return the scipy NNLS solver directly."""
    return nnls
